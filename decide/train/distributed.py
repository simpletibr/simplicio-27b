"""Single-node data parallelism (torchrun) around the training loop.

Adapted from ``autojev/distributed.py`` of perplexity-ai/pplx-decider-v1.1-27b (Apache License 2.0; see LICENSE,
NOTICE).  Changes: the process group is ``gloo`` when CUDA is not available (so the 2-process test runs on CPU);
``broadcast_parameters`` sends only the trainable parameters (with LoRA the frozen backbone never changes, and
broadcasting 54 GB every step would cost more than the step); no PEP 695 syntax (``def batches[T]``) so the module
imports on Python 3.10.

Only rank 0 owns the optimizer.  After its step it broadcasts the updated trainable parameters; gradients are
averaged by DDP.  The loss of each micro-batch is scaled by ``count * world_size / group_size`` so that ranks with
a different number of rows still produce the gradient of the mean over the whole group.
"""
from __future__ import annotations

import os
from datetime import timedelta
from typing import Any, Callable, List, Sequence, Tuple, TypeVar

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

T = TypeVar("T")


class DistributedTraining:
    def __init__(self, rank: int = 0, world_size: int = 1, device: str = "cuda") -> None:
        self.rank = rank
        self.world_size = world_size
        self.device = torch.device(device)

    @classmethod
    def initialize(cls) -> "DistributedTraining":
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        if world_size == 1:
            return cls(device="cuda" if torch.cuda.is_available() else "cpu")
        local_rank = int(os.environ["LOCAL_RANK"])
        if int(os.environ["LOCAL_WORLD_SIZE"]) != world_size or int(os.environ["RANK"]) != local_rank:
            raise ValueError("DDP training supports one node only")
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            dist.init_process_group("nccl", timeout=timedelta(hours=2))
            return cls(local_rank, world_size, f"cuda:{local_rank}")
        dist.init_process_group("gloo", timeout=timedelta(minutes=30))
        return cls(local_rank, world_size, "cpu")

    @property
    def primary(self) -> bool:
        return self.rank == 0

    def barrier(self) -> None:
        if self.world_size > 1:
            dist.barrier()

    def on_primary(self, operation: Callable[[], object]) -> None:
        if self.primary:
            operation()
        self.barrier()

    def broadcast(self, value: Any) -> Any:
        if self.world_size == 1:
            return value
        values = [value if self.primary else None]
        dist.broadcast_object_list(values, src=0)
        return values[0]

    def wrap(self, model: torch.nn.Module) -> torch.nn.Module:
        if self.world_size == 1:
            return model
        return DistributedDataParallel(
            model,
            device_ids=[self.device.index] if self.device.type == "cuda" else None,
            broadcast_buffers=False,
            find_unused_parameters=True,
            gradient_as_bucket_view=True,
        )

    def batches(self, rows: Sequence[T], split: Callable[[Sequence[T]], List[List[T]]]) -> List[Tuple[List[T], int]]:
        """Micro-batches of this rank as ``(rows, real_count)``; a rank with fewer batches pads with a dummy of count 0."""
        if self.world_size == 1:
            return [(batch, len(batch)) for batch in split(rows)]
        shards = [split(rows[rank::self.world_size]) for rank in range(self.world_size)]
        local = shards[self.rank]
        return [
            (local[index], len(local[index])) if index < len(local) else ([rows[0]], 0)
            for index in range(max(map(len, shards)))
        ]

    def totals(self, loss: float, tokens: int) -> Tuple[float, int]:
        if self.world_size == 1:
            return loss, tokens
        values = torch.tensor([loss, tokens], dtype=torch.float64, device=self.device)
        dist.all_reduce(values)
        return float(values[0]), int(values[1])

    @torch.no_grad()
    def broadcast_parameters(self, model: torch.nn.Module) -> None:
        if self.world_size > 1:
            for parameter in model.parameters():
                if parameter.requires_grad:
                    dist.broadcast(parameter, src=0)

    def close(self) -> None:
        if self.world_size > 1:
            dist.destroy_process_group()
