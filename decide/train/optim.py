"""AdamW with fp32 master weights and moments kept on the CPU.

Adapted from ``autojev/optim.py`` of perplexity-ai/pplx-decider-v1.1-27b (Apache License 2.0; see LICENSE, NOTICE).
Changes: ``typing`` aliases instead of builtin generics and ``TypedDict``/``cast`` imports kept to what Python 3.10
has; the fused AdamW falls back to the plain one when the fused kernel is not available; ``pin_memory`` is only
asked for when a parameter is on a CUDA device (so the class also runs on CPU-only machines and in the tests).

The model's own parameters stay in their training dtype (bf16 on GPU).  After backward, each gradient is copied to
the CPU as fp32, AdamW updates the fp32 master copy there, and the result is copied back.  This is what lets a full
fine-tune of a 27B model keep its optimizer state out of GPU memory; with LoRA the same code simply has a few
parameters.
"""
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Tuple, TypedDict, cast

import torch


class ParameterGroup(TypedDict):
    params: List[torch.Tensor]
    lr: float


class OffloadState(TypedDict):
    format_version: int
    parameter_names: List[str]
    masters: List[torch.Tensor]
    optimizer: Dict[str, Any]


class CPUOffloadAdamW:
    def __init__(
        self,
        named_parameters: Iterable[Tuple[str, torch.nn.Parameter]],
        lr: float,
        weight_decay: float = 0.01,
    ) -> None:
        parameters = [(name, parameter) for name, parameter in named_parameters if parameter.requires_grad]
        if not parameters:
            raise ValueError("The optimizer needs trainable parameters.")
        self.names = [name for name, _ in parameters]
        if len(set(self.names)) != len(self.names):
            raise ValueError("Parameter names must be unique.")
        self.parameters = [parameter for _, parameter in parameters]
        self.masters = [
            torch.nn.Parameter(parameter.detach().to(device="cpu", dtype=torch.float32, copy=True))
            for parameter in self.parameters
        ]
        self.gradients = [torch.empty_like(master) for master in self.masters]
        pin = any(parameter.is_cuda for parameter in self.parameters)
        self.buffers = {
            dtype: torch.empty(
                max(parameter.numel() for parameter in self.parameters if parameter.dtype == dtype),
                dtype=dtype,
                device="cpu",
                pin_memory=pin,
            )
            for dtype in {parameter.dtype for parameter in self.parameters}
        }
        try:
            self.optimizer = torch.optim.AdamW(self.masters, lr=lr, weight_decay=weight_decay, fused=True)
        except (RuntimeError, ValueError):  # no fused kernel for this build/device
            self.optimizer = torch.optim.AdamW(self.masters, lr=lr, weight_decay=weight_decay)

    @property
    def param_groups(self) -> List[ParameterGroup]:
        return cast(List[ParameterGroup], self.optimizer.param_groups)

    def zero_grad(self, set_to_none: bool = True) -> None:
        self.optimizer.zero_grad(set_to_none=set_to_none)
        for parameter in self.parameters:
            if set_to_none:
                parameter.grad = None
            elif parameter.grad is not None:
                parameter.grad.zero_()

    @torch.no_grad()
    def step(self) -> None:
        for parameter, master, gradient in zip(self.parameters, self.masters, self.gradients):
            if parameter.grad is None:
                master.grad = None
                continue
            buffer = self.buffers[parameter.dtype][: parameter.numel()].view_as(parameter)
            buffer.copy_(parameter.grad, non_blocking=True)
            if parameter.is_cuda:
                torch.cuda.synchronize(parameter.device)
            gradient.copy_(buffer)
            master.grad = gradient
            parameter.grad = None
        self.optimizer.step()
        for parameter, master in zip(self.parameters, self.masters):
            if master.grad is not None:
                buffer = self.buffers[parameter.dtype][: parameter.numel()].view_as(parameter)
                buffer.copy_(master)
                parameter.copy_(buffer, non_blocking=True)
                if parameter.is_cuda:
                    torch.cuda.synchronize(parameter.device)
                master.grad = None

    def state_dict(self) -> OffloadState:
        return {
            "format_version": 1,
            "parameter_names": list(self.names),
            "masters": [master.detach() for master in self.masters],
            "optimizer": cast(Dict[str, Any], self.optimizer.state_dict()),
        }

    @torch.no_grad()
    def load_state_dict(self, state: OffloadState) -> None:
        if state["format_version"] != 1 or state["parameter_names"] != self.names:
            raise ValueError("Optimizer checkpoint does not match model parameter names and order.")
        saved_masters = state["masters"]
        if len(saved_masters) != len(self.masters):
            raise ValueError("Optimizer checkpoint has the wrong number of master tensors.")
        for master, saved in zip(self.masters, saved_masters):
            if saved.device.type != "cpu" or saved.dtype != torch.float32 or saved.shape != master.shape:
                raise ValueError("Master tensors must be CPU FP32 with matching parameter shapes.")
        self.optimizer.load_state_dict(state["optimizer"])
        for parameter, master, saved in zip(self.parameters, self.masters, saved_masters):
            master.copy_(saved)
            parameter.copy_(master)
        for parameter in self.parameters:
            if parameter.is_cuda:
                torch.cuda.synchronize(parameter.device)
                break
        self.zero_grad()
