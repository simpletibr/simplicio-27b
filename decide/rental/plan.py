"""The plan of the rented day: where everything lives and the exact command of every step (pure; nothing is run here).

``layout`` fixes the folders under ``--workdir``; ``*_command`` build the argv of ``decide.train`` (the trainer itself is never
imported by the orchestrator, so a plan can be printed on a laptop with no torch).  Paths and the seed are the orchestrator's;
the profile only chooses the trainer's options (``profile.train_argv``).
"""
from __future__ import annotations

import os
import shlex
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Mapping, Optional

from .profile import Profile

#: The first training seed (the trainer's default); seed ``n`` of the day is ``SEED_BASE + n - 1``.
SEED_BASE = 20261021
STAGED_MARK = ".staged-by-the-day"
WEIGHT_SUFFIX = ".safetensors"


@dataclass(frozen=True)
class Paths:
    workdir: Path
    corpus: Path
    labels: Path
    proxy: Path

    @property
    def state(self) -> Path:
        return self.workdir / "state"

    @property
    def logs(self) -> Path:
        return self.workdir / "logs"

    @property
    def data(self) -> Path:
        return self.workdir / "data"

    @property
    def start(self) -> Path:
        return self.workdir / "start"

    @property
    def targets(self) -> Path:
        return self.workdir / "targets"

    @property
    def targets_traces(self) -> Path:
        return self.workdir / "targets-traces"

    @property
    def traces(self) -> Path:
        return self.workdir / "traces"

    @property
    def runs(self) -> Path:
        return self.workdir / "runs"

    @property
    def soup(self) -> Path:
        return self.workdir / "soup"

    @property
    def final(self) -> Path:
        return self.workdir / "final"

    @property
    def calibrate(self) -> Path:
        return self.workdir / "calibrate"

    @property
    def g4(self) -> Path:
        return self.workdir / "g4"

    @property
    def public(self) -> Path:
        return self.workdir / "public"

    @property
    def report(self) -> Path:
        return self.workdir / "report"

    def seed_dir(self, index: int) -> Path:
        return self.runs / f"seed{index}"


def layout(workdir: "str | os.PathLike[str]", cfg: Mapping[str, Any]) -> Paths:
    root, inputs = Path(workdir), cfg["inputs"]
    data = root / "data"
    return Paths(workdir=root, corpus=data / inputs["corpus"], labels=data / inputs["labels"], proxy=data / inputs["proxy"])


def seeds(profile: Profile) -> List[int]:
    return [SEED_BASE + index for index in range(profile.seeds)]


def candidate(paths: Paths, seed_count: int) -> Path:
    """The checkpoint that gets calibrated: the only run's ``final``, or the soup of several."""
    return paths.seed_dir(1) / "final" if seed_count == 1 else paths.soup


# --------------------------------------------------------------------------- commands

def _python(python: Optional[str]) -> str:
    return python or sys.executable


def targets_command(paths: Paths, *, thinker: Optional[str] = None, python: Optional[str] = None) -> List[str]:
    argv = [_python(python), "-m", "decide.train.build_targets", "--corpus", str(paths.corpus), "--labels", str(paths.labels)]
    if thinker is not None:
        argv += ["--traces", str(paths.traces / thinker), "--thinker", thinker]       # the manifest of the traces is in the folder of the thinker
    return argv + ["--out", str(paths.targets_traces if thinker is not None else paths.targets)]


def train_command(profile: Profile, paths: Paths, index: int, *, with_traces: bool, resume: bool, python: Optional[str] = None) -> List[str]:
    head = [_python(python), "-m"]
    if profile.gpus > 1:
        head += ["torch.distributed.run", "--standalone", "--nproc-per-node", str(profile.gpus), "-m"]
    argv = head + ["decide.train.train", "--targets", str(paths.targets_traces if with_traces else paths.targets), "--checkpoint", str(paths.start),
                   "--out", str(paths.seed_dir(index)), "--seed", str(seeds(profile)[index - 1])]
    argv += profile.train_argv(with_traces=with_traces)
    if resume:
        argv.append("--resume")
    return argv


def soup_command(paths: Paths, count: int, *, python: Optional[str] = None) -> List[str]:
    return [_python(python), "-m", "decide.train.soup", "--inputs", *(str(paths.seed_dir(i) / "final") for i in range(1, count + 1)), "--out", str(paths.soup)]


def temperature_command(paths: Paths, *, targets: Path, python: Optional[str] = None) -> List[str]:
    return [_python(python), "-m", "decide.train.temperature", "--checkpoint", str(paths.final), "--targets", str(targets), "--write",
            "--report", str(paths.calibrate / "report.json")]


# --------------------------------------------------------------------------- staging

def stage(source: "str | os.PathLike[str]", destination: "str | os.PathLike[str]") -> Path:
    """A checkpoint to calibrate without touching the one it came from: weights are hard links, every other file is a copy.

    Only a folder this function made (or an empty one) is replaced; anything else is refused."""
    src, dst = Path(source), Path(destination)
    if dst.exists():
        if (dst / STAGED_MARK).is_file():
            shutil.rmtree(dst)
        elif dst.is_dir() and not any(dst.iterdir()):
            dst.rmdir()
        else:
            raise FileExistsError(f"{dst} exists and was not made by this tool, so it was left alone; move it away")
    dst.mkdir(parents=True)
    for item in sorted(src.rglob("*")):
        target = dst / item.relative_to(src)
        if item.is_dir():
            target.mkdir(parents=True, exist_ok=True)
        elif item.suffix == WEIGHT_SUFFIX:
            try:
                os.link(item, target)
            except OSError:
                shutil.copy2(item, target)           # another file system: a copy costs time and disk but the bytes are the same
        else:
            shutil.copyfile(item, target)
    (dst / STAGED_MARK).write_text(f"staged from {src}\n", encoding="utf-8")
    return dst


# --------------------------------------------------------------------------- the printed plan

def describe(profile: Profile, cfg: Mapping[str, Any], paths: Paths, *, thinker: Optional[str] = None, python: Optional[str] = None) -> List[str]:
    """The day, step by step, as text (what ``--dry-run`` prints).  Commands are shown with the executable shortened to ``python``."""
    def show(argv: List[str]) -> str:
        return shlex.join(["python" if argv[0] == _python(python) else argv[0], *argv[1:]])

    count = profile.seeds
    hub = cfg["hub"]
    lines = [f"1. fetch: {cfg['inputs']['corpus']}, {cfg['inputs']['labels']} and {cfg['inputs']['proxy']} from {hub['data_repo']}; "
             f"{cfg['start']['repo']}@{cfg['start']['revision'][:7]} into {paths.start}",
             f"2. targets: {show(targets_command(paths, python=python))}   (stops the day below {cfg['gates']['label_coverage']:.0%} of the corpus labeled)"]
    if thinker is None:
        lines.append("3. traces: skipped unless --gt-report is a passing report of the GT gate (thinker.enabled is auto)")
    else:
        t = cfg["thinker"]
        lines += [f"3. traces: {t['model']} thinking, {t['budget_per_question']} tokens per question (cap {t['max_tokens_cap']}) at "
                  f"{t.get('base_url') or 'thinker.base_url (not set)'}",
                  f"4. targets_traces: {show(targets_command(paths, thinker=thinker, python=python))}"]
    for index in range(1, count + 1):
        lines.append(f"5. train, seed {seeds(profile)[index - 1]}: {show(train_command(profile, paths, index, with_traces=thinker is not None, resume=False, python=python))}")
    lines.append(f"6. soup: {show(soup_command(paths, count, python=python))}" if count > 1 else "6. soup: one seed, nothing to average")
    lines.append(f"7. calibrate: stage {candidate(paths, count)} into {paths.final}, then "
                 f"{show(temperature_command(paths, targets=paths.targets_traces if thinker is not None else paths.targets, python=python))}")
    lines.append(f"8. g4: baseline and new checkpoint on the home private proxy; the day stops unless new >= baseline + {cfg['gates']['g4_margin_points']} points")
    lines.append("9. public_round: the full public requests file through the kit with the new checkpoint (needs --public-requests)")
    lines.append(f"10. publish: {paths.final} and the results to {hub['weights_repo']} (private)")
    return lines
