"""The report of the day: what each phase did, how long it took (measured), what the estimator said (UNVERIFIED), and the gates.

Only numbers the machine measured are presented as measurements.  The cost is the measured seconds of the phases times the
number of GPUs times the list price of one GPU-second; it is the cost of the phases that ran, not the invoice (the invoice also
bills the idle time between them).
"""
from __future__ import annotations

from typing import Any, Dict, Mapping, Optional

from . import hardware as HW
from .profile import Profile
from .state import PHASES, DayState

#: The day's phases that have a line in the estimator (the others are CPU-only bookkeeping).
ESTIMATED_AS = {"fetch": "setup", "traces": "traces", "train": "train", "soup": "soup", "calibrate": "calibrate", "g4": "g4",
                "public_round": "public_round", "publish": "publish"}


def day_report(state: DayState, estimate_report: Mapping[str, Any], *, profile: Profile, gates: Optional[Mapping[str, Any]] = None,
               notes: Optional[list] = None) -> Dict[str, Any]:
    estimates = {phase["name"]: phase for phase in estimate_report["phases"]}
    price = HW.gpu(profile.gpu).usd_per_second
    phases: Dict[str, Any] = {}
    measured_seconds = 0.0
    for name in PHASES:
        record = state.record(name)
        seconds = record.get("seconds")
        measured_seconds += seconds or 0.0
        guess = estimates.get(ESTIMATED_AS.get(name, ""))
        phases[name] = {"status": record["status"], "seconds_measured": seconds, "reason": record.get("reason"), "detail": record.get("detail"),
                        "estimate": None if guess is None else {"status": "UNVERIFIED", "seconds": guess["seconds"], "usd": guess["usd"], "basis": guess["basis"]}}
    return {
        "profile": profile.name,
        "phases": phases,
        "measured": {"seconds": measured_seconds, "usd": measured_seconds * profile.gpus * price,
                     "basis": f"measured seconds of the phases x {profile.gpus} x {profile.gpu} at the list price {price * 3600:.2f} USD per GPU-hour ({HW.PRICE_SOURCE}); "
                              "not the invoice, which also bills the idle time between phases"},
        "estimated": {"status": "UNVERIFIED", "totals": estimate_report["totals"]},
        "gates": dict(gates or {}),
        "notes": list(notes or []),
    }


def render(report: Mapping[str, Any]) -> str:
    lines = [f"day report, profile {report['profile']}"]
    for name, phase in report["phases"].items():
        measured = "-" if phase["seconds_measured"] is None else f"{phase['seconds_measured'] / 3600:.2f} h measured"
        guess = "" if not phase["estimate"] else f"; estimate {phase['estimate']['seconds'] / 3600:.2f} h UNVERIFIED" if phase["estimate"]["seconds"] is not None else "; no estimate"
        reason = f" ({phase['reason']})" if phase.get("reason") else ""
        lines.append(f"  {name}: {phase['status']}{reason}, {measured}{guess}")
    measured = report["measured"]
    lines.append(f"measured so far: {measured['seconds'] / 3600:.2f} h, {measured['usd']:.2f} USD at list price ({measured['basis']})")
    for name, gate in report["gates"].items():
        lines.append(f"  gate {name}: {gate}")
    lines += [f"note: {note}" for note in report["notes"]]
    return "\n".join(lines)
