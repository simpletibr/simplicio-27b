"""The few names of the scoring kit that the engine and the runner need, with stand-ins when the kit is absent.

`Engine`, `Unsupported`, `NativeAbstention` and `validate` come from `decision_index.engines.base` when the kit is
installed, so a response accepted here is accepted by the kit's own runner.  Without the kit the stand-ins below
apply the same rules (the same checks, written out).  Nothing here imports torch: the runner and the report modules
must work in an environment that has nothing but the standard library.
"""
from __future__ import annotations

import math
from typing import Any, Dict, Mapping

try:
    from decision_index.engines.base import Engine, NativeAbstention, Unsupported, validate
    KIT_ENGINES = True
except ImportError:                                          # the scoring kit is optional for loading, labeling and reports
    KIT_ENGINES = False

    class Unsupported(ValueError):                           # type: ignore[no-redef]
        """The engine cannot answer this request (not a failure of the run)."""

    class NativeAbstention(Exception):                       # type: ignore[no-redef]
        def __init__(self, raw: Any) -> None:
            self.raw = raw
            super().__init__("Native model selected its additional abstention outcome")

    def validate(questions: Mapping[str, Any], response: Mapping[str, Any]) -> None:     # type: ignore[no-redef]
        if set(response.get("answers", {})) != set(questions):
            raise ValueError("Question keys mismatch")
        for key, question in questions.items():
            answer = response["answers"][key]
            if answer.get("type") != question["type"]:
                raise ValueError("Type mismatch")
            if question["type"] == "choice":
                if answer.get("choice") not in question["criteria"]:
                    raise ValueError("Invalid chosen answer")
                probs = answer.get("probabilities", {})
                if (set(probs) != set(question["criteria"]) or any(not math.isfinite(v) or not 0 <= v <= 1 for v in probs.values())
                        or abs(sum(probs.values()) - 1) > 0.01):
                    raise ValueError("Incomplete/invalid probability distribution")
            elif question["type"] == "noul":
                if not math.isfinite(answer["noul"]) or not 0 <= answer["noul"] <= 1:
                    raise ValueError("Invalid noul probability")
            else:
                raise ValueError("Unsupported question type " + str(question["type"]))

    class Engine:                                            # type: ignore[no-redef]
        name = "engine"
        provenance: Dict[str, Any] = {}
        latency = "In-process request wall time including prompt construction; excludes model loading."

        def __init__(self, **options: Any) -> None:
            self.options = options

        def warmup(self) -> None:
            pass

        def runtime(self) -> Dict[str, Any]:
            return {}

        def synchronize(self) -> None:
            pass

        def close(self) -> None:
            pass


__all__ = ["Engine", "NativeAbstention", "Unsupported", "validate", "KIT_ENGINES"]
