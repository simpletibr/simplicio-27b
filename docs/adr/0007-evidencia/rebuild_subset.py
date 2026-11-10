"""Rebuild a subset of the base (release-v1) benchmarks of the Decision Index suite with the kit's own pipeline.

    python -I rebuild_subset.py KIT_ROOT WORK_ROOT GROUP [GROUP ...]

A GROUP is a comma-separated list of catalog ids that one kit builder makes together (for example 24,26,27,28,29).
Each group is acquired and normalized on its own, so one source that fails does not stop the others; `freeze` then
selects the rows of whatever was built and gives each request its run id and payload hash.
"""
import json
import sys
import traceback

kit, work = sys.argv[1:3]
groups = [[int(x) for x in group.split(",")] for group in sys.argv[3:]]
sys.path.insert(0, kit)

from decision_index.suite.build import acquire as acq  # noqa: E402
from decision_index.suite.build import freeze, rebuild  # noqa: E402
from decision_index.suite.build.layout import DATASETS, Layout  # noqa: E402

layout = Layout(work)
built, failed = [], {}
for group in groups:
    label = "+".join(DATASETS[n] for n in group)
    try:
        acq.acquire(layout, group, log=lambda message: print("   ", message, flush=True))
        rebuild.normalize(layout, group, log=lambda message: None)
        built.extend(group)
        print("OK  ", group, label, flush=True)
    except Exception as error:                                    # one source failing must not stop the rest
        failed[label] = f"{type(error).__name__}: {str(error)[:400]}"
        print("FAIL", group, label, failed[label], flush=True)
        traceback.print_exc(limit=2)

out = layout.suite / "release-v1-rebuilt"
manifest = freeze.freeze(layout, out, log=lambda message: None)
summary = {"built": built, "failed": failed, "rows_file": str(out / "selected-rows.jsonl"),
           "benchmarks": {b["dataset"]: {"catalog_id": b["catalog_id"], "requests": b["requests"], "cases": b["selected_cases"]} for b in manifest["benchmarks"]}}
print(json.dumps(summary, indent=1))
