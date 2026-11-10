"""Full rows (the kit's row format, with `_evaluation`) of the 'added' benchmarks, rebuilt from their pinned sources.

    python -I build_added_rows.py KIT_ROOT WORK_ROOT OUT.jsonl

The same adapters as the kit's `adapters_added.build`; the only difference is `proxy_tokens`, which this script sets to 0
(the kit counts it with tiktoken, which would download an encoding file; no score depends on it).  HoVer (61) is left out.
"""
import hashlib
import json
import sys

kit, work, out = sys.argv[1:4]
sys.path.insert(0, kit)

from decision_index.suite.build import adapters_added as A  # noqa: E402
from decision_index.suite.build.layout import Layout  # noqa: E402

ORDER = (57, 58, 62, 64, 59, 56)
layout = Layout(work)
counts = {}
with open(out, "w", encoding="utf-8", newline="\n") as f:
    for n in ORDER:
        count = 0
        for row in A.BUILDERS[n](layout):
            payload = {"model": A.MODEL, "state": row["state"], "questions": row["questions"]}
            digest = hashlib.sha256(A.dumps(payload).encode()).hexdigest()
            prefix = "calibration-v1:" if n == 56 else "candidates-v3:"
            group = row["metadata"]["group_id"] if n == 56 else (row["metadata"].pop("group", None) or row["metadata"]["group_id"])
            row["_evaluation"] = dict(run_id=prefix + row["id"], catalog_id=n, dataset=row["benchmark"], track=row["metadata"]["track"], group_id=group,
                                      payload_sha256=digest, benchmark_origin="Original-task benchmarks", proxy_tokens=0)
            f.write(A.dumps(row) + "\n")
            count += 1
        counts[A.SPECS[n]] = count
print(json.dumps(counts))
