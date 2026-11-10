"""Gold rows for the 'added' benchmarks of the Decision Index suite, rebuilt from their pinned public sources.

The requests and the expected answers come from the kit's own adapters (decision_index.suite.build.adapters_added);
this script only walks them and writes one compact line per request: the run id, the payload hash the kit would
give the request (so a published result can be shown to answer the SAME request), the scored questions and the gold.

    python -I build_gold.py KIT_ROOT WORK_ROOT OUT.jsonl

HoVer (61) is left out: it needs a 2.2 GB Wikipedia database.
"""
import hashlib
import json
import sys

kit, work, out = sys.argv[1:4]
sys.path.insert(0, kit)

from decision_index.scoring import added as S  # noqa: E402
from decision_index.suite.build import adapters_added as A  # noqa: E402
from decision_index.suite.build.layout import Layout  # noqa: E402

ORDER = (57, 58, 62, 64, 59, 56)

layout = Layout(work)
counts = {}
with open(out, "w", encoding="utf-8") as f:
    for n in ORDER:
        count = 0
        for row in A.BUILDERS[n](layout):
            payload = {"model": A.MODEL, "state": row["state"], "questions": row["questions"]}
            digest = hashlib.sha256(A.dumps(payload).encode()).hexdigest()
            prefix = "calibration-v1:" if n == 56 else "candidates-v3:"
            keys = S.scored_keys(n, row)
            questions = {}
            for key in keys:
                q = row["questions"][key]
                questions[key] = {"type": q["type"], "options": ["false", "true"] if q["type"] == "noul" else list(q["criteria"])}
            record = {
                "run_id": prefix + row["id"],
                "catalog_id": n,
                "track": row["metadata"]["track"],
                "group_id": row["metadata"].get("group") or row["metadata"]["group_id"],
                "payload_sha256": digest,
                "questions": questions,
                "expected": {key: row["expected"][key] for key in keys},
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
        counts[A.SPECS[n]] = count
print(json.dumps(counts))
