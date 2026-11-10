"""Put the rebuilt rows in the shape the study reads: the 0.3 edition filter and the exclusion list applied, the `state` dropped.

    python -I prepare_rows.py KIT_ROOT OUT_DIR ROWS.jsonl [ROWS.jsonl ...]

Writes OUT_DIR/slim-rows.jsonl (every field the kit's scorers read; no `state`) and OUT_DIR/index.jsonl (run id, payload hash,
and for each question its type and option keys in prompt order).
"""
import collections
import json
import sys
from pathlib import Path

kit, out_dir = sys.argv[1:3]
sources = sys.argv[3:]
sys.path.insert(0, kit)

from decision_index import editions  # noqa: E402

EDITION = "0.3"
keep = editions.in_edition(EDITION)
excluded = set(json.loads((Path(kit) / "hub" / "excluded-questions.json").read_text(encoding="utf-8"))["rows"])
out = Path(out_dir)
out.mkdir(parents=True, exist_ok=True)
counts, dropped = collections.Counter(), collections.Counter()
seen = set()
with (out / "slim-rows.jsonl").open("w", encoding="utf-8") as slim, (out / "index.jsonl").open("w", encoding="utf-8") as index:
    for source in sources:
        with open(source, encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                e = row["_evaluation"]
                if e["run_id"] in excluded:
                    dropped[(e["catalog_id"], "excluded question")] += 1
                    continue
                if not keep(e):
                    dropped[(e["catalog_id"], "not in edition 0.3")] += 1
                    continue
                assert e["run_id"] not in seen, e["run_id"]
                seen.add(e["run_id"])
                row.pop("state", None)
                slim.write(json.dumps(row, ensure_ascii=False) + "\n")
                questions = {key: {"type": q["type"], "options": None if q["type"] == "noul" else list(q["criteria"])} for key, q in row["questions"].items()}
                index.write(json.dumps({"run_id": e["run_id"], "catalog_id": e["catalog_id"], "payload_sha256": e["payload_sha256"], "questions": questions}, ensure_ascii=False) + "\n")
                counts[(e["catalog_id"], e["dataset"])] += 1
print(json.dumps({"kept": {f"{n} {name}": c for (n, name), c in sorted(counts.items())}, "dropped": {f"{n} {why}": c for (n, why), c in sorted(dropped.items())}}, indent=1))
