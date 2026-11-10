"""Pull, from one published results file, the answers to every request of the row index.

    python -I extract_ext.py INDEX.jsonl RESULTS(.jsonl|.jsonl.gz) OUT.json NAME

For every indexed request: the status of the published row, whether its payload hash equals the hash of the rebuilt request,
and for each question the probabilities in the index's option order (`noul`: the probability itself) plus the answer the
model itself reported.
"""
import gzip
import json
import sys

index_path, results_path, out_path, name = sys.argv[1:5]

index = {}
with open(index_path, encoding="utf-8") as f:
    for line in f:
        record = json.loads(line)
        index[record["run_id"]] = record

opener = gzip.open if results_path.endswith(".gz") else open
rows = {}
stats = {"lines": 0, "matched": 0, "hash_equal": 0, "hash_differs": 0, "hash_missing": 0, "ok": 0, "bad_answer": 0}
statuses, differs_by_benchmark = {}, {}
with opener(results_path, "rt", encoding="utf-8") as f:
    for line in f:
        if not line.strip():
            continue
        stats["lines"] += 1
        row = json.loads(line)
        record = index.get(row.get("run_id"))
        if record is None:
            continue
        stats["matched"] += 1
        status = row.get("status")
        statuses[status] = statuses.get(status, 0) + 1
        published_hash = row.get("payload_sha256")
        same = published_hash == record["payload_sha256"]
        if published_hash is None:
            stats["hash_missing"] += 1
        elif same:
            stats["hash_equal"] += 1
        else:
            stats["hash_differs"] += 1
            differs_by_benchmark[record["catalog_id"]] = differs_by_benchmark.get(record["catalog_id"], 0) + 1
        entry = {"s": status, "h": same}
        if status == "ok":
            answers = (row.get("response") or {}).get("answers") or {}
            probs, said = {}, {}
            try:
                for key, question in record["questions"].items():
                    answer = answers[key]
                    if question["type"] == "noul":
                        probs[key] = float(answer["noul"])
                        said[key] = probs[key]
                    else:
                        table = answer["probabilities"]
                        probs[key] = [float(table[option]) for option in question["options"]]
                        said[key] = answer["choice"]
                entry["p"], entry["c"] = probs, said
                stats["ok"] += 1
            except (KeyError, TypeError, ValueError):
                stats["bad_answer"] += 1
                entry["s"] = "bad_answer"
        rows[row["run_id"]] = entry

stats["indexed"] = len(index)
stats["missing"] = len(index) - len(rows)
stats["statuses"] = statuses
stats["hash_differs_by_benchmark"] = differs_by_benchmark
with open(out_path, "w", encoding="utf-8") as f:
    json.dump({"name": name, "stats": stats, "rows": rows}, f)
print(name, json.dumps(stats))
