"""Pull, from one published results file, the answers to the requests of the gold file.

    python -I extract.py GOLD.jsonl RESULTS(.jsonl|.jsonl.gz) OUT.json NAME

For every gold request: the status of the published row, whether its payload hash equals the hash of the rebuilt
request (only then is it the same request), and for each scored question the probabilities in the gold's option order
(`noul`: [1 - p, p]) plus the answer the model itself reported (`choice`, or the noul probability).
"""
import gzip
import json
import sys

gold_path, results_path, out_path, name = sys.argv[1:5]

gold = {}
with open(gold_path, encoding="utf-8") as f:
    for line in f:
        record = json.loads(line)
        gold[record["run_id"]] = record

opener = gzip.open if results_path.endswith(".gz") else open
rows = {}
stats = {"lines": 0, "matched": 0, "duplicate": 0, "hash_equal": 0, "hash_differs": 0, "hash_missing": 0, "ok": 0, "bad_answer": 0}
statuses = {}
with opener(results_path, "rt", encoding="utf-8") as f:
    for line in f:
        if not line.strip():
            continue
        stats["lines"] += 1
        # cheap prefilter: the run id sits at the start of every line the kit writes
        head = line[:160]
        if '"candidates-v3:' not in head and '"calibration-v1:' not in head:
            continue
        row = json.loads(line)
        record = gold.get(row.get("run_id"))
        if record is None:
            continue
        if row["run_id"] in rows:
            stats["duplicate"] += 1
        stats["matched"] += 1
        status = row.get("status")
        statuses[status] = statuses.get(status, 0) + 1
        published_hash = row.get("payload_sha256")
        if published_hash is None:
            stats["hash_missing"] += 1
        elif published_hash == record["payload_sha256"]:
            stats["hash_equal"] += 1
        else:
            stats["hash_differs"] += 1
        entry = {"s": status, "h": published_hash == record["payload_sha256"]}
        if status == "ok":
            stats["ok"] += 1
            answers = (row.get("response") or {}).get("answers") or {}
            probs, said = {}, {}
            try:
                for key, question in record["questions"].items():
                    answer = answers[key]
                    if question["type"] == "noul":
                        p = float(answer["noul"])
                        probs[key] = [1.0 - p, p]
                        said[key] = p
                    else:
                        table = answer["probabilities"]
                        probs[key] = [float(table[option]) for option in question["options"]]
                        said[key] = answer["choice"]
                entry["p"], entry["c"] = probs, said
            except (KeyError, TypeError, ValueError):
                stats["bad_answer"] += 1
                entry["s"] = "bad_answer"
        rows[row["run_id"]] = entry

stats["gold"] = len(gold)
stats["missing"] = len(gold) - len(rows)
stats["statuses"] = statuses
with open(out_path, "w", encoding="utf-8") as f:
    json.dump({"name": name, "stats": stats, "rows": rows}, f)
print(name, json.dumps(stats))
