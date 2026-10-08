import csv, json, math, os, random, re, sys, time, urllib.request
CSV, OUT, N, SEED = sys.argv[1], sys.argv[2], int(sys.argv[3]), 3407
URL = "http://127.0.0.1:8080/v1/chat/completions"
os.makedirs(OUT, exist_ok=True)
rows = list(csv.DictReader(open(CSV, encoding="utf-8")))
idx = list(range(len(rows))); random.Random(SEED).shuffle(idx); idx = sorted(idx[:N])
res_path = os.path.join(OUT, "results.jsonl")
done = set()
if os.path.exists(res_path):
    done = {json.loads(l)["row"] for l in open(res_path, encoding="utf-8")}
def build(row, i):
    opts = [row["Correct Answer"], row["Incorrect Answer 1"], row["Incorrect Answer 2"], row["Incorrect Answer 3"]]
    order = list(range(4)); random.Random(SEED * 1000 + i).shuffle(order)
    letters = "ABCD"; shown = [opts[j] for j in order]
    correct = letters[order.index(0)]
    prompt = ("Answer the following multiple choice question. Reply with only the letter of the correct option (A, B, C or D).\n\n"
              f"Question: {row['Question']}\n" + "".join(f"{letters[k]}) {shown[k]}\n" for k in range(4)))
    return prompt, correct
def parse(text):
    m = re.search(r"ANSWER\W*([ABCD])\b", text, re.I)
    if m: return m.group(1).upper()
    found = re.findall(r"\b([ABCD])\b", text)
    return found[-1] if found else None
def wilson(k, n, z=1.959964):
    if n == 0: return (0.0, 0.0)
    p = k / n; d = 1 + z*z/n; c = (p + z*z/(2*n)) / d; h = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / d
    return (max(0.0, c-h), min(1.0, c+h))
for i, row_i in enumerate(idx):
    if row_i in done: continue
    prompt, correct = build(rows[row_i], row_i)
    body = json.dumps({"model": "simplicio-27b", "temperature": 0, "max_tokens": 768,
                       "chat_template_kwargs": {"enable_thinking": False},
                       "messages": [{"role": "user", "content": prompt}]}).encode()
    t0 = time.time()
    req = urllib.request.Request(URL, data=body, headers={"Content-Type": "application/json"})
    try:
        d = json.load(urllib.request.urlopen(req, timeout=7200))
        text = d["choices"][0]["message"]["content"] or ""
        fin = d["choices"][0].get("finish_reason"); ctoks = d.get("usage", {}).get("completion_tokens")
        ptoks = d.get("usage", {}).get("prompt_tokens")
        err = None
    except Exception as e:
        text, fin, ctoks, ptoks, err = "", None, None, None, type(e).__name__
    pred = parse(text)
    rec = {"row": row_i, "correct": correct, "pred": pred, "ok": pred == correct, "finish_reason": fin,
           "prompt_tokens": ptoks, "completion_tokens": ctoks, "latency_s": round(time.time()-t0, 1), "error": err, "raw": text}
    with open(res_path, "a", encoding="utf-8") as f: f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"{len(done)+1}/{N} ok={rec['ok']} pred={pred} tok={ctoks} lat={rec['latency_s']}s err={err}", flush=True)
    done.add(row_i)
recs = [json.loads(l) for l in open(res_path, encoding="utf-8")]
k = sum(r["ok"] for r in recs); lo, hi = wilson(k, len(recs))
summary = {"benchmark": "GPQA Diamond (subset)", "n": len(recs), "correct": k, "accuracy": round(k/len(recs), 4),
           "wilson95": [round(lo, 4), round(hi, 4)], "unparsed": sum(r["pred"] is None for r in recs),
           "errors": sum(r["error"] is not None for r in recs), "mean_completion_tokens": round(sum((r["completion_tokens"] or 0) for r in recs)/len(recs), 1),
           "config": {"model": "Simplicio 27B Q4_K_M", "sha256": "25371da44b53f5b3b6d8beb4208a2b39d545e557842beff22269bdc2e3bcd311",
                      "llama_cpp_commit": "a11f57b", "spec": "none", "temperature": 0, "max_tokens": 768, "thinking": False,
                      "subset_seed": SEED, "subset_size": N, "option_shuffle_seed_base": SEED*1000}}
json.dump(summary, open(os.path.join(OUT, "summary.json"), "w"), ensure_ascii=False, indent=2)
print("SUMMARY", json.dumps(summary, ensure_ascii=False))
