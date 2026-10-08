import json, math, os, re, sys, time, urllib.request
INPUT, OUT = sys.argv[1], sys.argv[2]
os.makedirs(OUT, exist_ok=True)
rows = [json.loads(l) for l in open(INPUT, encoding="utf-8")]
res_path = os.path.join(OUT, "results.jsonl")
done = {json.loads(l)["id"] for l in open(res_path, encoding="utf-8")} if os.path.exists(res_path) else set()
URL = "http://127.0.0.1:8080/v1/chat/completions"
def parse(text):
    m = re.search(r"ANSWER\W*([A-Z])\b", text, re.I)
    if m: return m.group(1).upper()
    found = re.findall(r"\b([A-Z])\b", text)
    return found[-1] if found else None
for n, r in enumerate(rows, 1):
    if r["id"] in done: continue
    prompt = r["question"].rstrip() + "\n\nReply with only the letter of the correct option."
    body = json.dumps({"model": "simplicio-27b", "temperature": 0, "max_tokens": 768,
                       "chat_template_kwargs": {"enable_thinking": False},
                       "messages": [{"role": "user", "content": prompt}]}).encode()
    t0 = time.time()
    try:
        d = json.load(urllib.request.urlopen(urllib.request.Request(URL, data=body, headers={"Content-Type": "application/json"}), timeout=7200))
        text = d["choices"][0]["message"]["content"] or ""; ct = d["usage"]["completion_tokens"]; fin = d["choices"][0]["finish_reason"]; err = None
    except Exception as e:
        text, ct, fin, err = "", None, None, type(e).__name__
    gold = r["answer"].strip().upper()
    pred = parse(text)
    rec = {"id": r["id"], "gold": gold, "pred": pred, "ok": pred == gold, "finish_reason": fin, "completion_tokens": ct,
           "latency_s": round(time.time()-t0, 1), "error": err, "raw": text}
    with open(res_path, "a", encoding="utf-8") as f: f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"{n}/{len(rows)} ok={rec['ok']} pred={pred} tok={ct} err={err}", flush=True)
def wilson(k, n, z=1.959964):
    if n == 0: return [0.0, 0.0]
    p = k/n; d = 1+z*z/n; c = (p+z*z/(2*n))/d; h = z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return [round(max(0.0, c-h), 4), round(min(1.0, c+h), 4)]
recs = [json.loads(l) for l in open(res_path, encoding="utf-8")]
k = sum(r["ok"] for r in recs)
summary = {"benchmark": "HLE multipleChoice, text-only (subset, no judge)", "n": len(recs), "correct": k,
           "accuracy": round(k/len(recs), 4), "wilson95": wilson(k, len(recs)), "unparsed": sum(r["pred"] is None for r in recs),
           "errors": sum(r["error"] is not None for r in recs),
           "config": {"model": "Simplicio 27B Q4_K_M", "sha256": "25371da44b53f5b3b6d8beb4208a2b39d545e557842beff22269bdc2e3bcd311",
                      "llama_cpp_commit": "a11f57b", "spec": "none", "temperature": 0, "max_tokens": 768, "thinking": False, "subset_seed": 3407}}
json.dump(summary, open(os.path.join(OUT, "summary.json"), "w"), ensure_ascii=False, indent=2)
print("SUMMARY", json.dumps(summary, ensure_ascii=False))
