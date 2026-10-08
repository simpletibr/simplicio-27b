import json, math, os, sys, time, urllib.request
REPO, INPUT, OUT, MAX_TOK = sys.argv[1], sys.argv[2], sys.argv[3], 1024
sys.path.insert(0, REPO)
import evaluation_lib
os.makedirs(OUT, exist_ok=True)
resp_path = os.path.join(OUT, "responses.jsonl")
rows = evaluation_lib.read_prompt_list(INPUT)
done = set()
if os.path.exists(resp_path):
    done = {json.loads(l)["prompt"] for l in open(resp_path, encoding="utf-8")}
URL = "http://127.0.0.1:8080/v1/chat/completions"
for n, inp in enumerate(rows, 1):
    if inp.prompt in done: continue
    body = json.dumps({"model": "simplicio-27b", "temperature": 0, "max_tokens": MAX_TOK,
                       "chat_template_kwargs": {"enable_thinking": False},
                       "messages": [{"role": "user", "content": inp.prompt}]}).encode()
    t0 = time.time()
    try:
        d = json.load(urllib.request.urlopen(urllib.request.Request(URL, data=body, headers={"Content-Type": "application/json"}), timeout=7200))
        text = d["choices"][0]["message"]["content"] or ""; ct = d["usage"]["completion_tokens"]; fin = d["choices"][0]["finish_reason"]; err = None
    except Exception as e:
        text, ct, fin, err = "", None, None, type(e).__name__
    with open(resp_path, "a", encoding="utf-8") as f:
        f.write(json.dumps({"prompt": inp.prompt, "response": text, "key": inp.key, "completion_tokens": ct,
                            "finish_reason": fin, "latency_s": round(time.time()-t0, 1), "error": err}, ensure_ascii=False) + "\n")
    print(f"{n}/{len(rows)} tok={ct} fin={fin} lat={round(time.time()-t0,1)} err={err}", flush=True)
def wilson(k, n, z=1.959964):
    if n == 0: return [0.0, 0.0]
    p = k/n; d = 1+z*z/n; c = (p+z*z/(2*n))/d; h = z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return [round(max(0.0, c-h), 4), round(min(1.0, c+h), 4)]
prompt_to_response = evaluation_lib.read_prompt_to_response_dict(resp_path)
summary = {"benchmark": "IFBench test (subset, prompt-level)", "n": len(rows), "metrics": {},
           "config": {"model": "Simplicio 27B Q4_K_M", "sha256": "25371da44b53f5b3b6d8beb4208a2b39d545e557842beff22269bdc2e3bcd311",
                      "llama_cpp_commit": "a11f57b", "spec": "none", "temperature": 0, "max_tokens": MAX_TOK, "thinking": False,
                      "grader": "allenai/IFBench evaluation_lib (official)", "subset_seed": 3407}}
for name, func in [("strict", evaluation_lib.test_instruction_following_strict), ("loose", evaluation_lib.test_instruction_following_loose)]:
    outs = [func(inp, prompt_to_response) for inp in rows]
    k = sum(bool(o.follow_all_instructions) for o in outs)
    summary["metrics"][name] = {"prompt_level_correct": k, "prompt_level_accuracy": round(k/len(rows), 4), "wilson95": wilson(k, len(rows))}
json.dump(summary, open(os.path.join(OUT, "summary.json"), "w"), ensure_ascii=False, indent=2)
print("SUMMARY", json.dumps(summary, ensure_ascii=False))
