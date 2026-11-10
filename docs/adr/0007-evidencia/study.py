"""Cross-family ensemble study on six public Decision Index benchmarks rebuilt from their sources.

    python -I study.py GOLD.jsonl EXT_DIR RESULTS_DIR BOARD_INDEX.json OUT.json

What it measures, with published per-row probabilities only (no GPU, no model is run):
  1. validation: the score of every single model on every benchmark against the run's own benchmark-summary.json,
     and score and skill against the board;
  2. every pair of the 12 single models, probabilities averaged 50/50 (the REAX 1.3 rule), against its best member,
     also with each benchmark left out in turn;
  3. diversity: how often two models disagree, inside a family and across families;
  4. every pair again with ONE weight fitted on one half of the requests and measured on the other half;
  5. subsets of 1 to 4 models chosen on one half and measured on the other;
  6. paired bootstrap intervals (requests resampled inside each benchmark) for the headline comparisons;
  7. what the board itself says about REAX 1.3 against the public models it descends from, on the private parts.

The aggregate ("mini-index") uses the board's own area weights over the six benchmarks: knowledge = mean(MMLU-Pro, BBH),
language = RAGTruth, retrieval = PhishNChips, tools = When2Call, arts = New Yorker.  Each benchmark is chance-corrected
with the board's baseline first.  It is NOT the public index (37 benchmarks) and says nothing certain about the Full score.
"""
import hashlib
import itertools
import json
import os
import sys

import numpy as np

gold_path, ext_dir, results_dir, board_path, out_path = sys.argv[1:6]

BENCH = (57, 58, 59, 56, 62, 64)
SHORT = {56: "PhishNChips", 57: "MMLU-Pro", 58: "BBH", 59: "RAGTruth", 62: "When2Call", 64: "New Yorker"}
AREA_W = {"knowledge": 0.258494, "language": 0.258494, "retrieval": 0.200229, "tools": 0.182783, "arts": 0.1}
AREA_OF = {57: "knowledge", 58: "knowledge", 59: "language", 56: "retrieval", 62: "tools", 64: "arts"}
#: name -> (family, board name, licence of the weights as read on the Hub, 2026-11-10)
MODELS = {
    "glide": ("Q", "GLiDE 27B", "fastino-community"),
    "d3-ultra": ("Q", "vLLM-SR d3-ultra (Decision 3.0, 27B)", "apache-2.0"),
    "nimble": ("Q", "Bespoke Nimble v4.1 Max (27B)", "cc-by-nc-4.0"),
    "d3-turbo": ("Q", "vLLM-SR d3-turbo v1.1 (27B)", "apache-2.0"),
    "d3": ("Q", "vLLM-SR d3 (27B)", "apache-2.0"),
    "knowline": ("G", "KnowLine 31B Gen1.5", "apache-2.0"),
    "flexdm": ("G", "FlexDM-LoRA 2.1 · Gemma-4-31B", "cc-by-sa-4.0"),
    "sieve": ("G", "Sieve 31B v3", "apache-2.0"),
    "turion": ("G", "Turion Jev 31B", "apache-2.0"),
    "h2o": ("G", "H2O Lightning 31B v1.3.0", "other"),
    "jeffan": ("G", "Jeffan-31b-1710", "apache-2.0"),
    "liquamen": ("G", "liquamen (Gemma 4 31B)", "apache-2.0"),
}
REFERENCE = {"reax": ("X", "REAX 1.3", "apache-2.0")}
SINGLES = list(MODELS)
APACHE = [name for name, (_, _, lic) in MODELS.items() if lic == "apache-2.0"]
SEED = 20261110
RESAMPLES = 2000
GRID = [round(0.1 * i, 1) for i in range(11)]

# --------------------------------------------------------------------------- gold
gold = [json.loads(line) for line in open(gold_path, encoding="utf-8")]
N = len(gold)
assert all(len(g["questions"]) == 1 for g in gold), "every request of these six benchmarks has one scored question"
KEY = [next(iter(g["questions"])) for g in gold]
KMAX = max(len(g["questions"][KEY[i]]["options"]) for i, g in enumerate(gold))
bench = np.array([g["catalog_id"] for g in gold])
k_options = np.array([len(g["questions"][KEY[i]]["options"]) for i, g in enumerate(gold)])


def gold_index(i):
    question, expected = gold[i]["questions"][KEY[i]], gold[i]["expected"][KEY[i]]
    return (1 if expected else 0) if question["type"] == "noul" else question["options"].index(expected)


gidx = np.array([gold_index(i) for i in range(N)])
half = np.array([int(hashlib.sha256(g["run_id"].encode()).hexdigest(), 16) % 2 for g in gold])
HALF = {0: np.flatnonzero(half == 0), 1: np.flatnonzero(half == 1)}
INDEX_OF = {n: np.flatnonzero(bench == n) for n in BENCH}

KIT_CHANCE, CHANCE = {}, {}
for n in BENCH:
    rows = INDEX_OF[n]
    if n == 59:                                              # F1 on the positive class
        prevalence = float(np.mean(gidx[rows] == 1))
        KIT_CHANCE[n] = prevalence / (prevalence + 0.5)      # chance_on_rows of the kit's summary
        CHANCE[n] = 2 * prevalence / (1 + prevalence)        # the board's baseline: F1 of always answering "hallucinated"
    else:
        KIT_CHANCE[n] = CHANCE[n] = float(np.mean(1.0 / k_options[rows]))

# --------------------------------------------------------------------------- models
P, SAID = {}, {}
for name in list(MODELS) + list(REFERENCE):
    data = json.load(open(os.path.join(ext_dir, name + ".json"), encoding="utf-8"))
    probs = np.zeros((N, KMAX))
    said = np.zeros(N, dtype=np.int64)
    for i, g in enumerate(gold):
        entry = data["rows"][g["run_id"]]
        assert entry["s"] == "ok" and entry["h"], (name, g["run_id"])
        values = entry["p"][KEY[i]]
        probs[i, :len(values)] = values
        stated = entry["c"][KEY[i]]
        question = g["questions"][KEY[i]]
        said[i] = (1 if stated >= 0.5 else 0) if question["type"] == "noul" else question["options"].index(stated)
    P[name], SAID[name] = probs, said

VALID = np.arange(KMAX)[None, :] < k_options[:, None]
LOGP = {name: np.where(VALID, np.log(np.clip(P[name], 1e-12, None)), -1e30) for name in P}


# --------------------------------------------------------------------------- scoring
def bench_scores(pred, rows=None):
    """Kit metric per benchmark (accuracy; F1 on the hallucinated class for RAGTruth) of a prediction vector."""
    b, p, g = (bench, pred, gidx) if rows is None else (bench[rows], pred[rows], gidx[rows])
    out = {}
    for n in BENCH:
        mask = b == n
        pp, gg = p[mask], g[mask]
        if n == 59:
            tp = int(np.sum((pp == 1) & (gg == 1)))
            fp = int(np.sum((pp == 1) & (gg == 0)))
            fn = int(np.sum((pp == 0) & (gg == 1)))
            out[n] = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0
        else:
            out[n] = float(np.mean(pp == gg)) if len(pp) else 0.0
    return out


def skills(scores):
    return {n: min(1.0, max(0.0, (scores[n] - CHANCE[n]) / (1.0 - CHANCE[n]))) for n in BENCH}


def mini_index(scores, drop=None):
    """100 x the area-weighted mean skill; `drop` leaves one benchmark out (the weights of the remaining areas are renormalized)."""
    sk = skills(scores)
    total, weight = 0.0, 0.0
    for area, w in AREA_W.items():
        members = [sk[n] for n in BENCH if AREA_OF[n] == area and n != drop]
        if members:
            total += w * sum(members) / len(members)
            weight += w
    return 100.0 * total / weight


def mini(pred, rows=None, drop=None):
    return mini_index(bench_scores(pred, rows), drop)


def ensemble(names, weights=None, kind="mean"):
    """Prediction of a combination: arithmetic mean of the probabilities (REAX's rule) or a log-linear pool."""
    w = np.full(len(names), 1.0 / len(names)) if weights is None else np.asarray(weights, float) / float(np.sum(weights))
    table = P if kind == "mean" else LOGP
    total = np.zeros((N, KMAX))
    for weight, name in zip(w, names):
        if weight:
            total += weight * table[name]
    if kind != "mean":
        total = np.where(VALID, total, -np.inf)
    return np.argmax(total, axis=1), total


def proper_scores(names):
    _, total = ensemble(names)
    probs = total / total.sum(axis=1, keepdims=True)
    p_gold = probs[np.arange(N), gidx]
    onehot = np.zeros_like(probs)
    onehot[np.arange(N), gidx] = 1.0
    return {"nll": float(np.mean(-np.log(np.clip(p_gold, 1e-12, None)))), "brier": float(np.mean(np.sum((probs - onehot) ** 2, axis=1)))}


report = {"requests": N, "per_benchmark_requests": {SHORT[n]: int(len(INDEX_OF[n])) for n in BENCH}, "chance": {SHORT[n]: CHANCE[n] for n in BENCH},
          "seed": SEED, "resamples": RESAMPLES}

# --------------------------------------------------------------------------- 1. validation
board = json.load(open(board_path, encoding="utf-8"))
by_name = {m["name"]: m for m in board["models"]}
validation, singles = {}, {}
worst_summary = worst_raw = worst_skill = 0.0
for name in list(MODELS) + list(REFERENCE):
    family, board_name, licence = (MODELS.get(name) or REFERENCE[name])
    own = bench_scores(SAID[name])                           # the model's own stated answers, as the kit scores them
    own_skill = skills(own)
    summary = {b["catalog_id"]: b for b in json.load(open(os.path.join(results_dir, name, "benchmark-summary.json"), encoding="utf-8"))["benchmarks"]}
    entry = by_name[board_name]
    rows = {}
    for n in BENCH:
        published = summary[n]["score"]
        on_board = entry["benchmarks"][str(n)]
        rows[SHORT[n]] = {"ours": own[n], "published": published, "board_raw": on_board["raw"], "our_skill": own_skill[n], "board_skill": on_board["skill"]}
        worst_summary = max(worst_summary, abs(own[n] - published))
        worst_raw = max(worst_raw, abs(round(own[n], 4) - on_board["raw"]))
        worst_skill = max(worst_skill, abs(round(own_skill[n], 4) - on_board["skill"]))
        assert abs(KIT_CHANCE[n] - summary[n]["detail"]["chance_on_rows"]) < 1e-12, (name, n)
    validation[name] = rows
    latency = entry.get("latency") or {}
    singles[name] = {"family": family, "board_name": board_name, "licence": licence, "full": entry["v03"]["v"], "same_skills": entry["v03"]["same_skills"],
                     "new_domains": entry["v03"]["new_domains"], "public_index": entry["v03"]["pub"], "mini_index": mini_index(own), "mini_index_without_phish": mini_index(own, drop=56),
                     "said_vs_argmax_differ": int(np.sum(SAID[name] != np.argmax(P[name], axis=1))), "latency_median_ms": latency.get("median"), "latency_mean_ms": latency.get("mean"),
                     "skill": {SHORT[n]: own_skill[n] for n in BENCH}, "score": {SHORT[n]: own[n] for n in BENCH}, **proper_scores([name])}
report["validation"] = {"max_abs_difference_vs_published_summary": worst_summary, "max_abs_difference_vs_board_raw_4dp": worst_raw,
                        "max_abs_difference_vs_board_skill_4dp": worst_skill, "rows": validation}
report["singles"] = singles

PRED = {name: np.argmax(P[name], axis=1) for name in P}
MINI = {name: mini(PRED[name]) for name in P}
MINI_DROP = {name: {n: mini(PRED[name], drop=n) for n in BENCH} for name in P}
MINI_HALF = {name: (mini(PRED[name], HALF[0]), mini(PRED[name], HALF[1])) for name in P}


def kind_of(a, b):
    return "cross" if MODELS[a][0] != MODELS[b][0] else "within-" + MODELS[a][0]


def by_kind(values):
    """values: list of (kind, number) -> summary per kind."""
    out = {}
    for kind in ("cross", "within-Q", "within-G"):
        chosen = np.array([v for k, v in values if k == kind])
        out[kind] = {"pairs": int(len(chosen)), "mean": float(chosen.mean()), "median": float(np.median(chosen)), "min": float(chosen.min()), "max": float(chosen.max()),
                     "share_positive": float(np.mean(chosen > 0))}
    return out


# --------------------------------------------------------------------------- 2, 3 and 4. pairs: 50/50, diversity, one fitted weight
pairs = []
for a, b in itertools.combinations(SINGLES, 2):
    pred, _ = ensemble([a, b])
    pred_log, _ = ensemble([a, b], kind="logpool")
    best = a if MINI[a] >= MINI[b] else b
    scores = bench_scores(pred)
    value = mini_index(scores)
    drop_gain = {SHORT[n]: mini_index(scores, drop=n) - max(MINI_DROP[a][n], MINI_DROP[b][n]) for n in BENCH}
    # one weight for the pair, fitted on one half and measured on the other; w = 0 or 1 is a single member, so the fit can also say "do not combine"
    curve = {}
    for w in GRID:
        p = pred if w == 0.5 else (PRED[a] if w == 1 else PRED[b] if w == 0 else ensemble([a, b], weights=[w, 1 - w])[0])
        curve[w] = (mini(p, HALF[0]), mini(p, HALF[1]))
    w0, w1 = max(GRID, key=lambda w: curve[w][0]), max(GRID, key=lambda w: curve[w][1])
    single0 = 1.0 if MINI_HALF[a][0] >= MINI_HALF[b][0] else 0.0        # the better single member according to half 0
    single1 = 1.0 if MINI_HALF[a][1] >= MINI_HALF[b][1] else 0.0
    fitted_gain = ((curve[w0][1] - curve[single0][1]) + (curve[w1][0] - curve[single1][0])) / 2
    equal_gain = ((curve[0.5][1] - curve[single0][1]) + (curve[0.5][0] - curve[single1][0])) / 2
    pairs.append({"a": a, "b": b, "kind": kind_of(a, b), "mini_index": value, "mini_index_logpool": mini(pred_log), "best_member": best, "best_member_mini": MINI[best],
                  "gain": value - MINI[best], "gain_leaving_out": drop_gain, "member_gap": abs(MINI[a] - MINI[b]),
                  "disagree": float(np.mean(PRED[a] != PRED[b])), "either_right": float(np.mean((PRED[a] == gidx) | (PRED[b] == gidx))),
                  "fitted_weight_of_a": [w0, w1], "held_out_fitted": (curve[w0][1] + curve[w1][0]) / 2, "held_out_gain_fitted": fitted_gain, "held_out_gain_equal": equal_gain,
                  "skill": {SHORT[n]: skills(scores)[n] for n in BENCH}})
report["pairs"] = sorted(pairs, key=lambda x: -x["mini_index"])
report["pairs_by_kind"] = {
    "gain_equal_weights": by_kind([(p["kind"], p["gain"]) for p in pairs]),
    "gain_equal_weights_members_within_2_points": by_kind([(p["kind"], p["gain"]) for p in pairs if p["member_gap"] <= 2.0]),
    "held_out_gain_fitted_weight": by_kind([(p["kind"], p["held_out_gain_fitted"]) for p in pairs]),
    "held_out_gain_equal_weights": by_kind([(p["kind"], p["held_out_gain_equal"]) for p in pairs]),
    "disagree": by_kind([(p["kind"], p["disagree"]) for p in pairs]),
    "either_right": by_kind([(p["kind"], p["either_right"]) for p in pairs]),
    "gain_leaving_out": {SHORT[n]: by_kind([(p["kind"], p["gain_leaving_out"][SHORT[n]]) for p in pairs]) for n in BENCH},
}


# --------------------------------------------------------------------------- 5. subsets chosen on one half, measured on the other
def split_half(pool, max_size=4):
    table = {}
    for size in range(1, max_size + 1):
        for subset in itertools.combinations(pool, size):
            pred = PRED[subset[0]] if size == 1 else ensemble(list(subset))[0]
            table[subset] = (mini(pred, HALF[0]), mini(pred, HALF[1]), mini(pred))
    out = {}
    for size in range(1, max_size + 1):
        candidates = [s for s in table if len(s) == size]
        chosen_on_0 = max(candidates, key=lambda s: table[s][0])
        chosen_on_1 = max(candidates, key=lambda s: table[s][1])
        best_all = max(candidates, key=lambda s: table[s][2])
        out[size] = {"chosen_on_half0": list(chosen_on_0), "measured_on_half1": table[chosen_on_0][1], "chosen_on_half1": list(chosen_on_1), "measured_on_half0": table[chosen_on_1][0],
                     "held_out_mean": (table[chosen_on_0][1] + table[chosen_on_1][0]) / 2, "best_on_all": list(best_all), "best_on_all_mini": table[best_all][2]}
    for size in range(2, max_size + 1):                       # gain over the single model chosen the same way, on the same half
        out[size]["held_out_gain_vs_single"] = ((out[size]["measured_on_half1"] - out[1]["measured_on_half1"]) + (out[size]["measured_on_half0"] - out[1]["measured_on_half0"])) / 2
    return out


report["split_half"] = {"all_12": split_half(SINGLES), "apache_only": split_half(APACHE)}

# --------------------------------------------------------------------------- 6. paired bootstrap
rng = np.random.default_rng(SEED)
SAMPLES = [np.concatenate([rng.choice(INDEX_OF[n], size=len(INDEX_OF[n]), replace=True) for n in BENCH]) for _ in range(RESAMPLES)]


def prediction_of(names, weights=None):
    return PRED[names[0]] if len(names) == 1 else ensemble(names, weights=weights)[0]


def compare(label, new_names, base_names, weights=None, drop=None):
    new, base = prediction_of(new_names, weights), prediction_of(base_names)
    deltas = np.array([mini(new, rows, drop) - mini(base, rows, drop) for rows in SAMPLES])
    new_scores, base_scores = skills(bench_scores(new)), skills(bench_scores(base))
    return {"label": label, "new": new_names, "weights": weights, "base": base_names, "left_out": SHORT.get(drop), "new_mini": mini(new, drop=drop), "base_mini": mini(base, drop=drop),
            "delta": mini(new, drop=drop) - mini(base, drop=drop), "ci95": [float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5))],
            "skill_delta_points": {SHORT[n]: 100 * (new_scores[n] - base_scores[n]) for n in BENCH}}


def better(a, b, drop=None):
    """The better of two single models on the aggregate (with `drop`: on the aggregate that leaves that benchmark out)."""
    if drop is None:
        return a if MINI[a] >= MINI[b] else b
    return a if MINI_DROP[a][drop] >= MINI_DROP[b][drop] else b


comparisons = [
    compare("GLiDE + KnowLine (top single of each family), 50/50, vs its best member", ["glide", "knowline"], [better("glide", "knowline")]),
    compare("GLiDE + KnowLine, 50/50, vs REAX 1.3", ["glide", "knowline"], ["reax"]),
    compare("GLiDE + KnowLine, 50/50, vs its best member, PhishNChips left out", ["glide", "knowline"], [better("glide", "knowline", 56)], drop=56),
    compare("KnowLine + d3-ultra (top Apache of each family by Full), 50/50, vs its best member", ["knowline", "d3-ultra"], [better("knowline", "d3-ultra")]),
    compare("KnowLine + d3-ultra, 50/50, vs its best member, PhishNChips left out", ["knowline", "d3-ultra"], [better("knowline", "d3-ultra", 56)], drop=56),
    compare("KnowLine + d3-ultra, weights 0.7 / 0.3 (the weight both halves fitted), vs KnowLine", ["knowline", "d3-ultra"], ["knowline"], weights=[0.7, 0.3]),
    compare("KnowLine + d3-turbo (Apache), 50/50, vs its best member", ["knowline", "d3-turbo"], [better("knowline", "d3-turbo")]),
    compare("control inside Gemma: KnowLine + Sieve vs its best member", ["knowline", "sieve"], [better("knowline", "sieve")]),
    compare("control inside Gemma: KnowLine + Jeffan vs its best member", ["knowline", "jeffan"], [better("knowline", "jeffan")]),
    compare("control inside Qwen: GLiDE + d3-ultra vs its best member", ["glide", "d3-ultra"], [better("glide", "d3-ultra")]),
    compare("control inside Qwen: GLiDE + d3-turbo vs its best member", ["glide", "d3-turbo"], [better("glide", "d3-turbo")]),
    compare("REAX 1.3 vs GLiDE", ["reax"], ["glide"]),
    compare("REAX 1.3 vs GLiDE, PhishNChips left out", ["reax"], ["glide"], drop=56),
    compare("REAX 1.3 vs KnowLine, PhishNChips left out", ["reax"], ["knowline"], drop=56),
]
apache = report["split_half"]["apache_only"]
for size in (2, 3):
    comparisons.append(compare(f"best Apache subset of {size} (the one both halves chose) vs best Apache single", apache[size]["best_on_all"], apache[1]["best_on_all"]))
report["comparisons"] = comparisons
report["reference"] = {"reax_mini_index": MINI["reax"], "reax_mini_index_without_phish": MINI_DROP["reax"][56]}
report["proper_scores"] = {"glide+knowline": proper_scores(["glide", "knowline"]), "knowline+d3-ultra": proper_scores(["knowline", "d3-ultra"]), "knowline+d3-turbo": proper_scores(["knowline", "d3-turbo"]),
                           "knowline": proper_scores(["knowline"]), "d3-ultra": proper_scores(["d3-ultra"]), "glide": proper_scores(["glide"]), "reax": proper_scores(["reax"])}

# --------------------------------------------------------------------------- 7. the board's private parts: REAX 1.3 against the public models it descends from
ancestors = {"KnowLine 31B Gen1.5": "Part G blend", "Jeffan-31b-1710": "Part G lineage", "vLLM-SR d3-ultra (Decision 3.0, 27B)": "Part Q blend", "baselabs mammoth-1 (27B)": "Part Q lineage"}
reax = by_name["REAX 1.3"]["v03"]
private = {"reax": {k: reax[k] for k in ("v", "same_skills", "new_domains", "pub")}, "ancestors": {}}
for board_name, role in ancestors.items():
    v = by_name[board_name]["v03"]
    private["ancestors"][board_name] = {"role": role, **{k: v[k] for k in ("v", "same_skills", "new_domains", "pub")}}
for part in ("v", "same_skills", "new_domains", "pub"):
    private["reax_minus_best_ancestor_" + part] = reax[part] - max(a[part] for a in private["ancestors"].values())
report["board_private_evidence"] = private

with open(out_path, "w", encoding="utf-8") as f:
    json.dump(report, f, indent=1)

# --------------------------------------------------------------------------- printed summary
print(f"requests {N}: " + ", ".join(f"{SHORT[n]} {len(INDEX_OF[n])}" for n in BENCH))
print(f"validation: max |ours - published summary| = {worst_summary:.2e};  max |ours - board raw| (4 dp) = {worst_raw:.4f};  max |our skill - board skill| (4 dp) = {worst_skill:.4f}")
print("\nSINGLES (mini-index; board Full and public index for reference)")
print(f"{'model':10s} fam {'mini':>6s} {'-phish':>6s} {'Full':>6s} {'pub':>6s}  " + " ".join(f"{SHORT[n][:9]:>9s}" for n in BENCH) + "  lat med/mean  licence")
for name in sorted(P, key=lambda x: -singles[x]["mini_index"]):
    info = singles[name]
    print(f"{name:10s}  {info['family']}  {info['mini_index']:6.2f} {info['mini_index_without_phish']:6.2f} {info['full']:6.2f} {info['public_index']:6.2f}  "
          + " ".join(f"{info['skill'][SHORT[n]]:9.3f}" for n in BENCH) + f"  {info['latency_median_ms']:.0f}/{info['latency_mean_ms']:.0f}  {info['licence']}")
print("\nPAIRS by kind (mini-index points; gain = pair minus its best member)")
for label, table in (("50/50, all requests", "gain_equal_weights"), ("50/50, members within 2 points", "gain_equal_weights_members_within_2_points"),
                     ("50/50, held-out halves", "held_out_gain_equal_weights"), ("one fitted weight, held-out", "held_out_gain_fitted_weight")):
    for kind, s in report["pairs_by_kind"][table].items():
        print(f"  {label:32s} {kind:9s} n={s['pairs']:2d}  mean {s['mean']:+.2f}  median {s['median']:+.2f}  range [{s['min']:+.2f}, {s['max']:+.2f}]  positive {100 * s['share_positive']:.0f}%")
for kind, s in report["pairs_by_kind"]["disagree"].items():
    e = report["pairs_by_kind"]["either_right"][kind]
    print(f"  diversity {kind:9s} answers differ on {100 * s['mean']:.1f}% of requests; at least one member right on {100 * e['mean']:.1f}%")
print("  50/50 gain with one benchmark left out (mean by kind):")
for n in BENCH:
    t = report["pairs_by_kind"]["gain_leaving_out"][SHORT[n]]
    print(f"    without {SHORT[n]:11s} cross {t['cross']['mean']:+.2f}   within-Q {t['within-Q']['mean']:+.2f}   within-G {t['within-G']['mean']:+.2f}")
print("\nTOP 12 PAIRS (50/50)")
for p in report["pairs"][:12]:
    print(f"  {p['a']:9s}+ {p['b']:9s} {p['kind']:9s} mini {p['mini_index']:6.2f}  best member {p['best_member']:9s} {p['best_member_mini']:6.2f}  gain {p['gain']:+.2f}  "
          f"fitted w {p['fitted_weight_of_a'][0]}/{p['fitted_weight_of_a'][1]} held-out gain {p['held_out_gain_fitted']:+.2f}  differ {100 * p['disagree']:.1f}%")
for pool_name, table in report["split_half"].items():
    print(f"\nSPLIT-HALF, pool {pool_name} (chosen on one half, measured on the other; mean of the two directions)")
    for size, row in table.items():
        gain = row.get("held_out_gain_vs_single")
        print(f"  size {size}: held-out {row['held_out_mean']:6.2f}" + (f"  gain vs single {gain:+.2f}" if gain is not None else "") + f"  | chosen: {'+'.join(row['chosen_on_half0'])} / {'+'.join(row['chosen_on_half1'])}")
print("\nCOMPARISONS (paired bootstrap, 95%)")
for c in comparisons:
    print(f"  {c['label']}: {c['new_mini']:.2f} vs {c['base_mini']:.2f}  delta {c['delta']:+.2f} [{c['ci95'][0]:+.2f}, {c['ci95'][1]:+.2f}]")
    print("      by benchmark (skill points): " + ", ".join(f"{k} {v:+.1f}" for k, v in c["skill_delta_points"].items()))
print("\nPROPER SCORES (mean over requests; lower is better)")
for label, row in report["proper_scores"].items():
    print(f"  {label:20s} NLL {row['nll']:.4f}  Brier {row['brier']:.4f}")
print("\nBOARD, private parts: REAX 1.3 minus the best public model it descends from")
print("  " + ", ".join(f"{part} {private['reax_minus_best_ancestor_' + part]:+.2f}" for part in ("v", "same_skills", "new_domains", "pub")))
