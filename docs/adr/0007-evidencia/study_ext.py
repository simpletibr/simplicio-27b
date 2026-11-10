"""Cross-family ensemble study on every public benchmark that could be rebuilt, scored by the kit's own functions.

    python -I study_ext.py KIT_ROOT SLIM_ROWS.jsonl EXT_DIR BOARD_INDEX.json OUT.json

Every score here is produced by `decision_index` itself (benchmark_summary, added.report, score_panel, index02): this script
only builds the result rows of a combination (probabilities averaged question by question) and hands them to the kit.
A benchmark enters the aggregate only if, for every one of the 13 published runs, the kit's raw score AND skill on the rebuilt
rows equal the board's to 4 decimals.  The aggregate is the 0.3 public index formula restricted to those benchmarks
("partial index"): the same area weights and the same 1.2 weight for the starred benchmarks.
"""
import collections
import copy
import hashlib
import itertools
import json
import os
import sys
import time

kit, rows_path, ext_dir, board_path, out_path = sys.argv[1:6]
sys.path.insert(0, kit)

from decision_index import editions  # noqa: E402
from decision_index.scoring import added, index02  # noqa: E402
from decision_index.scoring import report as kit_report  # noqa: E402

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
ALL = {**MODELS, **REFERENCE}
APACHE = [name for name, (_, _, lic) in MODELS.items() if lic == "apache-2.0"]
TOLERANCE = 0.00011

SPEC = index02.spec("0.3")
ADDED = {int(n) for n in SPEC["added"]}
INDEX_IDS = [n for a in SPEC["areas"] for n in a["benchmarks"]]
AREA_OF = {n: a["id"] for a in SPEC["areas"] for n in a["benchmarks"]}
METRICS = {int(n): (m["name"], m["key"]) for n, m in SPEC.get("metrics", {}).items()}

ROWS = []
with open(rows_path, encoding="utf-8") as f:
    for line in f:
        row = json.loads(line)
        if row["_evaluation"]["catalog_id"] in INDEX_IDS:
            ROWS.append(row)
NAMES = {}
for row in ROWS:
    NAMES.setdefault(row["_evaluation"]["catalog_id"], row["_evaluation"]["dataset"])
REBUILT = sorted(NAMES)
HALF_OF = {}
for row in ROWS:
    group = f'{row["_evaluation"]["catalog_id"]}:{row["_evaluation"]["group_id"]}'
    HALF_OF[row["_evaluation"]["run_id"]] = int(hashlib.sha256(group.encode()).hexdigest(), 16) % 2
ROWS_HALF = {h: [r for r in ROWS if HALF_OF[r["_evaluation"]["run_id"]] == h] for h in (0, 1)}

M = {name: json.load(open(os.path.join(ext_dir, name + ".json"), encoding="utf-8"))["rows"] for name in ALL}


class FakeSuite:
    """What the kit's scorers ask of a suite: the rows (already filtered by edition and exclusions) and the edition."""

    def __init__(self, rows):
        self._rows = rows
        self.edition = editions.get("0.3")

    def rows(self, apply_exclusions=False):
        return iter(self._rows)


def usable(name, run_id):
    entry = M[name].get(run_id)
    return entry if entry is not None and entry["s"] == "ok" and entry["h"] else None


def results_of(names, weights=None, own_choice=False):
    """Result rows in the kit's format.  One model with own_choice: its stated answers.  Otherwise the weighted mean of the probabilities."""
    weights = [1.0 / len(names)] * len(names) if weights is None else [w / sum(weights) for w in weights]
    out = {}
    for row in ROWS:
        e = row["_evaluation"]
        entries = [usable(name, e["run_id"]) for name in names]
        if any(entry is None for entry in entries):
            continue                                             # not answered by every member: the kit counts it as pending (wrong)
        answers = {}
        for key, q in row["questions"].items():
            if q["type"] == "noul":
                answers[key] = {"type": "noul", "noul": sum(w * entry["p"][key] for w, entry in zip(weights, entries))}
                continue
            options = list(q["criteria"])
            if len(entries) == 1:
                probs = entries[0]["p"][key]
                choice = entries[0]["c"][key] if own_choice else options[max(range(len(options)), key=probs.__getitem__)]
            else:
                probs = [sum(w * entry["p"][key][i] for w, entry in zip(weights, entries)) for i in range(len(options))]
                choice = options[max(range(len(options)), key=probs.__getitem__)]
            answers[key] = {"type": "choice", "choice": choice, "probabilities": dict(zip(options, probs))}
        out[e["run_id"]] = {"run_id": e["run_id"], "catalog_id": e["catalog_id"], "status": "ok", "response": {"answers": answers}, "total_wall_ms": 0.0}
    return out


def spec_for(ids):
    s = copy.deepcopy(SPEC)
    s["areas"] = [dict(a, benchmarks=[n for n in a["benchmarks"] if n in ids]) for a in s["areas"]]
    s["areas"] = [a for a in s["areas"] if a["benchmarks"]]
    s["not_in_index"] = []
    return s


def kit_benchmarks(results, rows=None):
    """Raw score, skill and coverage of every rebuilt benchmark, exactly as the kit's 0.3 index computes them."""
    rows = ROWS if rows is None else rows
    base = [r for r in rows if r["_evaluation"]["catalog_id"] not in ADDED]
    extra = collections.defaultdict(list)
    for r in rows:
        if r["_evaluation"]["catalog_id"] in ADDED:
            extra[r["_evaluation"]["catalog_id"]].append(r)
    summary = kit_report.benchmark_summary(None, results, "study", None, rows=base, metrics=METRICS)
    added_reports = {n: added.report(n, rs, results) for n, rs in sorted(extra.items())}
    entry = index02.index_entry(FakeSuite(rows), results, summary, added_reports, spec_for(set(REBUILT)))
    return {int(n): {"raw": v["raw"], "skill": v["skill"], "coverage": v["coverage"]} for n, v in entry["benchmarks"].items()}


def partial_index(values, ids):
    scores, areas = index02.aggregate({n: values[n] for n in ids}, spec_for(set(ids)))
    return scores["balanced_skill"], {a["id"]: a["skill"] for a in areas}


started = time.time()
report = {"rebuilt_benchmarks": {str(n): NAMES[n] for n in REBUILT}, "requests": len(ROWS)}

# --------------------------------------------------------------------------- 1. singles and validation against the board
board = json.load(open(board_path, encoding="utf-8"))
by_name = {m["name"]: m for m in board["models"]}
SINGLE_VALUES, coverage_gaps = {}, {}
worst = {n: {"raw": 0.0, "skill": 0.0, "model": None} for n in REBUILT}
for name in ALL:
    values = kit_benchmarks(results_of([name], own_choice=True))
    SINGLE_VALUES[name] = values
    entry = by_name[ALL[name][1]]
    for n in REBUILT:
        on_board = entry["benchmarks"].get(str(n))
        if on_board is None:
            worst[n] = {"raw": 9.0, "skill": 9.0, "model": name}
            continue
        d_raw, d_skill = abs(values[n]["raw"] - on_board["raw"]), abs(values[n]["skill"] - on_board["skill"])
        if max(d_raw, d_skill) > max(worst[n]["raw"], worst[n]["skill"]):
            worst[n] = {"raw": d_raw, "skill": d_skill, "model": name}
        if values[n]["coverage"] < 1:
            coverage_gaps.setdefault(name, {})[NAMES[n]] = values[n]["coverage"]
    print(f"scored {name} ({time.time() - started:.0f}s)", flush=True)
VALID = [n for n in REBUILT if max(worst[n]["raw"], worst[n]["skill"]) <= TOLERANCE]
report["validation"] = {NAMES[n]: {"catalog_id": n, "area": AREA_OF[n], "max_abs_raw_difference": worst[n]["raw"], "max_abs_skill_difference": worst[n]["skill"],
                                   "worst_model": worst[n]["model"], "used": n in VALID} for n in REBUILT}
report["used_benchmarks"] = {a["id"]: [NAMES[n] for n in a["benchmarks"] if n in VALID] for a in SPEC["areas"]}
report["coverage_gaps"] = coverage_gaps
report["requests_used"] = sum(1 for r in ROWS if r["_evaluation"]["catalog_id"] in VALID)

INDEX = {name: partial_index(SINGLE_VALUES[name], VALID)[0] for name in ALL}
report["singles"] = {}
for name in ALL:
    family, board_name, licence = ALL[name]
    v03 = by_name[board_name]["v03"]
    latency = by_name[board_name].get("latency") or {}
    report["singles"][name] = {"family": family, "board_name": board_name, "licence": licence, "partial_index": INDEX[name], "areas": partial_index(SINGLE_VALUES[name], VALID)[1],
                               "board_public_index": v03["pub"], "board_full": v03["v"], "same_skills": v03["same_skills"], "new_domains": v03["new_domains"],
                               "latency_median_ms": latency.get("median"), "latency_mean_ms": latency.get("mean"),
                               "skill": {NAMES[n]: SINGLE_VALUES[name][n]["skill"] for n in VALID}}


def kind_of(a, b):
    return "cross" if MODELS[a][0] != MODELS[b][0] else "within-" + MODELS[a][0]


def describe(values):
    values = sorted(values)
    count = len(values)
    return {"n": count, "mean": sum(values) / count, "median": (values[count // 2] if count % 2 else (values[count // 2 - 1] + values[count // 2]) / 2),
            "min": values[0], "max": values[-1], "share_positive": sum(v > 0 for v in values) / count} if count else {"n": 0}


# --------------------------------------------------------------------------- 2. every pair, probabilities averaged 50/50
PAIR_VALUES = {}
pairs = []
for a, b in itertools.combinations(SINGLES, 2):
    values = kit_benchmarks(results_of([a, b]))
    PAIR_VALUES[(a, b)] = values
    value = partial_index(values, VALID)[0]
    best = a if INDEX[a] >= INDEX[b] else b
    deltas = {n: values[n]["skill"] - SINGLE_VALUES[best][n]["skill"] for n in VALID}
    over_both = {n: values[n]["skill"] - max(SINGLE_VALUES[a][n]["skill"], SINGLE_VALUES[b][n]["skill"]) for n in VALID}
    pairs.append({"a": a, "b": b, "kind": kind_of(a, b), "partial_index": value, "best_member": best, "best_member_index": INDEX[best], "gain": value - INDEX[best],
                  "member_gap": abs(INDEX[a] - INDEX[b]), "benchmarks_better_than_best_member": sum(d > 0.0005 for d in deltas.values()),
                  "benchmarks_worse_than_best_member": sum(d < -0.0005 for d in deltas.values()),
                  "benchmarks_better_than_both_members": sum(d > 0.0005 for d in over_both.values()),
                  "latency_mean_sum_ms": (report["singles"][a]["latency_mean_ms"] or 0) + (report["singles"][b]["latency_mean_ms"] or 0),
                  "skill_delta_vs_best_member": {NAMES[n]: deltas[n] for n in VALID}})
    print(f"pair {a}+{b} ({time.time() - started:.0f}s)", flush=True)
report["pairs"] = sorted(pairs, key=lambda p: -p["partial_index"])
report["pairs_by_kind"] = {
    "gain": {kind: describe([p["gain"] for p in pairs if p["kind"] == kind]) for kind in ("cross", "within-Q", "within-G")},
    "gain_members_within_2_points": {kind: describe([p["gain"] for p in pairs if p["kind"] == kind and p["member_gap"] <= 2.0]) for kind in ("cross", "within-Q", "within-G")},
    "gain_apache_only": {kind: describe([p["gain"] for p in pairs if p["kind"] == kind and p["a"] in APACHE and p["b"] in APACHE]) for kind in ("cross", "within-Q", "within-G")},
}

# --------------------------------------------------------------------------- 3. headline combinations on each half, and one fitted weight
HEADLINE = [("glide", "knowline"), ("knowline", "d3-ultra"), ("knowline", "d3-turbo"), ("glide", "flexdm"), ("glide", "sieve"), ("sieve", "d3-ultra"), ("knowline", "jeffan"), ("glide", "d3-ultra")]
half_single = {name: [partial_index(kit_benchmarks(results_of([name]), ROWS_HALF[h]), VALID)[0] for h in (0, 1)] for name in sorted({m for pair in HEADLINE for m in pair} | {"reax"})}
report["halves"] = {"singles": half_single, "pairs": []}
GRID = (0.3, 0.4, 0.5, 0.6, 0.7)
for a, b in HEADLINE:
    curve = {}
    for w in GRID:
        results = results_of([a, b], weights=[w, 1 - w])
        curve[w] = [partial_index(kit_benchmarks(results, ROWS_HALF[h]), VALID)[0] for h in (0, 1)]
    curve[1.0], curve[0.0] = half_single[a], half_single[b]
    options = (0.0,) + GRID + (1.0,)
    w0, w1 = max(options, key=lambda w: curve[w][0]), max(options, key=lambda w: curve[w][1])
    s0, s1 = (1.0 if half_single[a][0] >= half_single[b][0] else 0.0), (1.0 if half_single[a][1] >= half_single[b][1] else 0.0)
    report["halves"]["pairs"].append({"a": a, "b": b, "equal_weights": curve[0.5], "best_member_chosen_on_other_half": [curve[s1][0], curve[s0][1]],
                                      "gain_equal_weights": [curve[0.5][0] - curve[s1][0], curve[0.5][1] - curve[s0][1]],
                                      "fitted_weight_of_a": [w0, w1], "held_out_fitted": [curve[w1][0], curve[w0][1]],
                                      "held_out_gain_fitted": ((curve[w1][0] - curve[s1][0]) + (curve[w0][1] - curve[s0][1])) / 2})
    print(f"halves {a}+{b} ({time.time() - started:.0f}s)", flush=True)

# --------------------------------------------------------------------------- 4. three members (does not fit one 96 GB card in FP8: an upper reference only)
triples = []
for names in (("d3-turbo", "knowline", "jeffan"), ("glide", "knowline", "jeffan"), ("glide", "knowline", "sieve"), ("d3-ultra", "knowline", "sieve")):
    value = partial_index(kit_benchmarks(results_of(list(names))), VALID)[0]
    best = max(names, key=lambda n: INDEX[n])
    triples.append({"members": list(names), "partial_index": value, "best_member": best, "gain": value - INDEX[best]})
report["triples"] = triples

with open(out_path, "w", encoding="utf-8") as f:
    json.dump(report, f, indent=1)

# --------------------------------------------------------------------------- printed summary
print(f"\nrebuilt {len(REBUILT)} index benchmarks, {len(ROWS)} requests; validated against the board and used: {len(VALID)} ({report['requests_used']} requests)")
for n in REBUILT:
    if n not in VALID:
        print(f"  NOT USED {n} {NAMES[n]}: max raw difference {worst[n]['raw']:.4f}, skill {worst[n]['skill']:.4f} (worst: {worst[n]['model']})")
print("  used, by area: " + "; ".join(f"{area}: {', '.join(names)}" for area, names in report["used_benchmarks"].items()))
if coverage_gaps:
    print("  coverage below 1:", json.dumps(coverage_gaps))
print("\nSINGLES (partial index = the 0.3 public index formula over the used benchmarks)")
print(f"{'model':10s} fam {'partial':>7s} {'pub':>6s} {'Full':>6s}  " + " ".join(f"{a['id'][:9]:>9s}" for a in SPEC["areas"]) + "   lat mean  licence")
for name in sorted(ALL, key=lambda x: -INDEX[x]):
    s = report["singles"][name]
    print(f"{name:10s}  {s['family']}  {s['partial_index']:7.2f} {s['board_public_index']:6.2f} {s['board_full']:6.2f}  " + " ".join(f"{100 * s['areas'].get(a['id'], 0):9.1f}" for a in SPEC["areas"])
          + f"   {s['latency_mean_ms']:.0f}  {s['licence']}")
print("\nPAIRS, 50/50 (gain = pair minus its best member, partial-index points)")
for label, key in (("all pairs", "gain"), ("members within 2 points", "gain_members_within_2_points"), ("Apache-2.0 members only", "gain_apache_only")):
    for kind, s in report["pairs_by_kind"][key].items():
        if s["n"]:
            print(f"  {label:26s} {kind:9s} n={s['n']:2d}  mean {s['mean']:+.2f}  median {s['median']:+.2f}  range [{s['min']:+.2f}, {s['max']:+.2f}]  positive {100 * s['share_positive']:.0f}%")
print("\nTOP 15 PAIRS")
for p in report["pairs"][:15]:
    print(f"  {p['a']:9s}+ {p['b']:9s} {p['kind']:9s} {p['partial_index']:6.2f}  best member {p['best_member']:9s} {p['best_member_index']:6.2f}  gain {p['gain']:+.2f}  "
          f"benchmarks better/worse than best member {p['benchmarks_better_than_best_member']}/{p['benchmarks_worse_than_best_member']}, better than both {p['benchmarks_better_than_both_members']}  mean latency sum {p['latency_mean_sum_ms']:.0f} ms")
print(f"\nREAX 1.3 on the same benchmarks: {INDEX['reax']:.2f}")
print("\nHALVES (each pair on the two halves of the requests; best member chosen on the other half)")
for row in report["halves"]["pairs"]:
    print(f"  {row['a']}+{row['b']}: 50/50 gain {row['gain_equal_weights'][0]:+.2f} / {row['gain_equal_weights'][1]:+.2f};  fitted weight of {row['a']} {row['fitted_weight_of_a'][0]}/{row['fitted_weight_of_a'][1]}, held-out gain {row['held_out_gain_fitted']:+.2f}")
print("\nTHREE MEMBERS (reference only)")
for t in triples:
    print(f"  {'+'.join(t['members'])}: {t['partial_index']:.2f}  gain over {t['best_member']} {t['gain']:+.2f}")
print(f"\ntotal {time.time() - started:.0f}s")
