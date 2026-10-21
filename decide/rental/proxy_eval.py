"""Measure a model on the home-made private proxy and compare two runs: the measurement behind G1, G2 and G4, as four commands.

    python -m decide.rental.proxy_eval requests --proxy work/proxy --out work/proxy-requests.jsonl
    python -m decide.rental.proxy_eval run --requests work/proxy-requests.jsonl --model flash=dir:work/flash --out work/g2/flash
    python -m decide.rental.proxy_eval run --requests work/proxy-requests.jsonl --model base=hub:ORG/REPO@<40-hex commit> --out work/g2/base
    python -m decide.rental.proxy_eval score --proxy work/proxy --results flash=work/g2/flash base=work/g2/base --out scores.json
    python -m decide.rental.proxy_eval compare --proxy work/proxy --new work/g2/new --baseline work/g2/flash --margin 1.0

``requests`` writes the proxy as the JSONL the kit reads (the decider alone: no reasoning block).  ``run`` answers it with the kit
(``decide.sd``, Squad A) and resumes where an earlier run stopped.  ``score`` prints the proxy score of each run (100 x the mean
chance-corrected skill per source, the scorer of ``decide.data.proxy_private``) and writes ``{name: skill}`` for
``python -m decide.data.proxy_private validate`` (gate G1, whose names must be the board's engine ids).  ``compare`` is gate G2 (or
any margin): the new run against the baseline on the same questions, with the paired bootstrap interval; it is the same function
the rented day uses for G4 (``decide.rental.gates.g4_verdict``).

A run with a question that has no row (or an invalid one) is never scored as if it were whole: ``score`` refuses to write and
``compare`` says INVALID.  ``--margin`` has no default and there is no flag to skip the comparison.

Exit codes: 0 done (and PASS); 1 an error (a run the kit did not finish, a missing file); 2 usage; 3 the gate did not pass
(FAIL or INVALID, or an incomplete run in ``score``).

UNVERIFIED: the kit has not loaded a model of the real size through this; the tests use a stand-in for it.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from decide.data import proxy_private as PP
from decide.data import schema as S

from . import gates as G
from . import kit as K

EXIT_OK, EXIT_ERROR, EXIT_USAGE, EXIT_STOPPED = 0, 1, 2, 3
RESULTS_TABLE = "results.parquet"
_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
_HUB = re.compile(r"([A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*)@([0-9a-f]{40})")
MODEL_SYNTAX = "NAME=dir:PATH (a checkpoint folder on this machine) or NAME=hub:ORG/REPO@COMMIT (a Hub model at a 40-character commit)"


class ProxyEvalError(RuntimeError):
    """A run cannot be scored or compared, with the reason."""


def parse_model(text: str) -> K.ModelRef:
    """``flash=dir:work/flash`` or ``pplx=hub:perplexity-ai/pplx-decider-v1.2@<commit>`` -> a :class:`decide.rental.kit.ModelRef`."""
    name, separator, spec = text.partition("=")
    if not separator or not _NAME.fullmatch(name):
        raise ValueError(f"--model {text!r}: write {MODEL_SYNTAX}; the name is letters, digits, '.', '_' or '-'")
    kind, _, rest = spec.partition(":")
    if kind == "dir" and rest:
        return K.ModelRef(name=name, path=Path(rest))
    match = _HUB.fullmatch(rest) if kind == "hub" else None
    if match:
        return K.ModelRef(name=name, repo=match.group(1), revision=match.group(2))
    raise ValueError(f"--model {text!r}: write {MODEL_SYNTAX} (a branch name such as @main is not a pinned revision)")


def parse_pair(text: str) -> Tuple[str, Path]:
    """``flash=work/g2/flash`` -> ``("flash", Path(...))``."""
    name, separator, folder = text.partition("=")
    if not separator or not folder or not _NAME.fullmatch(name):
        raise ValueError(f"{text!r}: write NAME=FOLDER (the folder a `run` wrote)")
    return name, Path(folder)


def results_table(folder: "str | os.PathLike[str]") -> Path:
    table = Path(folder) / RESULTS_TABLE
    if not table.is_file():
        raise ProxyEvalError(f"{table} does not exist: {folder} is not a folder that `run` wrote (or the run never started)")
    return table


def _score(proxy: "str | os.PathLike[str]", folder: "str | os.PathLike[str]") -> Dict[str, Any]:
    return PP.score_rows(G.proxy_requests(proxy), PP.read_results(results_table(folder)))


def cmd_requests(args: argparse.Namespace) -> int:
    rows = G.proxy_to_requests(args.proxy, args.out)
    print(f"{rows} proxy rows written to {args.out}")
    return EXIT_OK


def cmd_run(args: argparse.Namespace, kit: Any) -> int:
    model = parse_model(args.model)
    kit = kit if kit is not None else K.SdKit(quant=args.quant, device=args.device, tta_rotations=args.tta_rotations)
    info = dict(kit.run(model, args.requests, args.out))
    status = info.get("status")
    seconds = info.get("seconds") or 0
    print(json.dumps({"model": model.name, "status": status, "rows": info.get("rows"), "tokens": info.get("tokens"), "seconds": seconds,
                      "tokens_per_second": (info.get("tokens") or 0) / seconds if seconds else None}, indent=2))
    if status != "complete":
        print(f"error: the kit ended the {model.name} run with status {status!r}; run the same command again, it resumes", file=sys.stderr)
        return EXIT_ERROR
    return EXIT_OK


def cmd_score(args: argparse.Namespace) -> int:
    scores: Dict[str, float] = {}
    incomplete: List[str] = []
    for name, folder in map(parse_pair, args.results):
        scored = _score(args.proxy, folder)
        points = None if scored["proxy"] is None else 100 * scored["proxy"]
        print(f"{name}: {'no scored question' if points is None else format(points, '.3f') + ' proxy points'}, {scored['n']} questions scored, "
              f"{scored['missing']} without a row, {scored['bad']} invalid, NLL {scored['nll'] if scored['nll'] is None else round(scored['nll'], 4)}")
        if scored["missing"] or scored["bad"] or scored["proxy"] is None:
            incomplete.append(f"{name} ({scored['missing']} without a row, {scored['bad']} invalid)")
        else:
            scores[name] = scored["proxy"]
    if incomplete:
        print("INVALID: " + "; ".join(incomplete) + ". Run `run` again for them (it resumes); nothing was written.", file=sys.stderr)
        return EXIT_STOPPED
    if args.out:
        target = Path(args.out)
        target.parent.mkdir(parents=True, exist_ok=True)
        pending = target.with_name(target.name + ".tmp")
        pending.write_text(json.dumps(scores, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(pending, target)
        print(f"scores written to {target}")
    return EXIT_OK


def cmd_compare(args: argparse.Namespace) -> int:
    new, baseline = results_table(args.new), results_table(args.baseline)
    verdict = G.g4_verdict(G.proxy_requests(args.proxy), list(PP.read_results(new)), list(PP.read_results(baseline)),
                           margin_points=args.margin, resamples=args.resamples, seed=args.seed)
    if args.out:
        target = Path(args.out)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(verdict, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    print(verdict["status"])
    if verdict["delta_points"] is not None:
        low, high = verdict["ci95"]
        print(f"new {verdict['new']['proxy_points']:.3f} - baseline {verdict['baseline']['proxy_points']:.3f} = {verdict['delta_points']:+.3f} proxy points "
              f"(margin {args.margin}); paired bootstrap 95% interval [{low:+.3f}, {high:+.3f}] over {verdict['questions']} questions")
    if verdict["reasons"]:
        print("reasons: " + ", ".join(verdict["reasons"]))
    if verdict["notes"]:
        print("notes: " + ", ".join(verdict["notes"]))
    print(verdict["scope"])
    return EXIT_OK if verdict["status"] == "PASS" else EXIT_STOPPED


def _margin(text: str) -> float:
    try:
        value = float(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{text!r} is not a number") from None
    if not math.isfinite(value) or value < 0:
        raise argparse.ArgumentTypeError(f"{text!r}: the margin is a number of proxy points, 0 or more")
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m decide.rental.proxy_eval", description=__doc__.split("\n\n")[0],
                                     epilog=__doc__.split("\n\n", 1)[1], formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)

    requests = commands.add_parser("requests", help="write the proxy as the JSONL the kit reads")
    requests.add_argument("--proxy", required=True, help="the proxy folder made by decide.data.proxy_private build")
    requests.add_argument("--out", required=True, help="the JSONL file to write")

    run = commands.add_parser("run", help="answer the requests with a model, through the kit (resumes)")
    run.add_argument("--requests", required=True, help="the JSONL that `requests` wrote")
    run.add_argument("--model", required=True, help=MODEL_SYNTAX)
    run.add_argument("--out", required=True, help="the folder of this run (the kit writes results.parquet there)")
    run.add_argument("--quant", choices=("nf4", "fp16", "bf16", "none"), default="bf16", help="how the kit loads the weights (default: bf16; a 27B on 2 T4 needs nf4)")
    run.add_argument("--device", default="auto", help="the kit's device (default: auto)")
    run.add_argument("--tta-rotations", type=int, default=1, help="option orders averaged per question (default: 1, the published single order)")

    score = commands.add_parser("score", help="the proxy score of one or more runs (gate G1 input)")
    score.add_argument("--proxy", required=True)
    score.add_argument("--results", nargs="+", required=True, metavar="NAME=FOLDER", help="the runs to score")
    score.add_argument("--out", help="write {name: skill} here, for `proxy_private validate --scores`")

    compare = commands.add_parser("compare", help="the new run against the baseline, same questions (gate G2, or any margin)")
    compare.add_argument("--proxy", required=True)
    compare.add_argument("--new", required=True, help="the folder of the run being tested")
    compare.add_argument("--baseline", required=True, help="the folder of the run it must beat")
    compare.add_argument("--margin", required=True, type=_margin, help="proxy points the new run must gain (G2: 1.0; G4: 1.5); no default on purpose")
    compare.add_argument("--resamples", type=int, default=1000, help="bootstrap resamples (default: 1000)")
    compare.add_argument("--seed", type=int, default=G.DEFAULT_SEED)
    compare.add_argument("--out", help="also write the verdict as JSON here")
    return parser


def main(argv: Optional[Sequence[str]] = None, *, kit: Any = None) -> int:
    """Run one command.  ``kit`` replaces :class:`decide.rental.kit.SdKit` (tests pass a stand-in)."""
    args = build_parser().parse_args(argv)
    try:                                                # what is written wrong on the command line is a usage error, before anything runs
        if args.command == "run":
            parse_model(args.model)
        elif args.command == "score":
            for item in args.results:
                parse_pair(item)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_USAGE
    try:
        if args.command == "requests":
            return cmd_requests(args)
        if args.command == "run":
            return cmd_run(args, kit)
        if args.command == "score":
            return cmd_score(args)
        return cmd_compare(args)
    except (ProxyEvalError, K.KitError, S.ManifestError, FileNotFoundError, ImportError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_ERROR


if __name__ == "__main__":
    sys.exit(main())
