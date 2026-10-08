#!/usr/bin/env python3
"""Validador do envelope do simplicio-loop (50 pontos). Só biblioteca padrão, sem rede.

Mede se uma saída de texto segue o formato descrito no README (seção "Output format") e
produzido por generate_dataset.py. Não executa nada do que a saída afirma: "2 passed" ou
"COMMIT_READY" dentro de <validate>/<deliver> são texto, não prova.

Uso:
    python3 scripts/loop_check.py ARQUIVO...            # '-' lê a entrada padrão
    python3 scripts/loop_check.py --jsonl ARQUIVO       # valida o campo de cada linha
    python3 scripts/loop_check.py --complete ARQUIVO    # exige os 50 pontos
Saída: 0 se tudo passou, 1 se alguma saída falhou (ou um arquivo não pôde ser lido),
2 em erro de uso. Na API: check(text, complete=False) -> lista de problemas (vazia = ok).
Cada problema é "regra: detalhe"; a regra (texto antes do primeiro ':') serve para contar razões.

Especificação (fonte entre parênteses)
1. A saída é um único <simplicio_loop> ... </simplicio_loop>; nada além de espaço fora dele (README).
2. Dentro dele, as fases <orient>, <plan>, <patch>, <validate>, <deliver>, cada uma uma vez,
   nessa ordem, abertas e fechadas, sem aninhar. Cada tag fica sozinha na linha (README, dataset).
3. Um ponto é uma linha "[Ponto N: Nome]" (dataset) ou "[Point N: Name]" (README, traduzido).
   Faixas: orient 1-10, plan 11-20, patch 21-30, validate 31-40, deliver 41-50 (generate_dataset.py).
   Ponto numerado fora da faixa da fase onde aparece, ou fora de qualquer fase, é problema.
4. <patch> tem um ou mais blocos "<<<< SEARCH" / "====" / ">>>> REPLACE", marcadores de exatamente
   4 caracteres, sozinhos na linha e sem espaço à esquerda (README; benchmarks/harness/harness.py).
   SEARCH não pode ser vazio (harness: empty_search). REPLACE pode ser vazio (apagar trecho).
   Os 3 a 7+ caracteres do Git/Aider (<<<<<<< SEARCH) são problema.

Escolhas onde a especificação é ambígua (sempre a leitura mais literal)
- Modo padrão (envelope): o README diz "cada fase é uma lista de pontos", mas o próprio exemplo
  abrevia com "…", e mostra <patch> e <deliver> sem ponto numerado. Por isso o padrão NÃO exige
  que os 50 pontos existam, nem que sejam únicos ou crescentes; só confere o que aparece.
- --complete / complete=True: o system prompt do repo diz "os 50 pontos do Simplicio-Loop". Este modo
  acrescenta: todos os 50 pontos, cada um uma vez, na faixa da sua fase, em ordem crescente.
- O nome do ponto não é conferido (o README traduz "Identificacao de Raiz" por "Root" e o gerador
  usa nomes diferentes nos exemplos manuais e nos gerados); só o número conta. "Ponto" e "Point" valem.
- Linhas de texto livre dentro de uma fase são permitidas (o README usa "…"). Texto entre as fases, ou
  fora de <simplicio_loop> (inclusive um bloco <think>), é problema. O README manda rodar sem thinking.
- Marcadores só têm significado em <patch>. Como no harness, dentro de um bloco aberto tudo é conteúdo
  até o ">>>> REPLACE" (um "<<<< SEARCH" ou "</patch>" ali dentro não é interpretado). Exceção: um
  ">>>> REPLACE" antes do "====" encerra o bloco e é reportado, pois falta uma parte obrigatória.
- Não confere se o SEARCH existe no arquivo-alvo (isso exige o código; é o que o harness mede).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter

ROOT_TAG = "simplicio_loop"
PHASES = ("orient", "plan", "patch", "validate", "deliver")
POINT_RANGES = {
    "orient": (1, 10),
    "plan": (11, 20),
    "patch": (21, 30),
    "validate": (31, 40),
    "deliver": (41, 50),
}
DEFAULT_FIELD = "simplicio_trajectory"
MAX_SHOWN_LINES = 20

TAG_RE = re.compile(r"^<(/?)(simplicio_loop|orient|plan|patch|validate|deliver)>$")
POINT_START_RE = re.compile(r"^\[(?:Ponto|Point)\b")
POINT_RE = re.compile(r"^\[(?:Ponto|Point)\s+(\d{1,4})\s*:\s*[^\]\n]+\]")
SEARCH_RE = re.compile(r"^<{4} SEARCH\s*$")
DIVIDER_RE = re.compile(r"^={4}\s*$")
REPLACE_RE = re.compile(r"^>{4} REPLACE\s*$")
ANY_SEARCH_RE = re.compile(r"^(<+) SEARCH\s*$")
ANY_REPLACE_RE = re.compile(r"^(>+) REPLACE\s*$")
LONG_DIVIDER_RE = re.compile(r"^={5,}\s*$")


class _Scan:
    """Percorre as linhas uma vez e acumula problemas e fatos para as checagens finais."""

    def __init__(self) -> None:
        self.problems: list[str] = []
        self.root_open = False
        self.root_closed = False
        self.current: str | None = None
        self.current_line = 0
        self.opens: dict[str, list[int]] = {p: [] for p in PHASES}
        self.order: list[str] = []
        self.points: list[tuple[int, int, str | None]] = []
        self.outside: list[int] = []
        self.between: list[int] = []
        self.out_tags: list[str] = []
        self.block: str | None = None
        self.block_line = 0
        self.search_lines: list[str] = []
        self.blocks_done = 0

    def add(self, rule: str, detail: str) -> None:
        self.problems.append(f"{rule}: {detail}")

    def run(self, text: str) -> None:
        for no, raw in enumerate(text.splitlines(), 1):
            if self.block is not None:
                self._in_block(no, raw)
                continue
            line = raw.strip()
            if not line:
                continue
            tag = TAG_RE.match(line)
            if tag:
                self._tag(no, tag.group(2), bool(tag.group(1)))
            elif self.current == "patch" and self._patch_marker(no, raw):
                continue
            else:
                self._text(no, line)
        self._finish()

    def _tag(self, no: int, name: str, close: bool) -> None:
        if name == ROOT_TAG:
            self._root_tag(no, close)
        elif close:
            if self.current == name:
                self.current = None
            else:
                aberta = f" (a fase aberta é <{self.current}>)" if self.current else ""
                self.add("tag-sem-par", f"linha {no}: </{name}> sem <{name}> aberta{aberta}")
        else:
            if not self.root_open or self.root_closed:
                self.out_tags.append(name)
            if self.current is not None:
                self.add(
                    "fase-nao-fechada",
                    f"<{self.current}> (linha {self.current_line}) não foi fechada antes de <{name}> (linha {no})",
                )
            self.opens[name].append(no)
            self.order.append(name)
            self.current = name
            self.current_line = no

    def _root_tag(self, no: int, close: bool) -> None:
        if not close:
            if self.root_open:
                self.add("envelope", f"linha {no}: <{ROOT_TAG}> repetida")
            self.root_open = True
        elif not self.root_open:
            self.add("envelope", f"linha {no}: </{ROOT_TAG}> sem <{ROOT_TAG}> aberta")
        elif self.root_closed:
            self.add("envelope", f"linha {no}: </{ROOT_TAG}> repetida")
        else:
            if self.current is not None:
                self.add(
                    "fase-nao-fechada",
                    f"<{self.current}> (linha {self.current_line}) não foi fechada antes de </{ROOT_TAG}> (linha {no})",
                )
                self.current = None
            self.root_closed = True

    def _text(self, no: int, line: str) -> None:
        if not self.root_open or self.root_closed:
            self.outside.append(no)
            return
        if not POINT_START_RE.match(line):
            if self.current is None:
                self.between.append(no)
            return
        found = POINT_RE.match(line)
        if not found:
            self.add("ponto-malformado", f"linha {no}: esperado '[Ponto N: Nome]', veio {line[:40]!r}")
            return
        number = int(found.group(1))
        self.points.append((no, number, self.current))
        if self.current is None:
            self.add("ponto-sem-fase", f"linha {no}: ponto {number} fora de qualquer fase")
            return
        low, high = POINT_RANGES[self.current]
        if not low <= number <= high:
            self.add(
                "ponto-fora-da-fase",
                f"linha {no}: ponto {number} dentro de <{self.current}>, que aceita {low}-{high}",
            )

    def _patch_marker(self, no: int, raw: str) -> bool:
        """Trata marcadores fora de bloco dentro de <patch>. True se a linha foi consumida."""
        if SEARCH_RE.match(raw):
            self.block, self.block_line, self.search_lines = "search", no, []
            return True
        found = ANY_SEARCH_RE.match(raw) or ANY_REPLACE_RE.match(raw)
        if found and len(found.group(1)) != 4:
            self.add(
                "patch-marcador",
                f"linha {no}: marcador de {len(found.group(1))} caracteres ({raw.strip()!r}); "
                "o loop usa 4: '<<<< SEARCH', '====', '>>>> REPLACE'",
            )
            return True
        if LONG_DIVIDER_RE.match(raw):
            self.add("patch-marcador", f"linha {no}: divisor de {len(raw.strip())} caracteres; o loop usa '===='")
            return True
        if DIVIDER_RE.match(raw) or REPLACE_RE.match(raw):
            self.add("patch-bloco", f"linha {no}: {raw.strip()!r} fora de um bloco aberto por '<<<< SEARCH'")
            return True
        return False

    def _in_block(self, no: int, raw: str) -> None:
        if self.block == "search":
            if DIVIDER_RE.match(raw):
                if "\n".join(self.search_lines) == "":
                    self.add("patch-search-vazio", f"linha {self.block_line}: bloco com SEARCH vazio")
                self.block = "replace"
            elif REPLACE_RE.match(raw):
                self.add("patch-bloco", f"linha {no}: '>>>> REPLACE' sem '====' (bloco da linha {self.block_line})")
                self.block = None
            else:
                self.search_lines.append(raw)
        elif REPLACE_RE.match(raw):
            self.blocks_done += 1
            self.block = None

    def _finish(self) -> None:
        if self.block is not None:
            self.add("patch-bloco", f"bloco aberto na linha {self.block_line} nunca fechado com '>>>> REPLACE'")
        if self.current is not None:
            self.add("fase-nao-fechada", f"<{self.current}> (linha {self.current_line}) nunca foi fechada")
        if not self.root_open:
            self.add("envelope", f"falta <{ROOT_TAG}>")
        elif not self.root_closed:
            self.add("envelope", f"falta </{ROOT_TAG}>")
        if self.outside:
            self.add(
                "fora-do-envelope",
                f"{len(self.outside)} linha(s) de texto fora de <{ROOT_TAG}> (primeira na linha {self.outside[0]})",
            )
        if self.out_tags:
            self.add("fase-fora-do-envelope", "fases fora de <simplicio_loop>: " + ", ".join(f"<{t}>" for t in self.out_tags))
        if self.between:
            self.add(
                "texto-fora-das-fases",
                f"{len(self.between)} linha(s) de texto entre fases (primeira na linha {self.between[0]})",
            )
        for phase in PHASES:
            lines = self.opens[phase]
            if not lines:
                self.add("fase-ausente", f"falta <{phase}>")
            elif len(lines) > 1:
                self.add("fase-repetida", f"<{phase}> aparece {len(lines)} vezes (linhas {', '.join(map(str, lines))})")
        found = list(dict.fromkeys(self.order))
        if found != [p for p in PHASES if p in found]:
            self.add(
                "fase-fora-de-ordem",
                f"ordem encontrada: {', '.join(found)}; a ordem do loop é {', '.join(PHASES)}",
            )
        if self.opens["patch"] and self.blocks_done == 0 and self.block is None:
            self.add("patch-sem-bloco", "<patch> sem nenhum bloco '<<<< SEARCH' / '====' / '>>>> REPLACE' completo")

    def complete_problems(self) -> list[str]:
        """Checagens do modo --complete: os 50 pontos, uma vez cada, em ordem crescente."""
        out: list[str] = []
        valid = [(no, n, ph) for no, n, ph in self.points if ph and POINT_RANGES[ph][0] <= n <= POINT_RANGES[ph][1]]
        by_number: dict[int, list[int]] = {}
        for no, n, _ in self.points:
            by_number.setdefault(n, []).append(no)
        for n, lines in sorted(by_number.items()):
            if len(lines) > 1:
                out.append(f"ponto-repetido: ponto {n} aparece {len(lines)} vezes (linhas {', '.join(map(str, lines))})")
        have = {n for _, n, _ in valid}
        for phase in PHASES:
            if not self.opens[phase]:
                continue  # a fase inteira já foi reportada como fase-ausente
            low, high = POINT_RANGES[phase]
            missing = [n for n in range(low, high + 1) if n not in have]
            if missing:
                out.append(f"ponto-ausente: <{phase}> sem os pontos {', '.join(map(str, missing))}")
        numbers = [(no, n) for no, n, _ in valid]
        for (_, before), (no, after) in zip(numbers, numbers[1:]):
            if after < before:
                out.append(f"ponto-fora-de-ordem: ponto {after} (linha {no}) vem depois do ponto {before}")
                break
        return out


def check(text: object, *, complete: bool = False) -> list[str]:
    """Devolve a lista de problemas do envelope; lista vazia significa que a saída segue o loop."""
    if not isinstance(text, str):
        return [f"entrada: a saída deve ser texto, veio {type(text).__name__}"]
    if not text.strip():
        return ["entrada: saída vazia"]
    scan = _Scan()
    scan.run(text)
    return scan.problems + (scan.complete_problems() if complete else [])


def rule_of(problem: str) -> str:
    return problem.split(":", 1)[0]


def _check_file(path: str, complete: bool) -> bool:
    """Imprime o resultado de um arquivo. True se passou."""
    try:
        if path == "-":
            text = sys.stdin.read()
        else:
            with open(path, "rb") as fh:
                text = fh.read().decode("utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        print(f"FALHA  {path}\n  - leitura: não foi possível ler ({exc})")
        return False
    problems = check(text, complete=complete)
    if not problems:
        print(f"OK     {path}")
        return True
    print(f"FALHA  {path}")
    for problem in problems:
        print(f"  - {problem}")
    return False


def _check_jsonl(path: str, field: str, complete: bool) -> bool:
    """Valida o campo de cada linha de um JSONL e imprime a contagem. True se todas passaram."""
    try:
        with open(path, "rb") as fh:
            raw_lines = fh.read().decode("utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as exc:
        print(f"FALHA  {path}\n  - leitura: não foi possível ler ({exc})")
        return False
    total = passed = 0
    failures: list[tuple[int, list[str]]] = []
    reasons: Counter[str] = Counter()
    for no, raw in enumerate(raw_lines, 1):
        if not raw.strip():
            continue
        total += 1
        try:
            row = json.loads(raw)
        except json.JSONDecodeError as exc:
            problems = [f"jsonl: a linha não é JSON ({exc.msg})"]
        else:
            if not isinstance(row, dict):
                problems = ["jsonl: a linha não é um objeto JSON"]
            elif field not in row:
                problems = [f"campo: a linha não tem o campo {field!r}"]
            else:
                problems = check(row[field], complete=complete)
        if problems:
            failures.append((no, problems))
            reasons.update({rule_of(p) for p in problems})
        else:
            passed += 1
    modo = "completo (50 pontos)" if complete else "envelope"
    print(f"{path}: {total} linhas, {passed} passam, {total - passed} falham (campo {field!r}, modo {modo})")
    for no, problems in failures[:MAX_SHOWN_LINES]:
        print(f"  linha {no}: " + " | ".join(problems))
    if len(failures) > MAX_SHOWN_LINES:
        print(f"  ... e mais {len(failures) - MAX_SHOWN_LINES} linha(s) com falha")
    if reasons:
        print("  razões (linhas afetadas):")
        for rule, count in reasons.most_common():
            print(f"    {rule}: {count} de {total}")
    return not failures


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="loop_check.py",
        description="Confere se a saída do modelo segue o envelope do simplicio-loop (50 pontos).",
    )
    ap.add_argument("files", nargs="*", metavar="ARQUIVO", help="saída do modelo em texto ('-' lê a entrada padrão)")
    ap.add_argument("--jsonl", action="append", default=[], metavar="ARQUIVO",
                    help="JSONL cujo campo de saída de cada linha é validado (pode repetir)")
    ap.add_argument("--field", default=DEFAULT_FIELD, help=f"campo de saída do JSONL (padrão: {DEFAULT_FIELD})")
    ap.add_argument("--complete", action="store_true",
                    help="exige também os 50 pontos, cada um uma vez e em ordem crescente")
    args = ap.parse_args(argv)
    if not args.files and not args.jsonl:
        ap.error("informe ao menos um ARQUIVO ou --jsonl ARQUIVO")
    ok = True
    for path in args.files:
        ok = _check_file(path, args.complete) and ok
    for path in args.jsonl:
        ok = _check_jsonl(path, args.field, args.complete) and ok
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
