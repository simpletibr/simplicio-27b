#!/usr/bin/env python3
"""Validador do envelope do simplicio-loop (50 pontos). Só biblioteca padrão, sem rede.

Mede se uma saída de texto segue o formato descrito no README (seção "Output format") e
produzido por generate_dataset.py. Não executa nada do que a saída afirma: "2 passed" ou
"COMMIT_READY" dentro de <validate>/<deliver> são texto, não prova.

Uso:
    python3 scripts/loop_check.py ARQUIVO...            # '-' lê a entrada padrão
    python3 scripts/loop_check.py --jsonl ARQUIVO       # valida o campo de cada linha
    python3 scripts/loop_check.py --complete ARQUIVO    # exige os 50 pontos
Saída: 0 se tudo passou, 1 se alguma saída falhou (ou um arquivo não pôde ser lido, ou um
--jsonl não tem nenhuma linha), 2 em erro de uso. Na API: check(text, complete=False) -> lista
de problemas (vazia = ok). Cada problema é "regra: detalhe"; a regra (texto antes do primeiro
':') serve para contar razões.

Especificação (fonte entre parênteses)
1. A saída é um único <simplicio_loop> ... </simplicio_loop>; nada além de espaço fora dele (README).
2. Dentro dele, as fases <orient>, <plan>, <patch>, <validate>, <deliver>, cada uma uma vez,
   nessa ordem, abertas e fechadas, sem aninhar. Cada tag fica sozinha na linha (README, dataset).
   <orient>, <plan>, <validate> e <deliver> não podem ficar vazias (o README abrevia com "…", que conta).
   "Vazia" inclui só espaço em branco e só caracteres invisíveis (categoria Unicode Cf: U+200B, U+200C,
   U+200D, U+2060, U+FEFF...). Esses caracteres também são ignorados nas pontas de cada linha.
3. Um ponto é uma linha "[Ponto N: Nome]" (dataset) ou "[Point N: Name]" (README, traduzido), com ou
   sem diferença de caixa em "Ponto"/"Point" e com ou sem prefixo de lista ("- ", "* ", "1. ").
   Faixas: orient 1-10, plan 11-20, patch 21-30, validate 31-40, deliver 41-50 (generate_dataset.py).
   Ponto numerado fora da faixa da fase onde aparece, ou fora de qualquer fase, é problema.
   Prefixo de lista aceito: "- ", "* ", "+ ", "• ", "1. ", "1) ". Qualquer outra marcação antes do ponto
   ("**[Ponto 45: x]**", "– [Ponto 45: x]", "> [Ponto ...]", "# [Ponto ...]") é erro de formato
   (ponto-malformado) mesmo dentro da faixa: escolhi rejeitar em vez de aceitar mais prefixos. Uma
   referência no meio de uma frase ("conforme [Ponto 12: x]") não é linha de ponto e é ignorada.
4. <patch> tem um ou mais blocos "<<<< SEARCH" / "====" / ">>>> REPLACE", marcadores de exatamente
   4 caracteres, sozinhos na linha, sem indentação e com um único espaço antes de SEARCH/REPLACE
   (README; benchmarks/harness/harness.py). SEARCH não pode ser vazio nem só espaços e linhas em
   branco (harness: empty_search). REPLACE pode ser vazio (apagar trecho).

Escolhas onde a especificação é ambígua (sempre a leitura mais literal)
- Modo padrão (envelope): o README diz "cada fase é uma lista de pontos", mas o próprio exemplo
  abrevia com "…", e mostra <patch> e <deliver> sem ponto numerado. Por isso o padrão NÃO exige
  que os 50 pontos existam, nem que sejam únicos ou crescentes; só confere o que aparece.
- --complete / complete=True: o system prompt do repo diz "os 50 pontos do Simplicio-Loop". Este modo
  acrescenta: todos os 50 pontos, cada um uma vez, na faixa da sua fase, em ordem crescente.
- O nome do ponto não é conferido (o README traduz "Identificacao de Raiz" por "Root" e o gerador
  usa nomes diferentes nos exemplos manuais e nos gerados); só o número conta.
- Linhas de texto livre dentro de uma fase são permitidas (o README usa "…"). Texto entre as fases, ou
  fora de <simplicio_loop>, é problema.
- <think>: o README manda rodar sem thinking e o treino não tem blocos <think>. Por isso qualquer
  "<think>" ou "</think>" dentro de <simplicio_loop> é problema (think-no-envelope), em qualquer fase ou
  entre fases, sozinho na linha ou no meio dela, e um <think> sem fechamento também (think-nao-fechado).
  Dentro de um bloco SEARCH/REPLACE é conteúdo (um patch no próprio harness.py contém "</think>").
- Marcadores são exatos. O harness aceita de 4 a 7 caracteres; o README exige 4, então 5 a 7 (o estilo
  Git/Aider) é problema em qualquer lugar de <patch>, inclusive dentro de um bloco aberto (onde o
  harness fecharia o bloco com ele). Marcador SEARCH/REPLACE indentado ou com espaçamento diferente de
  um único espaço também é problema em qualquer lugar de <patch>. Marcador SEARCH/REPLACE fora de
  <patch>, ou entre tags <think> e </think> sozinhas na linha, é problema, e um bloco ali não conta como
  patch.
- Dentro de um bloco aberto, o resto é conteúdo até o marcador que o fecha (como no harness): um
  "<<<< SEARCH" exato ou "</patch>" ali dentro não é interpretado.

Limitações conhecidas
- Dentro de um bloco aberto, um "====" INDENTADO é conteúdo, não erro: pode ser o sublinhado de um
  título RST/Markdown no código sendo editado, e o harness também o trata como texto. Já uma linha
  inteira de 5 a 7 "=" é reportada mesmo sendo sublinhado, porque o harness a leria como divisor.
- Marcador de patch precedido de caractere invisível (U+200B...) não é reconhecido como marcador.
- Não confere se o SEARCH existe no arquivo-alvo (isso exige o código; é o que o harness mede).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from collections import Counter
from typing import NamedTuple

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
THINK_RE = re.compile(r"^<(/?)think>$", re.IGNORECASE)
THINK_ANY_RE = re.compile(r"</?think>", re.IGNORECASE)
_LIST_PREFIX = r"(?:(?:[-*+•]|\d+[.)])\s+)?"
POINT_START_RE = re.compile(rf"^{_LIST_PREFIX}\[(?:ponto|point)\b", re.IGNORECASE)
POINT_RE = re.compile(rf"^{_LIST_PREFIX}\[(?:ponto|point)\s+(\d{{1,4}})\s*:\s*[^\]\n]+\]", re.IGNORECASE)
# Linha que começa com marcação (**, __, –, >, #, `) e só então o ponto: parece ponto, mas está fora do formato.
DECORATED_POINT_RE = re.compile(r"^(?:[^\w\s\[\]]|_|\s)*\[(?:ponto|point)\b", re.IGNORECASE)
MARKER_RE = re.compile(
    r"^(?P<indent>[ \t]*)(?:(?P<lt><+)(?P<gap1>[ \t]*)SEARCH|(?P<gt>>+)(?P<gap2>[ \t]*)REPLACE|(?P<eq>={4,7}))[ \t]*$"
)


def _invisible(ch: str) -> bool:
    return ch.isspace() or unicodedata.category(ch) == "Cf"


def _trim(text: str) -> str:
    """strip() que também remove caracteres invisíveis (categoria Cf: U+200B, U+FEFF, U+200C, U+2060...)."""
    start, end = 0, len(text)
    while start < end and _invisible(text[start]):
        start += 1
    while end > start and _invisible(text[end - 1]):
        end -= 1
    return text[start:end]


def _is_blank(text: str) -> bool:
    return all(_invisible(ch) for ch in text)


class _Marker(NamedTuple):
    kind: str  # search | divider | replace
    length: int
    clean: bool  # sem indentação e com o espaço único do formato

    @property
    def exact(self) -> bool:
        return self.clean and self.length == 4

    @property
    def harness_marker(self) -> bool:
        """O que o harness reconheceria: sem indentação, espaço único e 4 a 7 caracteres."""
        return self.clean and 4 <= self.length <= 7


def _marker(raw: str) -> _Marker | None:
    """Classifica uma linha que parece marcador de SEARCH/REPLACE; None se não parece."""
    found = MARKER_RE.match(raw)
    if not found:
        return None
    if found["lt"]:
        kind, run, gap = "search", found["lt"], found["gap1"]
    elif found["gt"]:
        kind, run, gap = "replace", found["gt"], found["gap2"]
    else:
        kind, run, gap = "divider", found["eq"], " "
    return _Marker(kind, len(run), not found["indent"] and gap == " ")


class _Scan:
    """Percorre as linhas uma vez e acumula problemas e fatos para as checagens finais."""

    def __init__(self) -> None:
        self.problems: list[str] = []
        self.root_open = False
        self.root_closed = False
        self.current: str | None = None
        self.current_line = 0
        self.in_think = False
        self.opens: dict[str, list[int]] = {p: [] for p in PHASES}
        self.filled: dict[str, int] = {p: 0 for p in PHASES}
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
                self._count(raw)
                self._in_block(no, raw)
                continue
            line = _trim(raw)
            if not line:
                continue
            inside = self.root_open and not self.root_closed
            if inside and THINK_ANY_RE.search(line):
                self._think(no, line)
                if THINK_RE.match(line):
                    continue
            tag = TAG_RE.match(line)
            if tag:
                self._tag(no, tag.group(2), bool(tag.group(1)))
                continue
            self._count(line)
            if self.current == "patch" and not self.in_think and self._patch_marker(no, raw):
                continue
            mark = _marker(raw)
            if inside and mark is not None and mark.kind != "divider":
                rule = "patch-no-think" if self.in_think else "patch-fora-do-patch"
                where = "dentro de um <think>" if self.in_think else "fora de <patch>"
                self.add(rule, f"linha {no}: marcador {line!r} {where}")
                continue
            self._text(no, line)
        self._finish()

    def _count(self, line: str) -> None:
        if self.current is not None and not _is_blank(line):
            self.filled[self.current] += 1

    def _think(self, no: int, line: str) -> None:
        onde = f"dentro de <{self.current}>" if self.current else f"dentro de <{ROOT_TAG}>, entre fases"
        self.add("think-no-envelope", f"linha {no}: <think> ou </think> {onde}; o envelope não tem blocos de thinking")
        tag = THINK_RE.match(line)
        if tag:
            self.in_think = not tag.group(1)

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
            if DECORATED_POINT_RE.match(line):
                self.add(
                    "ponto-malformado",
                    f"linha {no}: ponto precedido de marcação ({line[:40]!r}); só '- ', '* ', '+ ', '• ', "
                    "'1. ' e '1) ' podem vir antes de '[Ponto N: Nome]'",
                )
            elif self.current is None:
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

    def _bad_marker(self, no: int, raw: str, mark: _Marker) -> None:
        why = []
        if mark.length != 4:
            why.append(f"{mark.length} caracteres (o loop usa 4)")
        if not mark.clean:
            why.append("indentado ou com espaçamento fora do formato")
        self.add(
            "patch-marcador",
            f"linha {no}: marcador {raw.strip()!r}: {'; '.join(why)}; "
            "o formato exato é '<<<< SEARCH', '====', '>>>> REPLACE', sozinho na linha e sem indentação",
        )

    def _patch_marker(self, no: int, raw: str) -> bool:
        """Trata marcadores fora de bloco dentro de <patch>. True se a linha foi consumida."""
        mark = _marker(raw)
        if mark is None:
            return False
        if not mark.exact:
            self._bad_marker(no, raw, mark)
        elif mark.kind == "search":
            self.block, self.block_line, self.search_lines = "search", no, []
        else:
            self.add("patch-bloco", f"linha {no}: {raw.strip()!r} fora de um bloco aberto por '<<<< SEARCH'")
        return True

    def _in_block(self, no: int, raw: str) -> None:
        mark = _marker(raw)
        if mark is not None and mark.kind == "divider" and not mark.clean:
            mark = None  # divisor indentado dentro de bloco é conteúdo (ver "Limitações conhecidas")
        if mark is not None:
            if not mark.exact:
                self._bad_marker(no, raw, mark)
            if mark.harness_marker and self.block == "search" and mark.kind == "divider":
                if all(_is_blank(line) for line in self.search_lines):
                    self.add("patch-search-vazio", f"linha {self.block_line}: bloco com SEARCH vazio ou só com espaços")
                self.block = "replace"
                return
            if mark.harness_marker and self.block == "search" and mark.kind == "replace":
                self.add("patch-bloco", f"linha {no}: '>>>> REPLACE' sem '====' (bloco da linha {self.block_line})")
                self.block = None
                return
            if mark.harness_marker and self.block == "replace" and mark.kind == "replace":
                self.blocks_done += 1
                self.block = None
                return
        if self.block == "search":
            self.search_lines.append(raw)

    def _finish(self) -> None:
        if self.block is not None:
            self.add("patch-bloco", f"bloco aberto na linha {self.block_line} nunca fechado com '>>>> REPLACE'")
        if self.current is not None:
            self.add("fase-nao-fechada", f"<{self.current}> (linha {self.current_line}) nunca foi fechada")
        if self.in_think:
            self.add("think-nao-fechado", "<think> aberto dentro do envelope nunca foi fechado com </think>")
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
            elif phase != "patch" and self.filled[phase] == 0:
                self.add("fase-vazia", f"<{phase}> (linha {lines[0]}) não tem conteúdo")
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
    if _is_blank(text):
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
    if total == 0:
        print(f"FALHA  {path}\n  - jsonl: nenhuma linha para validar (arquivo vazio ou só com linhas em branco)")
        return False
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
