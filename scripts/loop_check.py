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

Gramática (README, generate_dataset.py, benchmarks/harness/harness.py)
A saída é um único <simplicio_loop> ... </simplicio_loop>; nada além de espaço fora dele. Dentro
dele, toda linha não vazia casa com EXATAMENTE uma destas formas; qualquer outra é "texto-livre":
 (a) tag sozinha na linha, sem indentação: <simplicio_loop>, <orient>, <plan>, <patch>, <validate>,
     <deliver> e os fechamentos. As cinco fases aparecem uma vez cada, nessa ordem, abertas e
     fechadas, sem aninhar.
 (b) ponto exato: "[Ponto N: Nome]" ou "[Point N: Name]" no começo da linha. N tem 1 ou 2 dígitos,
     sem zero à esquerda; um espaço depois de Ponto/Point e depois de ':'; nome sem colchetes e sem
     espaço nas pontas; depois do ']' só o fim da linha ou UM espaço e o texto do ponto na mesma
     linha. A faixa da fase vale: orient 1-10, plan 11-20, patch 21-30, validate 31-40, deliver 41-50.
     Qualquer variação é "ponto-malformado": caixa diferente, espaço extra, N de 3 dígitos ou com zero
     à esquerda, prefixo de lista ("- ", "* ", "1. "), marcação antes ou colada depois do ']'
     ("**[Ponto 1: x]**", "– [Ponto 1: x]"), indentação.
 (c) em <patch>: marcadores "<<<< SEARCH", "====" e ">>>> REPLACE" de exatamente 4 caracteres,
     sozinhos na linha, sem indentação e com um único espaço antes de SEARCH/REPLACE; dentro de um
     bloco, as linhas de conteúdo; fora de bloco, só linhas em branco (e os pontos (b) e o "…").
     SEARCH não pode ser vazio (nem só espaços ou invisíveis). REPLACE pode ser vazio.
 (d) a única linha de conteúdo solto é "…" (reticências, U+2026), como no README.
Linhas em branco, ou só com espaço e caracteres invisíveis (categoria Unicode Cf: U+200B, U+200C,
U+200D, U+2060, U+FEFF...), são ignoradas. Um U+FEFF no início do texto (BOM) é descartado.
Cada fase precisa de ao menos uma linha de conteúdo (ponto, "…", ou marcador/conteúdo em <patch>);
as tags não contam, e linha só de pontuação ("---", "***", "...") é texto-livre e também não conta.

Interpretação de (b) (a única ambígua): o enunciado diz "[Ponto N: texto]", mas o README
("[Point 1: Root] Root confirmed at ...") e todas as 4.750 linhas de ponto do dataset trazem o texto do
ponto na mesma linha, depois do ']'. Por isso o texto depois de UM espaço é aceito; só o que vem
colado ao ']' é erro. A leitura "nada depois do ']'" rejeitaria 100% dos exemplos reais.

<think>: o README manda rodar sem thinking e o treino não tem blocos <think>. Toda linha dentro do
envelope é normalizada (NFKC, sem espaços e sem caracteres Cf, sem diferença de caixa) e, se contiver
"<think>" ou "</think>", é "think-no-envelope" (cobre "< think >", "<THINK>" e um U+200B no
meio da tag, sozinha ou no meio da linha); um <think> sem fechamento também é "think-nao-fechado".
Dentro de um bloco SEARCH/REPLACE é conteúdo (um patch no próprio harness.py contém "</think>").
Marcador SEARCH/REPLACE fora de <patch>, ou entre tags <think>, é erro e não conta como patch.

Escolhas onde a especificação é ambígua (sempre a leitura mais literal)
- Modo padrão (envelope): o README diz "cada fase é uma lista de pontos", mas o próprio exemplo
  abrevia com "…", e mostra <patch> e <deliver> sem ponto numerado. Por isso o padrão NÃO exige
  que os 50 pontos existam, nem que sejam únicos ou crescentes; só confere o que aparece.
- --complete / complete=True: o system prompt do repo diz "os 50 pontos do Simplicio-Loop". Este modo
  acrescenta: todos os 50 pontos, cada um uma vez, na faixa da sua fase, em ordem crescente.
- O nome do ponto não é conferido (o README traduz "Identificacao de Raiz" por "Root" e o gerador
  usa nomes diferentes nos exemplos manuais e nos gerados); só o número conta.
- Marcadores são exatos. O harness aceita de 4 a 7 caracteres; o README exige 4, então 5 a 7 (o estilo
  Git/Aider) é erro em qualquer lugar de <patch>, inclusive dentro de um bloco aberto (onde o harness
  fecharia o bloco com ele). Marcador SEARCH/REPLACE indentado ou com espaçamento diferente de um
  único espaço também é erro em qualquer lugar de <patch>.
- Dentro de um bloco aberto, o resto é conteúdo até o marcador que o fecha (como no harness): um
  "<<<< SEARCH" exato ou "</patch>" ali dentro não é interpretado.

Limitações conhecidas
- Dentro de um bloco aberto, um "====" INDENTADO é conteúdo, não erro: pode ser o sublinhado de um
  título RST/Markdown no código sendo editado, e o harness também o trata como texto. Já uma linha
  inteira de 5 a 7 "=" é reportada mesmo sendo sublinhado, porque o harness a leria como divisor.
- Texto livre antes de <simplicio_loop> ou depois de </simplicio_loop> é um único problema agregado.
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
ELLIPSIS = "…"
MAX_SHOWN_LINES = 20
MAX_FREE_TEXT = 10

TAG_RE = re.compile(r"^<(/?)(simplicio_loop|orient|plan|patch|validate|deliver)>$")
POINT_RE = re.compile(r"^\[(Ponto|Point) ([1-9]\d?): ([^\[\]\s](?:[^\[\]]*[^\[\]\s])?)\](?: (\S.*))?$")
# Parece uma linha de ponto (com prefixo de lista ou marcação na frente, ou escrita fora do formato).
POINT_LIKE_RE = re.compile(r"^(?:\d+[.)])?(?:[^\w\s\[\]]|_|\s)*\[(?:ponto|point)\b", re.IGNORECASE)
MARKER_RE = re.compile(
    r"^(?P<indent>[ \t]*)(?:(?P<lt><+)(?P<gap1>[ \t]*)SEARCH|(?P<gt>>+)(?P<gap2>[ \t]*)REPLACE|(?P<eq>={4,7}))[ \t]*$"
)


def _invisible(ch: str) -> bool:
    return ch.isspace() or unicodedata.category(ch) == "Cf"


def _is_blank(text: str) -> bool:
    return all(_invisible(ch) for ch in text)


def _normalize(text: str) -> str:
    """NFKC, sem espaços e sem caracteres Cf (U+200B, U+FEFF...), sem diferença de caixa."""
    return "".join(ch for ch in unicodedata.normalize("NFKC", text) if not _invisible(ch)).casefold()


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
        self.free: list[tuple[int, str, str | None]] = []
        self.out_tags: list[str] = []
        self.block: str | None = None
        self.block_line = 0
        self.search_lines: list[str] = []
        self.blocks_done = 0

    def add(self, rule: str, detail: str) -> None:
        self.problems.append(f"{rule}: {detail}")

    def run(self, text: str) -> None:
        if text.startswith("\ufeff"):
            text = text[1:]
        for no, raw in enumerate(text.splitlines(), 1):
            if self.block is not None:
                self._count(raw)
                self._in_block(no, raw)
                continue
            if _is_blank(raw):
                continue
            line = raw.rstrip()
            inside = self.root_open and not self.root_closed
            if inside:
                norm = _normalize(line)
                if "<think>" in norm or "</think>" in norm:
                    self._think(no, line, norm)
                    if norm in ("<think>", "</think>"):
                        continue
            tag = TAG_RE.match(line)
            if tag:
                self._tag(no, tag.group(2), bool(tag.group(1)))
            elif not inside:
                self.outside.append(no)
            else:
                self._line(no, raw, line)
        self._finish()

    def _count(self, line: str) -> None:
        if self.current is not None and not _is_blank(line):
            self.filled[self.current] += 1

    def _think(self, no: int, line: str, norm: str) -> None:
        onde = f"dentro de <{self.current}>" if self.current else f"dentro de <{ROOT_TAG}>, entre fases"
        self.add("think-no-envelope", f"linha {no}: <think> ou </think> {onde} ({line[:40]!r}); o envelope não tem thinking")
        if norm == "<think>":
            self.in_think = True
        elif norm == "</think>":
            self.in_think = False

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

    def _line(self, no: int, raw: str, line: str) -> None:
        """Linha não vazia, fora de bloco, dentro do envelope e que não é tag: tem de ser uma forma aceita."""
        if line == ELLIPSIS:
            self._count(line)
            return
        if self.current == "patch" and not self.in_think and self._patch_marker(no, raw):
            self._count(line)
            return
        mark = _marker(raw)
        if mark is not None and mark.kind != "divider":
            rule = "patch-no-think" if self.in_think else "patch-fora-do-patch"
            where = "entre tags <think>" if self.in_think else "fora de <patch>"
            self.add(rule, f"linha {no}: marcador {line!r} {where}")
            return
        found = POINT_RE.match(line)
        if found:
            self._point(no, int(found.group(2)))
            self._count(line)
        elif POINT_LIKE_RE.match(line):
            self.add(
                "ponto-malformado",
                f"linha {no}: ponto fora do formato exato '[Ponto N: Nome]' (N de 1 a 2 dígitos, sem prefixo, "
                f"sem espaço extra, sem marcação antes ou colada depois): {line[:40]!r}",
            )
        else:
            self.free.append((no, line, self.current))

    def _point(self, no: int, number: int) -> None:
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
        for no, line, phase in self.free[:MAX_FREE_TEXT]:
            onde = f"dentro de <{phase}>" if phase else "entre fases"
            self.add("texto-livre", f"linha {no}: {onde}, fora das formas aceitas (tag, ponto, '…', patch): {line[:40]!r}")
        if len(self.free) > MAX_FREE_TEXT:
            self.add("texto-livre", f"e mais {len(self.free) - MAX_FREE_TEXT} linha(s) fora das formas aceitas")
        for phase in PHASES:
            lines = self.opens[phase]
            if not lines:
                self.add("fase-ausente", f"falta <{phase}>")
            elif len(lines) > 1:
                self.add("fase-repetida", f"<{phase}> aparece {len(lines)} vezes (linhas {', '.join(map(str, lines))})")
            elif self.filled[phase] == 0:
                self.add("fase-vazia", f"<{phase}> (linha {lines[0]}) não tem conteúdo (ponto, '…' ou, em <patch>, bloco)")
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
