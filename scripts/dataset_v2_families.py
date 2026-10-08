"""Familias de bug sinteticas do dataset v2 (usado por scripts/build_dataset_v2.py).

Biblioteca padrao apenas, sem rede e sem modelo. Cada "forma" (shape) e uma funcao que recebe um
dominio, um gerador deterministico e o numero da variante e devolve um Draft: um arquivo Python
pequeno com UM bug (marcado por @@BUG@@), o trecho com defeito, o trecho corrigido e testes pytest
escritos como dados (Test/Step). Os testes sao renderizados em codigo pytest e tambem executados
em processo (observe) para extrair a falha observada sem depender do texto do pytest.

Nada aqui usa o modulo random: toda escolha vem de Rng, um fluxo de sha256 indexado pela chave
da forma, entao a saida nao depende de PYTHONHASHSEED nem da versao do Python.
"""

from __future__ import annotations

import ast
import hashlib
import keyword
import textwrap
from dataclasses import dataclass, field
from string import Template
from typing import Callable, NamedTuple

MARKER = "@@BUG@@"


# --------------------------------------------------------------------------------------------
# Escolhas deterministicas
# --------------------------------------------------------------------------------------------
class Rng:
    """Fluxo pseudoaleatorio deterministico: sha256(chave:contador). Sem estado global."""

    def __init__(self, *key: object) -> None:
        self.key = ":".join(str(k) for k in key)
        self.counter = 0

    def _next(self) -> int:
        digest = hashlib.sha256(f"{self.key}:{self.counter}".encode()).digest()
        self.counter += 1
        return int.from_bytes(digest[:8], "big")

    def below(self, n: int) -> int:
        return self._next() % n

    def int(self, low: int, high: int) -> int:
        return low + self.below(high - low + 1)

    def pick(self, seq):
        return seq[self.below(len(seq))]

    def sample(self, seq, n: int) -> list:
        pool = list(seq)
        out = []
        for _ in range(n):
            out.append(pool.pop(self.below(len(pool))))
        return out

    def ints(self, n: int, low: int, high: int) -> list[int]:
        """n inteiros distintos em [low, high]."""
        return self.sample(range(low, high + 1), n)


class Domain(NamedTuple):
    item: str    # singular, usado em nomes e textos
    items: str   # plural
    num: str     # atributo numerico
    txt: str     # atributo de texto


DOMAINS: tuple[Domain, ...] = tuple(Domain(*row) for row in (
    ("invoice", "invoices", "amount", "customer"),
    ("ticket", "tickets", "priority", "title"),
    ("order", "orders", "total", "buyer"),
    ("shipment", "shipments", "weight", "carrier"),
    ("sensor", "sensors", "reading", "zone"),
    ("student", "students", "score", "nickname"),
    ("player", "players", "points", "handle"),
    ("account", "accounts", "balance", "owner"),
    ("booking", "bookings", "nights", "guest"),
    ("task", "tasks", "hours", "assignee"),
    ("product", "products", "price", "sku"),
    ("patient", "patients", "dose", "ward"),
    ("vehicle", "vehicles", "mileage", "plate"),
    ("recipe", "recipes", "calories", "chef"),
    ("course", "courses", "credits", "teacher"),
    ("payment", "payments", "fee", "method"),
    ("employee", "employees", "salary", "department"),
    ("book", "books", "pages", "author"),
    ("flight", "flights", "duration", "airline"),
    ("event", "events", "capacity", "venue"),
    ("coupon", "coupons", "discount", "code"),
    ("donation", "donations", "value", "donor"),
    ("lesson", "lessons", "minutes", "topic"),
    ("package", "packages", "volume", "label"),
    ("meter", "meters", "usage", "tariff"),
    ("server", "servers", "load", "region"),
    ("survey", "surveys", "votes", "question"),
    ("article", "articles", "words", "headline"),
    ("playlist", "playlists", "tracks", "genre"),
    ("reservation", "reservations", "guests", "table"),
    ("contract", "contracts", "months", "client"),
    ("stock", "stocks", "shares", "symbol"),
    ("parcel", "parcels", "distance", "depot"),
    ("workout", "workouts", "reps", "exercise"),
    ("album", "albums", "sales", "artist"),
    ("device", "devices", "battery", "model"),
    ("harvest", "harvests", "output", "crop"),
    ("sample", "samples", "density", "batch"),
    ("delivery", "deliveries", "stops", "driver"),
    ("subscription", "subscriptions", "seats", "plan"),
    ("warehouse", "warehouses", "pallets", "aisle"),
    ("exam", "exams", "grade", "subject"),
    ("trip", "trips", "kilometers", "route"),
    ("ingredient", "ingredients", "grams", "brand"),
    ("meeting", "meetings", "attendees", "agenda"),
    ("refund", "refunds", "cents", "reason"),
    ("deposit", "deposits", "cents_in", "branch"),
    ("repair", "repairs", "parts", "mechanic"),
    ("customer", "customers", "visits", "segment"),
    ("shift", "shifts", "overtime", "crew"),
))
for _d in DOMAINS:
    for _word in _d:
        assert _word.isidentifier() and not keyword.iskeyword(_word), _word

NAMES = ("ana", "bruno", "carla", "diego", "elisa", "fabio", "gina", "hugo", "iris", "jonas",
         "karla", "luis", "marta", "nuno", "olga", "paulo", "rita", "sergio", "tania", "vitor")
WORDS = ("alpha", "bravo", "cedar", "delta", "ember", "fjord", "gamma", "harbor", "indigo", "juniper",
         "kilo", "lumen", "maple", "nectar", "oasis", "pixel", "quartz", "river", "sierra", "tundra")


# --------------------------------------------------------------------------------------------
# Testes como dados
# --------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Step:
    kind: str          # let | do | eq | raises | ok
    a: str = ""
    b: object = None


def Let(var: str, expr: str) -> Step:
    return Step("let", var, expr)


def Do(stmt: str) -> Step:
    return Step("do", stmt)


def Eq(expr: str, expected: object) -> Step:
    return Step("eq", expr, expected)


def Raises(expr: str, exc: str) -> Step:
    return Step("raises", expr, exc)


def Ok(expr: str) -> Step:
    return Step("ok", expr)


@dataclass(frozen=True)
class Test:
    name: str
    steps: tuple[Step, ...]

    def __init__(self, name: str, steps) -> None:
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "steps", tuple(steps))


def call(fn: str, *args: object) -> str:
    return f"{fn}({', '.join(repr(a) for a in args)})"


def render_test(test: Test) -> str:
    lines = [f"def test_{test.name}():"]
    for step in test.steps:
        if step.kind == "let":
            lines.append(f"    {step.a} = {step.b}")
        elif step.kind == "do":
            lines.append(f"    {step.a}")
        elif step.kind == "eq":
            lines.append(f"    assert {step.a} == {step.b!r}")
        elif step.kind == "raises":
            lines.append(f"    with pytest.raises({step.b}):")
            lines.append(f"        {step.a}")
        elif step.kind == "ok":
            lines.append(f"    assert {step.a}")
        else:
            raise ValueError(step.kind)
    return "\n".join(lines)


def _short(text: str, limit: int = 110) -> str:
    return text if len(text) <= limit else text[: limit - 3] + "..."


def observe(source: str, tests: list[Test]) -> dict[str, str | None]:
    """Roda os passos dos testes em processo contra `source`, na ordem, no mesmo modulo (como o
    pytest). Devolve {nome: None se passou, senao a mensagem da primeira falha}. O codigo e gerado
    por este script e nao tem laco infinito nem E/S."""
    ns: dict = {"__name__": "observed_module"}
    exec(compile(source, "<observed>", "exec"), ns)
    out: dict[str, str | None] = {}
    for test in tests:
        env = dict(ns)
        msg: str | None = None
        for step in test.steps:
            try:
                if step.kind == "let":
                    env[step.a] = eval(step.b, env)
                elif step.kind == "do":
                    exec(step.a, env)
                elif step.kind == "eq":
                    got = eval(step.a, env)
                    if not got == step.b:
                        msg = _short(f"{step.a} devolveu {got!r}; esperado {step.b!r}")
                elif step.kind == "raises":
                    try:
                        eval(step.a, env)
                    except BaseException as exc:  # noqa: BLE001 - qualquer excecao e dado
                        if type(exc).__name__ != step.b:
                            msg = _short(f"{step.a} levantou {type(exc).__name__} em vez de {step.b}")
                    else:
                        msg = _short(f"{step.a} nao levantou {step.b}")
                elif step.kind == "ok":
                    if not eval(step.a, env):
                        msg = _short(f"{step.a} e falso")
            except Exception as exc:  # noqa: BLE001
                msg = _short(f"{step.a if step.kind != 'let' else step.b} levantou {type(exc).__name__}: {exc}")
            if msg is not None:
                break
        out[f"test_{test.name}"] = msg
    return out


# --------------------------------------------------------------------------------------------
# Rascunho de um exemplo
# --------------------------------------------------------------------------------------------
@dataclass
class Draft:
    family: str
    shape: str
    stem: str
    chunk: str                 # codigo do alvo, com uma linha so com @@BUG@@
    bug: str                   # trecho com defeito (sem indentacao do marcador)
    fix: str                   # trecho corrigido
    tests: list[Test]
    fn: str                    # simbolo alvo (funcao ou classe)
    behavior: str              # "ela deve ..." (comportamento esperado, sem entregar a correcao)
    cause: str
    fix_desc: str
    api: str                   # o que a correcao usa
    lesson: str
    edge: str
    perf: str
    state: str = "Sem estado global: o resultado depende so dos argumentos."
    helpers: list[tuple[str, list[Test]]] = field(default_factory=list)


def T(text: str, **kw: object) -> str:
    return Template(textwrap.dedent(text)).substitute(**kw).strip("\n")


ShapeFn = Callable[[Domain, Rng, int], Draft]
SHAPES: dict[str, list[tuple[str, ShapeFn]]] = {}


def shape(family: str, name: str):
    def deco(fn: ShapeFn) -> ShapeFn:
        SHAPES.setdefault(family, []).append((name, fn))
        return fn

    return deco


def mk(family: str, shape_name: str, d: Domain, stem: str, chunk: str, bug: str, fix: str,
       tests: list[Test], fn: str, **text: str) -> Draft:
    assert MARKER in chunk
    return Draft(family=family, shape=shape_name, stem=stem, chunk=chunk,
                 bug=textwrap.dedent(bug).strip("\n"), fix=textwrap.dedent(fix).strip("\n"),
                 tests=tests, fn=fn, **text)


# --------------------------------------------------------------------------------------------
# Funcoes auxiliares corretas (ruido realista; cada uma com o seu proprio teste verde)
# --------------------------------------------------------------------------------------------
def _h_clamp(d: Domain, r: Rng):
    fn = f"clamp_{d.num}"
    code = T('''
        def $fn(value, low, high):
            """Limita $num ao intervalo [low, high]."""
            return max(low, min(high, value))
    ''', fn=fn, num=d.num)
    return fn, code, [Test(f"{fn}_limits", [Eq(call(fn, 15, 0, 10), 10), Eq(call(fn, -3, 0, 10), 0),
                                              Eq(call(fn, 4, 0, 10), 4)])]


def _h_label(d: Domain, r: Rng):
    fn = f"{d.item}_label"
    code = T('''
        def $fn(number):
            """Rotulo com tres digitos para um $item."""
            return "$item-" + str(number).zfill(3)
    ''', fn=fn, item=d.item)
    return fn, code, [Test(f"{fn}_padding", [Eq(call(fn, 7), f"{d.item}-007"), Eq(call(fn, 1234), f"{d.item}-1234")])]


def _h_unique(d: Domain, r: Rng):
    fn = f"unique_{d.items}"
    code = T('''
        def $fn(values):
            """Valores distintos em ordem crescente."""
            return sorted(set(values))
    ''', fn=fn)
    vals = [r.int(1, 9) for _ in range(6)]
    return fn, code, [Test(f"{fn}_dedup", [Eq(call(fn, vals), sorted(set(vals))), Eq(call(fn, []), [])])]


def _h_chunks(d: Domain, r: Rng):
    fn = f"{d.items}_in_groups"
    code = T('''
        def $fn(values, size):
            """Divide a lista em grupos de ate size elementos."""
            return [values[i:i + size] for i in range(0, len(values), size)]
    ''', fn=fn)
    vals = r.ints(7, 10, 60)
    return fn, code, [Test(f"{fn}_split", [Eq(call(fn, vals, 3), [vals[0:3], vals[3:6], vals[6:7]])])]


def _h_mean(d: Domain, r: Rng):
    fn = f"mean_{d.num}"
    code = T('''
        def $fn(values):
            """Media de $num; lista vazia da 0.0."""
            if not values:
                return 0.0
            return sum(values) / len(values)
    ''', fn=fn, num=d.num)
    return fn, code, [Test(f"{fn}_basic", [Eq(call(fn, [2, 4, 6]), 4.0), Eq(call(fn, []), 0.0)])]


def _h_first(d: Domain, r: Rng):
    fn = f"first_{d.item}"
    code = T('''
        def $fn(values):
            """Primeiro $item ou None quando nao ha nenhum."""
            return values[0] if values else None
    ''', fn=fn, item=d.item)
    w = r.pick(WORDS)
    return fn, code, [Test(f"{fn}_basic", [Eq(call(fn, [w, "x"]), w), Eq(call(fn, []), None)])]


def _h_join(d: Domain, r: Rng):
    fn = f"join_{d.txt}s"
    code = T('''
        def $fn(parts):
            """Junta as partes nao vazias com barra."""
            return "/".join(p for p in parts if p)
    ''', fn=fn)
    a, b = r.sample(WORDS, 2)
    return fn, code, [Test(f"{fn}_skips_empty", [Eq(call(fn, [a, "", b]), f"{a}/{b}")])]


def _h_gap(d: Domain, r: Rng):
    fn = f"{d.num}_gap"
    code = T('''
        def $fn(left, right):
            """Diferenca absoluta entre dois valores de $num."""
            return abs(left - right)
    ''', fn=fn, num=d.num)
    return fn, code, [Test(f"{fn}_symmetric", [Eq(call(fn, 3, 10), 7), Eq(call(fn, 10, 3), 7)])]


def _h_share(d: Domain, r: Rng):
    fn = f"{d.item}_share"
    code = T('''
        def $fn(part, whole):
            """Percentual de part em whole, com uma casa decimal."""
            if whole == 0:
                return 0.0
            return round(100 * part / whole, 1)
    ''', fn=fn)
    return fn, code, [Test(f"{fn}_basic", [Eq(call(fn, 1, 8), 12.5), Eq(call(fn, 5, 0), 0.0)])]


def _h_count(d: Domain, r: Rng):
    fn = f"count_{d.txt}"
    code = T('''
        def $fn(values, target):
            """Quantas vezes target aparece em values."""
            return sum(1 for v in values if v == target)
    ''', fn=fn)
    a, b = r.sample(WORDS, 2)
    return fn, code, [Test(f"{fn}_basic", [Eq(call(fn, [a, b, a, a], a), 3), Eq(call(fn, [b], a), 0)])]


def _h_initials(d: Domain, r: Rng):
    fn = f"{d.txt}_initials"
    code = T('''
        def $fn(text):
            """Iniciais maiusculas de cada palavra."""
            return "".join(word[0].upper() for word in text.split())
    ''', fn=fn)
    a, b = r.sample(WORDS, 2)
    return fn, code, [Test(f"{fn}_basic", [Eq(call(fn, f"{a} {b}"), a[0].upper() + b[0].upper())])]


def _h_last(d: Domain, r: Rng):
    fn = f"newest_{d.item}"
    code = T('''
        def $fn(values):
            """Ultimo $item da lista ou None."""
            if not values:
                return None
            return values[-1]
    ''', fn=fn, item=d.item)
    vals = r.ints(3, 1, 50)
    return fn, code, [Test(f"{fn}_basic", [Eq(call(fn, vals), vals[-1]), Eq(call(fn, []), None)])]


HELPERS = (_h_clamp, _h_label, _h_unique, _h_chunks, _h_mean, _h_first, _h_join, _h_gap, _h_share,
           _h_count, _h_initials, _h_last)


# --------------------------------------------------------------------------------------------
# Familia 1: off_by_one
# --------------------------------------------------------------------------------------------
FAM = "off_by_one"


@shape(FAM, "first_n_total")
def _off_first_n(d, r, k):
    fn = r.pick([f"total_first_{d.items}", f"sum_first_{d.num}", f"{d.num}_of_first"])
    vals = r.ints(6, 2, 40)
    n = r.int(2, 4)
    if k % 2 == 0:
        chunk = T('''
            def $fn(values, n):
                """Soma o $num dos n primeiros $items."""
                $m
        ''', fn=fn, num=d.num, items=d.items, m=MARKER)
        bug, fix = "return sum(values[:n - 1])", "return sum(values[:n])"
        cause = "o fatiamento values[:n - 1] descarta o ultimo dos n valores pedidos"
        guard = Test("empty_list", [Eq(call(fn, [], 3), 0)])
    else:
        chunk = T('''
            def $fn(values, n):
                """Soma o $num dos n primeiros $items."""
                total = 0
                $m
                    total += values[i]
                return total
        ''', fn=fn, num=d.num, items=d.items, m=MARKER)
        bug, fix = "for i in range(1, n):", "for i in range(n):"
        cause = "range(1, n) comeca no indice 1 e ignora o primeiro valor"
        guard = Test("single_zero", [Eq(call(fn, [5], 0), 0)])
    tests = [Test("first_n", [Eq(call(fn, vals, n), sum(vals[:n]))]),
             Test("all_values", [Eq(call(fn, vals, len(vals)), sum(vals))]),
             guard]
    return mk(FAM, "first_n_total", d, f"{d.item}_totals", chunk, bug, fix, tests, fn,
              behavior=f"somar o {d.num} dos n primeiros {d.items} da lista",
              cause=cause, fix_desc="ajustar o limite da soma para cobrir exatamente n valores",
              api="fatiamento e range nativos",
              lesson="Em somas dos n primeiros, conferir os dois extremos: values[:n] cobre n itens e range(n) comeca em 0.",
              edge=f"n igual ao tamanho da lista, n menor que o tamanho e lista vazia (n={n} e {len(vals)})",
              perf="Mesma complexidade O(n): a correcao troca um limite, sem passada extra.")


@shape(FAM, "page_slice")
def _off_page(d, r, k):
    fn = r.pick([f"page_of_{d.items}", f"{d.items}_page", f"paginate_{d.items}"])
    size = r.int(2, 4)
    entries = r.sample(NAMES, 9)
    if k % 2 == 0:
        chunk = T('''
            def $fn(entries, page, size):
                """Devolve os $items da pagina pedida; as paginas comecam em 1."""
                $m
                return entries[start:start + size]
        ''', fn=fn, items=d.items, m=MARKER)
        bug, fix = "start = page * size", "start = (page - 1) * size"
        cause = "start = page * size pula uma pagina inteira porque as paginas comecam em 1"
        fix_desc = "calcular o inicio como (page - 1) * size"
    else:
        chunk = T('''
            def $fn(entries, page, size):
                """Devolve os $items da pagina pedida; as paginas comecam em 1."""
                start = (page - 1) * size
                $m
        ''', fn=fn, items=d.items, m=MARKER)
        bug, fix = "return entries[start:start + size - 1]", "return entries[start:start + size]"
        cause = "o fim start + size - 1 devolve um item a menos por pagina"
        fix_desc = "usar start + size como fim da fatia"
    last = (len(entries) + size - 1) // size
    tests = [Test("first_page", [Eq(call(fn, entries, 1, size), entries[:size])]),
             Test("second_page", [Eq(call(fn, entries, 2, size), entries[size:2 * size])]),
             Test("last_partial_page", [Eq(call(fn, entries, last, size), entries[(last - 1) * size:last * size])]),
             Test("page_past_the_end", [Eq(call(fn, entries, last + 3, size), [])])]
    return mk(FAM, "page_slice", d, f"{d.items}_paging", chunk, bug, fix, tests, fn,
              behavior=f"devolver os {d.items} da pagina pedida, com paginas contadas a partir de 1",
              cause=cause, fix_desc=fix_desc, api="fatiamento nativo",
              lesson="Paginas costumam ser base 1 e indices base 0: o deslocamento e (page - 1) * size e a fatia termina em start + size.",
              edge=f"primeira pagina, segunda, ultima pagina parcial ({len(entries)} itens, tamanho {size}) e pagina alem do fim",
              perf="O(size) por pagina; a correcao nao muda o custo.")


@shape(FAM, "span_days")
def _off_span(d, r, k):
    unit_pt, unit_en = r.pick([("dias", "day"), ("noites", "night"), ("semanas", "week")])
    fn = f"{d.item}_{unit_en}s"
    chunk = T('''
        def $fn(first, last):
            """Quantidade de $unit do periodo, contando o primeiro e o ultimo."""
            $m
    ''', fn=fn, unit=unit_pt, m=MARKER)
    bug, fix = "return last - first", "return 1 + last - first"
    a = r.int(1, 9)
    b = a + r.int(2, 9)
    tests = [Test("single_day", [Eq(call(fn, a, a), 1)]),
             Test("range", [Eq(call(fn, a, b), b - a + 1)])]
    return mk(FAM, "span_days", d, f"{d.item}_periods", chunk, bug, fix, tests, fn,
              behavior=f"contar os {unit_pt} do periodo incluindo o primeiro e o ultimo",
              cause="a subtracao last - first nao conta as duas pontas do periodo",
              fix_desc="somar 1 para incluir as duas pontas", api="aritmetica inteira nativa",
              lesson="Contagem inclusiva de um intervalo e last - first + 1.",
              edge=f"periodo de um unico dia ({a},{a}) e periodo maior ({a},{b})",
              perf="O(1), sem mudanca.")


@shape(FAM, "last_n_entries")
def _off_last_n(d, r, k):
    fn = r.pick([f"last_{d.items}", f"latest_{d.items}", f"recent_{d.items}"])
    entries = r.sample(WORDS, 6)
    if k % 2 == 0:
        bug, fix = "return entries[-n + 1:]", "return entries[-n:]"
        cause = "o corte -n + 1 devolve um item a menos que o pedido"
        tests = [Test("last_two", [Eq(call(fn, entries, 2), entries[-2:])]),
                 Test("last_one", [Eq(call(fn, entries, 1), entries[-1:])]),
                 Test("more_than_available", [Eq(call(fn, entries[:2], 5), entries[:2])])]
    else:
        bug, fix = "return entries[len(entries) - n:]", "return entries[-n:]"
        cause = "len(entries) - n fica negativo quando n passa do tamanho e o corte comeca no lugar errado"
        tests = [Test("last_two", [Eq(call(fn, entries, 2), entries[-2:])]),
                 Test("exactly_all", [Eq(call(fn, entries, 6), entries)]),
                 Test("more_than_available", [Eq(call(fn, entries[:3], 5), entries[:3])])]
    chunk = T('''
        def $fn(entries, n):
            """Devolve os ultimos n $items (todos, se houver menos de n)."""
            $m
    ''', fn=fn, items=d.items, m=MARKER)
    return mk(FAM, "last_n_entries", d, f"{d.items}_recent", chunk, bug, fix, tests, fn,
              behavior=f"devolver os ultimos n {d.items}, ou todos quando houver menos de n",
              cause=cause, fix_desc="cortar com entries[-n:], que aceita n maior que o tamanho",
              api="fatiamento com indice negativo nativo",
              lesson="entries[-n:] ja trata n maior que o tamanho; calcular len - n pode ficar negativo.",
              edge="n menor que o tamanho, n igual ao tamanho ou maior que o tamanho da lista",
              perf="O(n) pela copia da fatia, igual ao anterior.")


@shape(FAM, "numbered_list")
def _off_numbered(d, r, k):
    fn = r.pick([f"numbered_{d.items}", f"{d.items}_menu", f"list_{d.items}"])
    sep = r.pick([". ", ") ", " - ", ": "])
    names = r.sample(NAMES, 3)
    chunk = T('''
        def $fn(names):
            """Numera os $items a partir de 1, no formato "1$sep" seguido do nome."""
            return [f"{i}$sep{name}" for i, name in enumerate(names)]
    ''', fn=fn, items=d.items, sep=sep)
    bug = f'return [f"{{i}}{sep}{{name}}" for i, name in enumerate(names)]'
    fix = f'return [f"{{i}}{sep}{{name}}" for i, name in enumerate(names, 1)]'
    assert chunk.count(bug) == 1
    chunk = chunk.replace(bug, MARKER)
    tests = [Test("starts_at_one", [Eq(call(fn, names), [f"{i}{sep}{n}" for i, n in enumerate(names, 1)])]),
             Test("empty", [Eq(call(fn, []), [])])]
    return mk(FAM, "numbered_list", d, f"{d.items}_menu", chunk, bug, fix, tests, fn,
              behavior=f"numerar os {d.items} comecando em 1",
              cause="enumerate(names) comeca a contar em 0",
              fix_desc="passar start=1 para enumerate", api="enumerate com start, nativo",
              lesson="enumerate comeca em 0; para numeracao humana passe o segundo argumento 1.",
              edge="lista com tres nomes e lista vazia", perf="O(n), sem mudanca.")


@shape(FAM, "loop_skips_last")
def _off_skip_last(d, r, k):
    fn = r.pick([f"sum_{d.num}", f"total_{d.num}_readings", f"add_up_{d.num}"])
    vals = r.ints(5, 3, 30)
    if k % 2 == 0:
        chunk = T('''
            def $fn(readings):
                """Soma todas as leituras de $num."""
                total = 0
                $m
                    total += readings[i]
                return total
        ''', fn=fn, num=d.num, m=MARKER)
        bug, fix = "for i in range(len(readings) - 1):", "for i in range(len(readings)):"
        cause = "range(len(readings) - 1) para antes da ultima leitura"
    else:
        chunk = T('''
            def $fn(readings):
                """Soma todas as leituras de $num."""
                total = 0
                i = 0
                $m
                    total += readings[i]
                    i += 1
                return total
        ''', fn=fn, num=d.num, m=MARKER)
        bug, fix = "while i < len(readings) - 1:", "while i < len(readings):"
        cause = "a condicao i < len(readings) - 1 sai do laco antes da ultima leitura"
    tests = [Test("all_readings", [Eq(call(fn, vals), sum(vals))]),
             Test("single_reading", [Eq(call(fn, vals[:1]), vals[0])]),
             Test("empty", [Eq(call(fn, []), 0)])]
    return mk(FAM, "loop_skips_last", d, f"{d.item}_readings", chunk, bug, fix, tests, fn,
              behavior=f"somar todas as leituras de {d.num}, inclusive a ultima",
              cause=cause, fix_desc="percorrer ate o ultimo indice", api="range/len nativos",
              lesson="O limite de um laco por indice e len(lista), nao len(lista) - 1.",
              edge="lista com varias leituras, uma unica leitura e lista vazia",
              perf="O(n), uma iteracao a mais que antes.")


@shape(FAM, "range_end_inclusive")
def _off_range_end(d, r, k):
    fn = r.pick([f"{d.item}_numbers", f"{d.item}_ids_between", f"number_{d.items}"])
    a = r.int(1, 8)
    b = a + r.int(2, 5)
    if k % 2 == 0:
        chunk = T('''
            def $fn(first, last):
                """Lista os numeros de $item de first ate last, inclusive."""
                $m
        ''', fn=fn, item=d.item, m=MARKER)
        bug, fix = "return list(range(first, last))", "return list(range(first, last + 1))"
        tests = [Test("inclusive_end", [Eq(call(fn, a, b), list(range(a, b + 1)))]),
                 Test("single", [Eq(call(fn, a, a), [a])])]
        cause = "range(first, last) exclui o valor final"
        behavior = f"listar os numeros de {d.item} de first ate last, incluindo last"
    else:
        chunk = T('''
            def $fn(first, last):
                """Soma os numeros de $item de first ate last, inclusive."""
                $m
        ''', fn=fn, item=d.item, m=MARKER)
        bug, fix = "return sum(range(first, last))", "return sum(range(first, last + 1))"
        tests = [Test("inclusive_sum", [Eq(call(fn, a, b), sum(range(a, b + 1)))]),
                 Test("single", [Eq(call(fn, a, a), a)])]
        cause = "range(first, last) exclui o valor final da soma"
        behavior = f"somar os numeros de {d.item} de first ate last, incluindo last"
    return mk(FAM, "range_end_inclusive", d, f"{d.item}_ranges", chunk, bug, fix, tests, fn,
              behavior=behavior, cause=cause, fix_desc="somar 1 ao fim do range",
              api="range nativo", lesson="range(a, b) e aberto em b; para incluir b use b + 1.",
              edge=f"intervalo de varios valores ({a}..{b}) e intervalo de um valor so",
              perf="O(n), um elemento a mais.")


# --------------------------------------------------------------------------------------------
# Familia 2: inverted_comparison
# --------------------------------------------------------------------------------------------
FAM = "inverted_comparison"


@shape(FAM, "direction_flip")
def _cmp_direction(d, r, k):
    above = r.below(2) == 0
    fn = f"{d.items}_above" if above else f"{d.items}_below"
    word = "acima" if above else "abaixo"
    good, bad = (">", "<") if above else ("<", ">")
    limit = r.int(10, 30)
    vals = [limit - 7, limit + 5, limit, limit + 11, limit - 2, limit + 1]
    vals = [vals[i] for i in r.sample(range(6), 6)]
    ref = [v for v in vals if (v > limit if above else v < limit)]
    if k % 2 == 0:
        chunk = T('''
            def $fn(values, limit):
                """Valores de $num estritamente $word do limite."""
                $m
        ''', fn=fn, num=d.num, word=word, m=MARKER)
        bug = f"return [v for v in values if v {bad} limit]"
        fix = f"return [v for v in values if v {good} limit]"
    else:
        chunk = T('''
            def $fn(values, limit):
                """Valores de $num estritamente $word do limite."""
                result = []
                for v in values:
                    $m
                        result.append(v)
                return result
        ''', fn=fn, num=d.num, word=word, m=MARKER)
        bug, fix = f"if v {bad} limit:", f"if v {good} limit:"
    tests = [Test("keeps_matching", [Eq(call(fn, vals, limit), ref)]),
             Test("limit_itself_excluded", [Eq(call(fn, [limit], limit), [])]),
             Test("empty", [Eq(call(fn, [], limit), [])])]
    return mk(FAM, "direction_flip", d, f"{d.item}_filters", chunk, bug, fix, tests, fn,
              behavior=f"devolver so os valores de {d.num} estritamente {word} do limite",
              cause=f"o operador {bad} seleciona exatamente os valores do lado oposto ao pedido",
              fix_desc=f"trocar o operador {bad} por {good}", api="operadores de comparacao nativos",
              lesson="Ao filtrar por limite, conferir o sentido do operador com um valor de cada lado e o proprio limite.",
              edge="valores dos dois lados do limite, o proprio limite (excluido) e lista vazia",
              perf="O(n), sem mudanca.")


@shape(FAM, "inclusive_boundary")
def _cmp_boundary(d, r, k):
    use_min = k % 2 == 0
    if use_min:
        fn = f"meets_minimum_{d.num}"
        good, bad = ">=", ">"
        doc = f"Verdadeiro quando o {d.num} atinge o minimo (o proprio minimo conta)."
        lim = r.int(10, 60)
        tests = [Test("below", [Eq(call(fn, lim - 1, lim), False)]),
                 Test("exactly_at_limit", [Eq(call(fn, lim, lim), True)]),
                 Test("above", [Eq(call(fn, lim + 5, lim), True)])]
        params = "value, minimum"
        expr = "value {op} minimum"
        behavior = f"aceitar o {d.num} que e igual ao minimo"
    else:
        fn = f"fits_limit_{d.num}"
        good, bad = "<=", "<"
        doc = f"Verdadeiro quando o {d.num} cabe no limite (o proprio limite conta)."
        lim = r.int(10, 60)
        tests = [Test("above", [Eq(call(fn, lim + 1, lim), False)]),
                 Test("exactly_at_limit", [Eq(call(fn, lim, lim), True)]),
                 Test("below", [Eq(call(fn, lim - 5, lim), True)])]
        params = "value, maximum"
        expr = "value {op} maximum"
        behavior = f"aceitar o {d.num} que e igual ao limite"
    chunk = T('''
        def $fn($params):
            """$doc"""
            $m
    ''', fn=fn, params=params, doc=doc, m=MARKER)
    bug = "return " + expr.format(op=bad)
    fix = "return " + expr.format(op=good)
    return mk(FAM, "inclusive_boundary", d, f"{d.item}_rules", chunk, bug, fix, tests, fn,
              behavior=behavior, cause=f"{bad} exclui o valor que e exatamente igual ao limite",
              fix_desc=f"usar {good} para que o limite conte", api="operadores de comparacao nativos",
              lesson="Regras inclusivas pedem >= ou <=; o valor exatamente no limite e o caso que denuncia o erro.",
              edge="valor no limite exato, um abaixo e um acima", perf="O(1), sem mudanca.")


@shape(FAM, "extreme_pick")
def _cmp_extreme(d, r, k):
    biggest = k % 2 == 0
    fn = f"top_{d.item}" if biggest else f"cheapest_{d.item}"
    good, bad = ("max", "min") if biggest else ("min", "max")
    word = "maior" if biggest else "menor"
    nums = r.ints(4, 5, 90)
    names = r.sample(WORDS, 4)
    records = [{"name": names[i], d.num: nums[i]} for i in range(4)]
    pick = max(records, key=lambda x: x[d.num]) if biggest else min(records, key=lambda x: x[d.num])
    chunk = T('''
        def $fn(records):
            """Devolve o registro com o $word $num."""
            $m
    ''', fn=fn, word=word, num=d.num, m=MARKER)
    bug = f'return {bad}(records, key=lambda r: r["{d.num}"])'
    fix = f'return {good}(records, key=lambda r: r["{d.num}"])'
    tests = [Test("picks_extreme", [Eq(call(fn, records), pick)]),
             Test("single_record", [Eq(call(fn, records[:1]), records[0])])]
    return mk(FAM, "extreme_pick", d, f"{d.items}_ranking", chunk, bug, fix, tests, fn,
              behavior=f"devolver o registro com o {word} {d.num}",
              cause=f"{bad}() escolhe o extremo oposto ao pedido",
              fix_desc=f"trocar {bad} por {good}", api=f"{good}() com key, nativo",
              lesson="max e min parecem intercambiaveis; confira com um conjunto de quatro valores distintos.",
              edge="quatro registros com valores distintos e lista de um registro so", perf="O(n), sem mudanca.")


@shape(FAM, "sort_direction")
def _cmp_sort(d, r, k):
    desc = k % 2 == 0
    fn = f"ranked_{d.items}" if desc else f"ordered_{d.items}"
    vals = r.sample(range(3, 90), 6)
    if desc:
        bug, fix = "return sorted(values)", "return sorted(values, reverse=True)"
        doc, ref, cause = f"Do maior para o menor {d.num}.", sorted(vals, reverse=True), "sorted() sem reverse devolve a ordem crescente"
        behavior = f"ordenar do maior para o menor {d.num}"
        fix_desc = "passar reverse=True a sorted"
    else:
        bug, fix = "return sorted(values, reverse=True)", "return sorted(values)"
        doc, ref, cause = f"Do menor para o maior {d.num}.", sorted(vals), "reverse=True inverte a ordem pedida (crescente)"
        behavior = f"ordenar do menor para o maior {d.num}"
        fix_desc = "remover reverse=True de sorted"
    chunk = T('''
        def $fn(values):
            """$doc"""
            $m
    ''', fn=fn, doc=doc, m=MARKER)
    tests = [Test("order", [Eq(call(fn, vals), ref)]),
             Test("single", [Eq(call(fn, vals[:1]), vals[:1])])]
    return mk(FAM, "sort_direction", d, f"{d.items}_order", chunk, bug, fix, tests, fn,
              behavior=behavior, cause=cause, fix_desc=fix_desc, api="sorted nativo",
              lesson="Direcao de ordenacao e um argumento explicito: confira-o contra um exemplo com seis valores embaralhados.",
              edge="seis valores embaralhados e lista de um valor", perf="O(n log n), sem mudanca.")


@shape(FAM, "membership_flip")
def _cmp_membership(d, r, k):
    allowed_ok = k % 2 == 0
    fn = f"is_allowed_{d.txt}" if allowed_ok else f"is_blocked_{d.txt}"
    good, bad = ("in", "not in") if allowed_ok else ("in", "not in")
    words = r.sample(WORDS, 4)
    lst_name = "allowed" if allowed_ok else "blocked"
    doc = f"Verdadeiro quando o {d.txt} esta na lista de {'permitidos' if allowed_ok else 'bloqueados'}."
    chunk = T('''
        def $fn(value, $lst):
            """$doc"""
            $m
    ''', fn=fn, lst=lst_name, doc=doc, m=MARKER)
    bug = f"return value {bad} {lst_name}"
    fix = f"return value {good} {lst_name}"
    tests = [Test("listed", [Eq(call(fn, words[0], words[:2]), True)]),
             Test("not_listed", [Eq(call(fn, words[3], words[:2]), False)]),
             Test("empty_list", [Eq(call(fn, words[0], []), False)])]
    return mk(FAM, "membership_flip", d, f"{d.item}_access", chunk, bug, fix, tests, fn,
              behavior=f"dizer se o {d.txt} esta na lista de {'permitidos' if allowed_ok else 'bloqueados'}",
              cause="not in inverte o resultado da pertinencia", fix_desc="trocar not in por in",
              api="operador in nativo", lesson="Um not in solto inverte todo o predicado; teste um valor presente e um ausente.",
              edge="valor presente, valor ausente e lista vazia", perf="O(n) na busca em lista, sem mudanca.")


# --------------------------------------------------------------------------------------------
# Familia 3: mutable_default
# --------------------------------------------------------------------------------------------
FAM = "mutable_default"


@shape(FAM, "list_default")
def _mut_list(d, r, k):
    fn = r.pick([f"add_{d.item}", f"collect_{d.item}", f"push_{d.item}"])
    param = r.pick(["bucket", "acc", "pending", "batch"])
    x, y, z = r.sample(WORDS, 3)
    chunk = T('''
        $m
            $param.append(entry)
            return $param
    ''', param=param, m=MARKER)
    doc = f'    """Acrescenta o {d.item} a lista; sem lista, comeca uma nova."""'
    bug = f"def {fn}(entry, {param}=[]):\n{doc}"
    fix = f"def {fn}(entry, {param}=None):\n{doc}\n    if {param} is None:\n        {param} = []"
    tests = [Test("fresh_list_each_call", [Let("first", call(fn, x)), Let("second", call(fn, y)),
                                           Eq("first", [x]), Eq("second", [y])]),
             Test("explicit_list_is_used", [Eq(call(fn, z, [x]), [x, z])])]
    return mk(FAM, "list_default", d, f"{d.items}_collector", chunk, bug, fix, tests, fn,
              behavior=f"devolver uma lista nova a cada chamada sem {param}, sem acumular entre chamadas",
              cause=f"o valor padrao {param}=[] e criado uma unica vez e compartilhado entre as chamadas",
              fix_desc=f"usar {param}=None e criar a lista dentro da funcao", api="comparacao com None nativa",
              lesson="Valor padrao mutavel e avaliado na definicao; use None e crie o objeto dentro da funcao.",
              edge="duas chamadas seguidas sem lista e uma chamada com lista explicita",
              perf="O(1) a mais por chamada para criar a lista.",
              state=f"O estado compartilhado e o valor padrao {param}=[] da funcao, criado uma vez na definicao.")


@shape(FAM, "dict_default")
def _mut_dict(d, r, k):
    fn = r.pick([f"count_{d.item}_visit", f"tally_{d.item}", f"record_{d.item}"])
    param = r.pick(["seen", "counts", "registry"])
    a, b = r.sample(WORDS, 2)
    chunk = T('''
        $m
            $param[name] = $param.get(name, 0) + 1
            return $param
    ''', param=param, m=MARKER)
    doc = f'    """Conta as ocorrencias de cada {d.item}; sem {param}, comeca do zero."""'
    bug = f"def {fn}(name, {param}={{}}):\n{doc}"
    fix = f"def {fn}(name, {param}=None):\n{doc}\n    if {param} is None:\n        {param} = {{}}"
    tests = [Test("independent_calls", [Let("first", call(fn, a)), Let("second", call(fn, a)),
                                        Eq("first", {a: 1}), Eq("second", {a: 1})]),
             Test("explicit_dict_is_updated", [Eq(call(fn, b, {b: 4}), {b: 5})])]
    return mk(FAM, "dict_default", d, f"{d.items}_tally", chunk, bug, fix, tests, fn,
              behavior=f"contar a partir do zero quando {param} nao for informado",
              cause=f"o dicionario padrao {param}={{}} e unico e guarda as contagens de chamadas anteriores",
              fix_desc=f"usar {param}=None e criar o dicionario na funcao", api="comparacao com None nativa",
              lesson="Dicionario como valor padrao tem o mesmo problema da lista: um unico objeto para todas as chamadas.",
              edge="duas chamadas independentes e uma chamada com dicionario explicito",
              perf="O(1) a mais por chamada.",
              state=f"O estado compartilhado e o dicionario padrao {param}={{}}, criado uma vez na definicao.")


@shape(FAM, "class_attribute")
def _mut_class(d, r, k):
    cls = d.item.capitalize() + r.pick(["Box", "Queue", "Basket", "Tray"])
    attr = r.pick(["entries", "pending", "history", "buffer"])
    x, y = r.sample(WORDS, 2)
    chunk = T('''
        class $cls:
            """Guarda $items adicionados."""

            $m

            def add(self, entry):
                self.$attr.append(entry)
                return len(self.$attr)
    ''', cls=cls, items=d.items, attr=attr, m=MARKER)
    bug = f"{attr} = []"
    fix = f"def __init__(self):\n    self.{attr} = []"
    tests = [Test("instances_do_not_share", [Let("a", f"{cls}()"), Let("b", f"{cls}()"),
                                              Do(f"a.add({x!r})"), Eq(f"b.{attr}", []), Eq(f"a.{attr}", [x])]),
             Test("add_returns_size", [Let("c", f"{cls}()"), Eq(f"c.add({y!r})", 1), Eq(f"c.add({y!r})", 2)])]
    return mk(FAM, "class_attribute", d, f"{d.items}_store", chunk, bug, fix, tests, cls,
              behavior=f"dar a cada instancia a sua propria lista de {d.items}",
              cause=f"{attr} = [] no corpo da classe cria uma lista unica compartilhada por todas as instancias",
              fix_desc="criar a lista em __init__ para cada instancia", api="__init__ e atributo de instancia, nativos",
              lesson="Atributo mutavel no corpo da classe e compartilhado; crie-o em __init__.",
              edge="duas instancias independentes e o tamanho devolvido por add",
              perf="O(1) a mais por instancia criada.",
              state=f"O estado compartilhado e o atributo de classe {attr} = [], visivel por todas as instancias.")


# --------------------------------------------------------------------------------------------
# Familia 4: missing_strip
# --------------------------------------------------------------------------------------------
FAM = "missing_strip"


@shape(FAM, "normalize_text")
def _strip_normalize(d, r, k):
    style = r.pick(["lower", "upper", "slug"])
    fn = f"normalize_{d.txt}" if style != "slug" else f"slug_{d.txt}"
    w1, w2 = r.sample(WORDS, 2)
    raw_forms = [f"  {w1.title()} {w2} ", f"\t{w1.upper()}\n", f"   {w2}"]
    if style == "lower":
        good, doc = "text.strip().lower()", "Remove os espacos das pontas e deixa em minusculas."
        ref = lambda s: s.strip().lower()  # noqa: E731
    elif style == "upper":
        good, doc = "text.strip().upper()", "Remove os espacos das pontas e deixa em maiusculas."
        ref = lambda s: s.strip().upper()  # noqa: E731
    else:
        good, doc = 'text.strip().lower().replace(" ", "-")', "Remove os espacos das pontas e vira slug com hifens."
        ref = lambda s: s.strip().lower().replace(" ", "-")  # noqa: E731
    bad = good.replace("text.strip()", "text") if k % 2 == 0 else good.replace("text.strip()", "text.lstrip()")
    chunk = T('''
        def $fn(text):
            """$doc"""
            $m
    ''', fn=fn, doc=doc, m=MARKER)
    tests = [Test("trims_and_converts", [Eq(call(fn, s), ref(s)) for s in raw_forms]),
             Test("already_clean", [Eq(call(fn, w1), ref(w1))])]
    return mk(FAM, "normalize_text", d, f"{d.txt}_text", chunk, "return " + bad, "return " + good, tests, fn,
              behavior=f"remover os espacos das pontas do {d.txt} antes de normalizar",
              cause="o texto nao passa por strip() dos dois lados antes da conversao" if k % 2 == 0
              else "lstrip() limpa so o inicio e deixa os espacos do fim",
              fix_desc="aplicar strip() antes das demais conversoes", api="str.strip nativo",
              lesson="Normalizacao de entrada comeca por strip(); teste com espaco e quebra de linha dos dois lados.",
              edge="espacos, tabulacao e quebra de linha nas pontas e texto ja limpo", perf="O(n) no tamanho do texto, sem mudanca.")


@shape(FAM, "split_fields")
def _strip_fields(d, r, k):
    sep = r.pick([",", ";", "|"])
    fn = f"parse_{d.txt}_list"
    a, b, c = r.sample(WORDS, 3)
    chunk = T('''
        def $fn(line):
            """Divide a linha por "$sep" e remove os espacos de cada campo."""
            $m
    ''', fn=fn, sep=sep, m=MARKER)
    bug = f'return [part for part in line.split("{sep}")]'
    fix = f'return [part.strip() for part in line.split("{sep}")]'
    line = f"{a} {sep} {b}{sep}   {c}  "
    tests = [Test("strips_each_field", [Eq(call(fn, line), [a, b, c])]),
             Test("single_field", [Eq(call(fn, f"  {a}"), [a])])]
    return mk(FAM, "split_fields", d, f"{d.txt}_parsing", chunk, bug, fix, tests, fn,
              behavior="separar a linha em campos sem espacos nas pontas de cada campo",
              cause="os campos saem de split() sem strip() e mantem os espacos ao redor",
              fix_desc="aplicar strip() a cada campo", api="str.strip e str.split nativos",
              lesson="split() nao limpa os campos: aplique strip() em cada parte.",
              edge=f"separador {sep!r} com espacos antes e depois e campo unico", perf="O(n), sem mudanca.")


@shape(FAM, "blank_check")
def _strip_blank(d, r, k):
    fn = f"is_blank_{d.txt}"
    if k % 2 == 0:
        bug = 'return text == ""'
    else:
        bug = "return len(text) == 0"
    fix = "return not text.strip()"
    w = r.pick(WORDS)
    chunk = T('''
        def $fn(text):
            """Verdadeiro quando o $txt esta vazio ou so tem espacos."""
            $m
    ''', fn=fn, txt=d.txt, m=MARKER)
    tests = [Test("empty", [Eq(call(fn, ""), True)]),
             Test("only_whitespace", [Eq(call(fn, "   "), True), Eq(call(fn, "\t\n"), True)]),
             Test("has_text", [Eq(call(fn, w), False), Eq(call(fn, f" {w} "), False)])]
    return mk(FAM, "blank_check", d, f"{d.txt}_checks", chunk, bug, fix, tests, fn,
              behavior=f"tratar como vazio o {d.txt} que so tem espacos",
              cause="a comparacao so reconhece a string vazia e trata espacos como texto",
              fix_desc="testar not text.strip()", api="str.strip nativo",
              lesson="Vazio de verdade inclui so espacos: teste not text.strip().",
              edge="string vazia, so espacos, tabulacao com quebra de linha e texto com espacos ao redor",
              perf="O(n) pelo strip, sem impacto pratico.")


@shape(FAM, "strip_prefix")
def _strip_prefix(d, r, k):
    prefix = r.pick(["ID-", "SKU-", "ORD-", "REF-", "TKT-"])
    fn = f"{d.txt}_without_prefix"
    num = r.int(10, 99)
    body = prefix[0] + str(num)
    chunk = T('''
        def $fn(code):
            """Remove o prefixo "$prefix" do codigo, sem mexer no resto."""
            $m
    ''', fn=fn, prefix=prefix, m=MARKER)
    bug = f'return code.lstrip("{prefix}")'
    fix = f'return code.removeprefix("{prefix}")'
    tests = [Test("keeps_rest_of_code", [Eq(call(fn, prefix + body), body)]),
             Test("no_prefix", [Eq(call(fn, str(num)), str(num))])]
    return mk(FAM, "strip_prefix", d, f"{d.txt}_codes", chunk, bug, fix, tests, fn,
              behavior=f"tirar so o prefixo {prefix!r}, preservando o restante do codigo",
              cause="lstrip() remove qualquer caractere do conjunto dado, nao o prefixo inteiro",
              fix_desc="usar removeprefix, que remove so o prefixo exato", api="str.removeprefix nativo",
              lesson="lstrip/rstrip recebem um conjunto de caracteres; para prefixo use removeprefix.",
              edge="codigo cujo resto comeca com um caractere do prefixo e codigo sem prefixo",
              perf="O(n), sem mudanca.")


# --------------------------------------------------------------------------------------------
# Familia 5: operation_order
# --------------------------------------------------------------------------------------------
FAM = "operation_order"


@shape(FAM, "expression_precedence")
def _ord_expression(d, r, k):
    kinds = ["midpoint", "line_total", "percent_of", "wrap_next", "mean_of_three", "flat_then_tax"]
    kind = kinds[(r.below(len(kinds)) + k) % len(kinds)]
    if kind == "midpoint":
        fn, params = f"midpoint_{d.num}", "low, high"
        doc, good, bad = f"Ponto medio entre dois valores de {d.num}.", "(low + high) / 2", "low + high / 2"
        a, b = r.int(2, 20) * 2, r.int(21, 40) * 2
        cases = [(a, b), (b, b)]
        ref = lambda x, y: (x + y) / 2  # noqa: E731
        behavior, cause = f"devolver a media aritmetica dos dois valores de {d.num}", "a divisao por 2 so atinge high porque / tem precedencia sobre +"
        edge = f"dois pares de valores, incluindo o par com valores iguais ({b},{b})"
    elif kind == "line_total":
        fn, params = f"line_total_{d.item}", "price, discount, qty"
        doc, good, bad = "Total da linha: (preco - desconto) * quantidade.", "(price - discount) * qty", "price - discount * qty"
        cases = [(r.int(20, 60), r.int(1, 9), r.int(2, 6)), (r.int(20, 60), 0, 3)]
        ref = lambda p, dc, q: (p - dc) * q  # noqa: E731
        behavior, cause = "aplicar o desconto ao preco unitario antes de multiplicar pela quantidade", "o desconto e multiplicado pela quantidade antes de subtrair, porque * tem precedencia sobre -"
        edge = "dois conjuntos de preco, desconto e quantidade, um deles com desconto zero"
    elif kind == "percent_of":
        fn, params = f"{d.item}_percent", "part, whole"
        doc, good, bad = "Percentual de part em whole.", "part / whole * 100", "part / (whole * 100)"
        w = r.pick([40, 50, 80, 200])
        cases = [(w // 4, w), (w // 2, w)]
        ref = lambda p, w_: p / w_ * 100  # noqa: E731
        behavior, cause = "devolver part / whole em porcentagem", "o parentese faz a divisao por whole * 100 em vez de multiplicar o quociente por 100"
        edge = f"duas fracoes de um total de {w}"
    elif kind == "wrap_next":
        fn, params = f"next_{d.item}_slot", "index, count"
        doc, good, bad = "Proximo indice circular: volta a 0 depois do ultimo.", "(index + 1) % count", "index + 1 % count"
        n = r.int(3, 6)
        cases = [(0, n), (n - 1, n)]
        ref = lambda i, c: (i + 1) % c  # noqa: E731
        behavior, cause = "voltar ao indice 0 depois do ultimo indice", "% tem precedencia sobre +, entao so 1 % count e calculado e o indice nunca volta a 0"
        edge = f"o primeiro indice (0) e o ultimo indice ({n - 1}) de {n} posicoes"
    elif kind == "mean_of_three":
        fn, params = f"mean3_{d.num}", "a, b, c"
        doc, good, bad = f"Media de tres valores de {d.num}.", "(a + b + c) / 3", "a + b + c / 3"
        cases = [(r.int(3, 30) * 3, r.int(3, 30) * 3, r.int(3, 30) * 3), (6, 6, 6)]
        ref = lambda a_, b_, c_: (a_ + b_ + c_) / 3  # noqa: E731
        behavior, cause = f"devolver a media dos tres valores de {d.num}", "so c e dividido por 3 porque / tem precedencia sobre +"
        edge = "um trio variado e um trio de valores iguais (6, 6, 6)"
    else:
        fn, params = f"price_with_tax_{d.item}", "price, flat, tax"
        doc, good, bad = "Aplica o desconto fixo primeiro e o imposto (fracao) depois.", "(price - flat) * (1 + tax)", "price * (1 + tax) - flat"
        cases = [(100, 10, 0.5), (r.int(40, 90), 5, 0.25)]
        ref = lambda p, f, t: (p - f) * (1 + t)  # noqa: E731
        behavior, cause = "descontar o valor fixo do preco antes de aplicar o imposto", "o imposto e aplicado antes do desconto fixo, o que muda o resultado"
        edge = "dois pares de preco e desconto fixo com imposto fracionario"
    chunk = T('''
        def $fn($params):
            """$doc"""
            $m
    ''', fn=fn, params=params, doc=doc, m=MARKER)
    tests = [Test("expected_values", [Eq(call(fn, *c), ref(*c)) for c in cases])]
    return mk(FAM, "expression_precedence", d, f"{d.item}_math", chunk, "return " + bad, "return " + good, tests, fn,
              behavior=behavior, cause=cause, fix_desc="colocar parenteses para impor a ordem das operacoes",
              api="parenteses e aritmetica nativa",
              lesson="Precedencia: * e / ganham de + e -; use parenteses sempre que a ordem importar.",
              edge=edge, perf="O(1), sem mudanca.")


@shape(FAM, "validate_after_normalize")
def _ord_validate(d, r, k):
    fn = f"clean_{d.txt}"
    min_len = r.int(3, 5)
    w = (r.pick(WORDS) * 2)[:min_len + 2]
    chunk = T('''
        def $fn(text, min_len):
            """Remove os espacos das pontas e rejeita (None) o que ficar curto demais."""
            $m
            return text
    ''', fn=fn, m=MARKER)
    bug = "if len(text) < min_len:\n    return None\ntext = text.strip()"
    fix = "text = text.strip()\nif len(text) < min_len:\n    return None"
    short = "  " + w[:min_len - 1] + "  "
    tests = [Test("short_after_strip", [Eq(call(fn, short, min_len), None)]),
             Test("long_enough", [Eq(call(fn, f"  {w} ", min_len), w)])]
    return mk(FAM, "validate_after_normalize", d, f"{d.txt}_cleaning", chunk, bug, fix, tests, fn,
              behavior=f"medir o tamanho do {d.txt} depois de remover os espacos das pontas",
              cause="o tamanho e medido antes do strip(), entao os espacos das pontas contam como caracteres",
              fix_desc="mover o strip() para antes da checagem de tamanho", api="str.strip nativo",
              lesson="Normalize antes de validar: a validacao deve ver o mesmo valor que sera guardado.",
              edge=f"texto curto cercado de espacos (minimo {min_len}) e texto longo o bastante", perf="O(n), sem mudanca.")


@shape(FAM, "accumulate_before_append")
def _ord_accumulate(d, r, k):
    fn = f"running_{d.num}"
    vals = r.sample(range(2, 30), 4)
    run = []
    t = 0
    for v in vals:
        t += v
        run.append(t)
    chunk = T('''
        def $fn(values):
            """Total acumulado de $num depois de cada valor."""
            out = []
            total = 0
            for v in values:
                $m
            return out
    ''', fn=fn, num=d.num, m=MARKER)
    bug = "out.append(total)\ntotal += v"
    fix = "total += v\nout.append(total)"
    tests = [Test("cumulative", [Eq(call(fn, vals), run)]),
             Test("empty", [Eq(call(fn, []), [])])]
    return mk(FAM, "accumulate_before_append", d, f"{d.item}_cumulative", chunk, bug, fix, tests, fn,
              behavior=f"incluir cada valor de {d.num} no total acumulado que e registrado para ele",
              cause="o total e registrado antes de somar o valor atual, entao a lista fica uma posicao atrasada",
              fix_desc="somar o valor antes de registrar o total", api="list.append nativo",
              lesson="Em acumuladores, a ordem entre atualizar e registrar define se o valor atual entra.",
              edge="quatro valores e lista vazia", perf="O(n), sem mudanca.")


@shape(FAM, "cap_after_scale")
def _ord_cap(d, r, k):
    fn = f"scaled_{d.num}"
    cap = r.int(50, 100)
    factor = r.int(2, 4)
    chunk = T('''
        def $fn(value, factor, cap):
            """Multiplica pelo fator e limita o resultado ao teto."""
            $m
    ''', fn=fn, m=MARKER)
    bug = "return min(value, cap) * factor"
    fix = "return min(value * factor, cap)"
    tests = [Test("capped_result", [Eq(call(fn, cap - 5, factor, cap), cap)]),
             Test("under_cap", [Eq(call(fn, 3, 2, cap), 6)])]
    return mk(FAM, "cap_after_scale", d, f"{d.item}_limits", chunk, bug, fix, tests, fn,
              behavior=f"limitar o resultado ja multiplicado ao teto de {d.num}",
              cause="o teto e aplicado ao valor antes do fator, entao o resultado pode passar do teto",
              fix_desc="multiplicar primeiro e limitar o produto", api="min nativo",
              lesson="Limite o resultado final, nao a entrada, quando o teto vale para o resultado.",
              edge="resultado acima do teto e resultado abaixo do teto", perf="O(1), sem mudanca.")


# --------------------------------------------------------------------------------------------
# Familia 6: wrong_type
# --------------------------------------------------------------------------------------------
FAM = "wrong_type"


@shape(FAM, "sum_numeric_text")
def _type_sum_text(d, r, k):
    fn = r.pick([f"total_{d.num}_text", f"sum_{d.num}_fields", f"add_{d.num}_strings"])
    nums = r.ints(4, 2, 40)
    fields = [str(n) for n in nums]
    chunk = T('''
        def $fn(fields):
            """Soma o $num recebido como texto, ex.: ["3", "4"]."""
            $m
    ''', fn=fn, num=d.num, m=MARKER)
    bug, fix = "return sum(fields)", "return sum(int(f) for f in fields)"
    tests = [Test("sums_numbers", [Eq(call(fn, fields), sum(nums))]),
             Test("empty", [Eq(call(fn, []), 0)])]
    return mk(FAM, "sum_numeric_text", d, f"{d.item}_input", chunk, bug, fix, tests, fn,
              behavior=f"somar o {d.num} recebido como texto e devolver um inteiro",
              cause="sum() recebe strings e falha com TypeError ao somar texto com inteiro",
              fix_desc="converter cada campo com int() antes de somar", api="int e sum nativos",
              lesson="Dados lidos de texto precisam de conversao explicita antes de qualquer aritmetica.",
              edge="quatro campos numericos e lista vazia", perf="O(n), uma conversao por campo.")


@shape(FAM, "integer_division")
def _type_division(d, r, k):
    if k % 2 == 0:
        fn = f"average_{d.num}"
        vals = r.ints(4, 2, 40)
        if sum(vals) % len(vals) == 0:
            vals[0] += 1
        chunk = T('''
            def $fn(values):
                """Media de $num como numero decimal (sem arredondar para baixo)."""
                $m
        ''', fn=fn, num=d.num, m=MARKER)
        bug, fix = "return sum(values) // len(values)", "return sum(values) / len(values)"
        tests = [Test("fractional_average", [Eq(call(fn, vals), sum(vals) / len(vals))]),
                 Test("exact_average", [Eq(call(fn, [4, 6]), 5.0)])]
        behavior = f"devolver a media de {d.num} com parte decimal"
        cause = "// faz divisao inteira e descarta a parte decimal da media"
        fix_desc = "trocar // por / para manter a parte decimal"
        edge = "media com parte decimal e media exata"
    else:
        fn = f"boxes_for_{d.items}"
        size = r.int(3, 8)
        chunk = T('''
            def $fn(count, size):
                """Quantas caixas de tamanho size sao precisas para guardar count $items."""
                $m
        ''', fn=fn, items=d.items, m=MARKER)
        bug, fix = "return count / size", "return (count + size - 1) // size"
        tests = [Test("rounds_up", [Eq(call(fn, size * 2 + 1, size), 3)]),
                 Test("exact_fit", [Eq(call(fn, size * 2, size), 2)]),
                 Test("none", [Eq(call(fn, 0, size), 0)])]
        behavior = "devolver o numero inteiro de caixas, arredondando para cima"
        cause = "/ devolve um decimal em vez de um inteiro arredondado para cima"
        fix_desc = "usar divisao inteira com arredondamento para cima"
        edge = f"quantidade que sobra ({size * 2 + 1}), quantidade exata e quantidade zero"
    return mk(FAM, "integer_division", d, f"{d.items}_stats", chunk, bug, fix, tests, fn,
              behavior=behavior, cause=cause, fix_desc=fix_desc, api="operadores / e // nativos",
              lesson="/ devolve decimal e // devolve inteiro piso: escolha pelo tipo que o contrato promete.",
              edge=edge, perf="O(n) na soma ou O(1), sem mudanca.")


@shape(FAM, "concat_number")
def _type_concat(d, r, k):
    fn = r.pick([f"label_{d.item}", f"{d.item}_caption", f"describe_{d.items}"])
    word = r.pick(WORDS)
    if k % 2 == 0:
        chunk = T('''
            def $fn(name, count):
                """Texto "nome: contagem" para um $item."""
                $m
        ''', fn=fn, item=d.item, m=MARKER)
        bug, fix = 'return name + ": " + count', 'return name + ": " + str(count)'
        tests = [Test("int_count", [Eq(call(fn, word, 12), f"{word}: 12")]),
                 Test("zero_count", [Eq(call(fn, word, 0), f"{word}: 0")])]
        behavior = "montar o texto com um numero inteiro como contagem"
    else:
        chunk = T('''
            def $fn(total):
                """Texto "Total de $items: N" para um total inteiro."""
                $m
        ''', fn=fn, items=d.items, m=MARKER)
        bug = f'return "Total de {d.items}: " + total'
        fix = f'return "Total de {d.items}: " + str(total)'
        tests = [Test("int_total", [Eq(call(fn, 7), f"Total de {d.items}: 7")]),
                 Test("zero_total", [Eq(call(fn, 0), f"Total de {d.items}: 0")])]
        behavior = "montar o texto com um total inteiro"
    return mk(FAM, "concat_number", d, f"{d.item}_labels", chunk, bug, fix, tests, fn,
              behavior=behavior, cause="concatenar str com int levanta TypeError",
              fix_desc="converter o numero com str() antes de concatenar", api="str nativo",
              lesson="Concatenacao so aceita texto; converta numeros com str() ou use f-string.",
              edge="contagem positiva e contagem zero", perf="O(n) no tamanho do texto, sem mudanca.")


@shape(FAM, "max_of_numeric_text")
def _type_max_text(d, r, k):
    use_max = k % 2 == 0
    fn = f"highest_{d.num}" if use_max else f"lowest_{d.num}"
    good_fn = "max" if use_max else "min"
    nums = [9, 10, 2, 100][: 4]
    nums = [nums[i] for i in r.sample(range(4), 4)]
    texts = [str(n) for n in nums]
    chunk = T('''
        def $fn(values):
            """Devolve o $word $num entre valores numericos recebidos como texto."""
            $m
    ''', fn=fn, word="maior" if use_max else "menor", num=d.num, m=MARKER)
    bug = f"return {good_fn}(values)"
    fix = f"return {good_fn}(values, key=int)"
    ref = str(max(nums) if use_max else min(nums))
    tests = [Test("numeric_order", [Eq(call(fn, texts), ref)]),
             Test("single", [Eq(call(fn, ["7"]), "7")])]
    return mk(FAM, "max_of_numeric_text", d, f"{d.item}_values", chunk, bug, fix, tests, fn,
              behavior="comparar os valores como numeros, nao como texto",
              cause="max() e min() comparam strings letra a letra, entao '10' fica antes de '2' e '9' vence '10'",
              fix_desc="passar key=int para comparar numericamente", api=f"{good_fn} com key=int, nativo",
              lesson="Strings numericas ordenam por texto; compare com key=int (ou converta antes).",
              edge="valores de um e de varios digitos (9, 10, 2, 100) e lista de um valor",
              perf="O(n) com uma conversao por comparacao.")


@shape(FAM, "parse_decimal")
def _type_parse_decimal(d, r, k):
    fn = f"parse_{d.num}"
    whole, frac = r.int(1, 40), r.pick([25, 5, 75, 5])
    text = f"{whole}.{frac}"
    chunk = T('''
        def $fn(text):
            """Converte o texto de $num (pode ter casas decimais) em numero."""
            $m
    ''', fn=fn, num=d.num, m=MARKER)
    bug, fix = "return int(text)", "return float(text)"
    tests = [Test("decimal_text", [Eq(call(fn, text), float(text))]),
             Test("integer_text", [Eq(call(fn, str(whole)), float(whole))])]
    return mk(FAM, "parse_decimal", d, f"{d.item}_parse", chunk, bug, fix, tests, fn,
              behavior=f"aceitar texto de {d.num} com casas decimais",
              cause="int() rejeita texto com ponto decimal e levanta ValueError",
              fix_desc="converter com float()", api="float nativo",
              lesson="Se a entrada pode ter casas decimais, converta com float (ou Decimal), nao com int.",
              edge="texto com casas decimais e texto inteiro", perf="O(n) no tamanho do texto, sem mudanca.")


@shape(FAM, "bool_from_text")
def _type_bool_text(d, r, k):
    fn = f"is_{d.item}_enabled"
    chunk = T('''
        def $fn(text):
            """Interpreta o texto "true"/"false" (qualquer caixa, com espacos) de um $item."""
            $m
    ''', fn=fn, item=d.item, m=MARKER)
    bug = "return bool(text)"
    fix = 'return text.strip().lower() == "true"'
    tests = [Test("true_forms", [Eq(call(fn, "true"), True), Eq(call(fn, " True "), True)]),
             Test("false_forms", [Eq(call(fn, "false"), False), Eq(call(fn, ""), False)])]
    return mk(FAM, "bool_from_text", d, f"{d.item}_flags", chunk, bug, fix, tests, fn,
              behavior="devolver False para o texto 'false', e True so para 'true'",
              cause="bool() de qualquer string nao vazia e True, inclusive 'false'",
              fix_desc="comparar o texto normalizado com 'true'", api="str.strip, str.lower nativos",
              lesson="bool(texto) so testa se esta vazio; interprete o conteudo explicitamente.",
              edge="'true', ' True ' com espacos, 'false' e texto vazio", perf="O(n), sem mudanca.")


# --------------------------------------------------------------------------------------------
# Familia 7: dict_key
# --------------------------------------------------------------------------------------------
FAM = "dict_key"


@shape(FAM, "wrong_key_name")
def _key_name(d, r, k):
    good_key, bad_key = r.pick([("qty", "quantity"), ("count", "amount"), ("units", "unit"), ("n", "num")])
    fn = r.pick([f"{d.item}_cost", f"{d.item}_line_total", f"price_of_{d.item}"])
    p, q = r.int(3, 20), r.int(2, 9)
    chunk = T('''
        def $fn(record):
            """Preco vezes quantidade; o registro tem as chaves "$num" e "$gk"."""
            $m
    ''', fn=fn, num=d.num, gk=good_key, m=MARKER)
    bug = f'return record["{d.num}"] * record["{bad_key}"]'
    fix = f'return record["{d.num}"] * record["{good_key}"]'
    rec = {d.num: p, good_key: q}
    tests = [Test("multiplies_fields", [Eq(call(fn, rec), p * q)]),
             Test("quantity_one", [Eq(call(fn, {d.num: p, good_key: 1}), p)])]
    return mk(FAM, "wrong_key_name", d, f"{d.item}_records", chunk, bug, fix, tests, fn,
              behavior=f"multiplicar o {d.num} pela quantidade guardada na chave '{good_key}'",
              cause=f"a chave '{bad_key}' nao existe no registro, que usa '{good_key}', e a leitura levanta KeyError",
              fix_desc=f"ler a chave '{good_key}'", api="indexacao de dict nativa",
              lesson="Confira o nome exato das chaves no contrato do registro; um KeyError aponta a chave errada.",
              edge="registro com quantidade variada e com quantidade 1", perf="O(1), sem mudanca.")


@shape(FAM, "missing_key_default")
def _key_default(d, r, k):
    default = r.pick(["n/d", "desconhecido", "sem dados", "-"])
    fn = f"{d.txt}_of"
    w = r.pick(WORDS)
    chunk = T('''
        def $fn(record):
            """Devolve o campo opcional "$txt", ou "$default" quando ele nao existe."""
            $m
    ''', fn=fn, txt=d.txt, default=default, m=MARKER)
    bug = f'return record["{d.txt}"]'
    fix = f'return record.get("{d.txt}", "{default}")'
    tests = [Test("present", [Eq(call(fn, {d.txt: w}), w)]),
             Test("absent", [Eq(call(fn, {"id": 1}), default)]),
             Test("empty_record", [Eq(call(fn, {}), default)])]
    return mk(FAM, "missing_key_default", d, f"{d.item}_lookup", chunk, bug, fix, tests, fn,
              behavior=f"devolver '{default}' quando a chave '{d.txt}' nao existir",
              cause="a indexacao direta levanta KeyError quando a chave esta ausente",
              fix_desc="usar dict.get com o valor padrao", api="dict.get nativo",
              lesson="Para chaves opcionais use dict.get(chave, padrao) em vez de indexar diretamente.",
              edge="chave presente, chave ausente e dicionario vazio", perf="O(1), sem mudanca.")


@shape(FAM, "count_occurrences")
def _key_count(d, r, k):
    fn = f"count_{d.txt}_values"
    words = r.sample(WORDS, 3)
    seq = [words[0], words[1], words[0], words[2], words[0]]
    chunk = T('''
        def $fn(values):
            """Conta quantas vezes cada $txt aparece."""
            counts = {}
            for key in values:
                $m
            return counts
    ''', fn=fn, txt=d.txt, m=MARKER)
    if k % 2 == 0:
        bug = "counts[key] = counts[key] + 1"
        cause = "counts[key] levanta KeyError na primeira ocorrencia, quando a chave ainda nao existe"
    else:
        bug = "counts[key] = 1"
        cause = "a atribuicao counts[key] = 1 reinicia a contagem em cada ocorrencia"
    fix = "counts[key] = counts.get(key, 0) + 1"
    ref = {words[0]: 3, words[1]: 1, words[2]: 1}
    tests = [Test("counts_repeated", [Eq(call(fn, seq), ref)]),
             Test("empty", [Eq(call(fn, []), {})])]
    return mk(FAM, "count_occurrences", d, f"{d.txt}_counts", chunk, bug, fix, tests, fn,
              behavior=f"contar todas as ocorrencias de cada {d.txt}",
              cause=cause, fix_desc="somar 1 ao valor atual usando dict.get com padrao 0", api="dict.get nativo",
              lesson="Contadores precisam de um valor inicial: counts.get(chave, 0) + 1.",
              edge="chave que repete tres vezes, chaves unicas e lista vazia", perf="O(n), sem mudanca.")


@shape(FAM, "group_rows")
def _key_group(d, r, k):
    fn = f"group_{d.items}_by_{d.txt}"
    a, b = r.sample(WORDS, 2)
    rows = [{d.txt: a, "id": 1}, {d.txt: b, "id": 2}, {d.txt: a, "id": 3}]
    chunk = T('''
        def $fn(rows):
            """Agrupa as linhas pelo campo "$txt"; cada grupo guarda todas as suas linhas."""
            groups = {}
            for row in rows:
                $m
            return groups
    ''', fn=fn, txt=d.txt, m=MARKER)
    bug = f'groups[row["{d.txt}"]] = [row]'
    fix = f'groups.setdefault(row["{d.txt}"], []).append(row)'
    ref = {a: [rows[0], rows[2]], b: [rows[1]]}
    tests = [Test("keeps_all_rows", [Eq(call(fn, rows), ref)]),
             Test("empty", [Eq(call(fn, []), {})])]
    return mk(FAM, "group_rows", d, f"{d.items}_groups", chunk, bug, fix, tests, fn,
              behavior="manter todas as linhas de cada grupo, sem perder as anteriores",
              cause="a atribuicao substitui a lista do grupo e so a ultima linha de cada chave sobrevive",
              fix_desc="acrescentar a linha com setdefault(...).append", api="dict.setdefault nativo",
              lesson="Agrupar exige setdefault(chave, []).append(item); atribuir sobrescreve o grupo.",
              edge="chave repetida em duas linhas, chave unica e lista vazia", perf="O(n), sem mudanca.")


@shape(FAM, "iterate_items")
def _key_items(d, r, k):
    fn = f"{d.items}_with_{d.num}"
    names = r.sample(WORDS, 4)
    stock = {names[0]: r.int(2, 9), names[1]: 0, names[2]: r.int(1, 5), names[3]: 0}
    chunk = T('''
        def $fn(stock):
            """Nomes cujo $num e maior que zero (stock mapeia nome para $num)."""
            $m
    ''', fn=fn, num=d.num, m=MARKER)
    bug = "return [name for name in stock if name > 0]"
    fix = "return [name for name, qty in stock.items() if qty > 0]"
    ref = [n for n, v in stock.items() if v > 0]
    tests = [Test("positive_only", [Eq(call(fn, stock), ref)]),
             Test("empty", [Eq(call(fn, {}), [])])]
    return mk(FAM, "iterate_items", d, f"{d.items}_stock", chunk, bug, fix, tests, fn,
              behavior=f"devolver os nomes com {d.num} maior que zero",
              cause="iterar o dicionario devolve so as chaves, e a comparacao usa o nome em vez do valor",
              fix_desc="iterar com .items() e comparar o valor", api="dict.items nativo",
              lesson="for k in d percorre chaves; use d.items() quando o valor entra na condicao.",
              edge="dois itens positivos, dois zerados e dicionario vazio", perf="O(n), sem mudanca.")


@shape(FAM, "merge_precedence")
def _key_merge(d, r, k):
    fn = f"merge_{d.item}_options"
    a, b = r.sample(WORDS, 2)
    chunk = T('''
        def $fn(defaults, overrides):
            """Combina as opcoes; o que o usuario informa em overrides vence os defaults."""
            $m
    ''', fn=fn, m=MARKER)
    bug = "return {**overrides, **defaults}"
    fix = "return {**defaults, **overrides}"
    tests = [Test("override_wins", [Eq(call(fn, {"mode": a, "limit": 5}, {"mode": b}), {"mode": b, "limit": 5})]),
             Test("no_overrides", [Eq(call(fn, {"mode": a}, {}), {"mode": a})])]
    return mk(FAM, "merge_precedence", d, f"{d.item}_options", chunk, bug, fix, tests, fn,
              behavior="fazer o valor informado pelo usuario vencer o padrao",
              cause="no desempacotamento, a ultima chave ganha, e os defaults vem por ultimo",
              fix_desc="inverter a ordem para que overrides venha por ultimo", api="desempacotamento de dict nativo",
              lesson="Em {**a, **b} o ultimo vence: coloque por ultimo o que deve ter prioridade.",
              edge="chave sobrescrita com uma chave extra preservada e sem sobrescrita", perf="O(n), sem mudanca.")


# --------------------------------------------------------------------------------------------
# Familia 8: return_flow
# --------------------------------------------------------------------------------------------
FAM = "return_flow"


@shape(FAM, "return_inside_loop")
def _ret_loop(d, r, k):
    fn = r.pick([f"total_{d.num}", f"add_all_{d.num}", f"sum_of_{d.items}"])
    vals = r.ints(4, 2, 30)
    chunk = T('''
        def $fn(values):
            """Soma todos os valores de $num."""
            total = 0
            $m
    ''', fn=fn, num=d.num, m=MARKER)
    bug = "for v in values:\n    total += v\n    return total"
    fix = "for v in values:\n    total += v\nreturn total"
    tests = [Test("sums_all", [Eq(call(fn, vals), sum(vals))]),
             Test("empty", [Eq(call(fn, []), 0)])]
    return mk(FAM, "return_inside_loop", d, f"{d.item}_sums", chunk, bug, fix, tests, fn,
              behavior=f"somar todos os valores de {d.num} e so entao devolver o total",
              cause="o return esta dentro do for e encerra a funcao na primeira iteracao",
              fix_desc="tirar o return do laco (um nivel de indentacao a menos)", api="nenhuma API nova, so indentacao",
              lesson="A indentacao do return decide se ele roda a cada iteracao ou uma vez ao fim.",
              edge="varios valores e lista vazia (a funcao precisa devolver 0, nao None)",
              perf="O(n) em vez de O(1) errado; a correcao percorre a lista toda.")


@shape(FAM, "missing_return")
def _ret_missing(d, r, k):
    fn = r.pick([f"clean_{d.items}", f"trim_{d.txt}s", f"tidy_{d.items}"])
    w1, w2 = r.sample(WORDS, 2)
    chunk = T('''
        def $fn(values):
            """Devolve os $txt sem espacos nas pontas."""
            result = [v.strip() for v in values]
            $m
    ''', fn=fn, txt=d.txt, m=MARKER)
    bug = "result = [v.strip() for v in values]"
    fix = "result = [v.strip() for v in values]\nreturn result"
    chunk = chunk.replace(f"    result = [v.strip() for v in values]\n    {MARKER}", f"    {MARKER}")
    tests = [Test("returns_cleaned_list", [Eq(call(fn, [f" {w1} ", f"{w2}  "]), [w1, w2])]),
             Test("empty", [Eq(call(fn, []), [])])]
    return mk(FAM, "missing_return", d, f"{d.txt}_cleanup", chunk, bug, fix, tests, fn,
              behavior=f"devolver a lista de {d.txt} sem espacos nas pontas",
              cause="a funcao monta o resultado mas termina sem return e devolve None",
              fix_desc="devolver result no fim da funcao", api="nenhuma API nova, so return",
              lesson="Toda funcao que produz um valor precisa de return explicito; sem ele o resultado e None.",
              edge="lista com espacos e lista vazia (deve devolver [], nao None)", perf="O(n), sem mudanca.")


@shape(FAM, "sort_returns_none")
def _ret_sort_none(d, r, k):
    fn = f"sorted_{d.items}" if k % 2 == 0 else f"ordered_{d.num}_list"
    vals = r.sample(range(4, 90), 5)
    chunk = T('''
        def $fn(values):
            """Devolve os valores de $num em ordem crescente."""
            $m
    ''', fn=fn, num=d.num, m=MARKER)
    bug, fix = "return values.sort()", "return sorted(values)"
    tests = [Test("returns_sorted_list", [Eq(call(fn, vals), sorted(vals))]),
             Test("empty", [Eq(call(fn, []), [])])]
    return mk(FAM, "sort_returns_none", d, f"{d.item}_sorting", chunk, bug, fix, tests, fn,
              behavior=f"devolver a lista de {d.num} ordenada",
              cause="list.sort() ordena no lugar e devolve None",
              fix_desc="usar sorted(), que devolve a lista ordenada", api="sorted nativo",
              lesson="list.sort() devolve None; para obter a lista ordenada use sorted().",
              edge="cinco valores embaralhados e lista vazia", perf="O(n log n); sorted() copia a lista.")


# --------------------------------------------------------------------------------------------
# Familia 9: copy_alias
# --------------------------------------------------------------------------------------------
FAM = "copy_alias"


@shape(FAM, "sort_in_place")
def _copy_sort(d, r, k):
    fn = f"ordered_{d.items}"
    vals = r.sample(range(5, 95), 5)
    if vals == sorted(vals):
        vals = vals[::-1]
    chunk = T('''
        def $fn(values):
            """Devolve uma lista ordenada de $num sem alterar a lista recebida."""
            $m
    ''', fn=fn, num=d.num, m=MARKER)
    bug = "values.sort()\nreturn values"
    fix = "return sorted(values)"
    tests = [Test("returns_sorted", [Eq(call(fn, vals), sorted(vals))]),
             Test("input_untouched", [Let("data", repr(vals)), Let("out", f"{fn}(data)"),
                                      Eq("out", sorted(vals)), Eq("data", vals)])]
    return mk(FAM, "sort_in_place", d, f"{d.items}_copy", chunk, bug, fix, tests, fn,
              behavior="ordenar sem modificar a lista que o chamador passou",
              cause="values.sort() reordena a lista do chamador no lugar e devolve a mesma lista",
              fix_desc="devolver sorted(values), que cria uma lista nova", api="sorted nativo",
              lesson="Funcoes que devolvem uma versao ordenada nao devem mutar o argumento: prefira sorted().",
              edge="a lista de entrada continua na ordem original depois da chamada", perf="O(n log n) mais uma copia O(n).",
              state="A funcao recebe a lista do chamador e a modifica no lugar, o que afeta o estado de quem chamou.")


@shape(FAM, "alias_assignment")
def _copy_alias(d, r, k):
    fn = f"with_extra_{d.item}"
    base = r.sample(WORDS, 3)
    extra = r.pick(NAMES)
    chunk = T('''
        def $fn(base, extra):
            """Devolve uma lista nova com o $item extra no fim, sem alterar base."""
            $m
            result.append(extra)
            return result
    ''', fn=fn, item=d.item, m=MARKER)
    if k % 2 == 0:
        bug, fix = "result = base", "result = list(base)"
    else:
        bug, fix = "result = base", "result = base[:]"
    tests = [Test("appends_extra", [Eq(call(fn, base, extra), base + [extra])]),
             Test("base_untouched", [Let("original", repr(list(base))), Do(f"{fn}(original, {extra!r})"),
                                     Eq("original", base)])]
    return mk(FAM, "alias_assignment", d, f"{d.items}_append", chunk, bug, fix, tests, fn,
              behavior="nao alterar a lista base ao montar a nova lista",
              cause="result = base cria um apelido para a mesma lista e append altera a lista do chamador",
              fix_desc="copiar a lista antes de acrescentar", api="list() / fatiamento nativos",
              lesson="Atribuir uma lista nao a copia; copie com list(x) ou x[:] antes de mutar.",
              edge="a lista original continua igual depois da chamada", perf="O(n) pela copia.",
              state="A funcao recebe a lista base do chamador e a modifica via apelido.")


@shape(FAM, "shallow_copy_nested")
def _copy_nested(d, r, k):
    fn = f"copy_{d.item}_config"
    a, b = r.sample(WORDS, 2)
    chunk = T('''
        def $fn(config):
            """Copia a configuracao de modo que alterar a copia nao altere o original."""
            $m
    ''', fn=fn, m=MARKER)
    bug = "return dict(config)"
    fix = "return {key: list(value) for key, value in config.items()}"
    tests = [Test("nested_lists_are_independent",
                  [Let("cfg", repr({a: [1, 2], b: [3]})), Let("clone", f"{fn}(cfg)"),
                   Do(f"clone[{a!r}].append(9)"), Eq("cfg", {a: [1, 2], b: [3]}),
                   Eq("clone", {a: [1, 2, 9], b: [3]})]),
             Test("same_content", [Eq(call(fn, {a: [1]}), {a: [1]})])]
    return mk(FAM, "shallow_copy_nested", d, f"{d.item}_config", chunk, bug, fix, tests, fn,
              behavior="entregar uma copia cujas listas internas sejam independentes das originais",
              cause="dict(config) e uma copia rasa: as listas internas continuam compartilhadas",
              fix_desc="copiar tambem cada lista interna", api="compreensao de dict e list() nativos",
              lesson="Copia rasa duplica so o nivel de cima; objetos aninhados mutaveis precisam de copia propria.",
              edge="alterar a lista da copia nao muda o original", perf="O(n) no tamanho total, em vez de O(k) nas chaves.",
              state="Os valores internos (listas) sao compartilhados entre o original e a copia rasa.")


@shape(FAM, "shared_rows")
def _copy_grid(d, r, k):
    fn = f"blank_{d.item}_grid"
    rows, cols = r.int(2, 3), r.int(2, 4)
    chunk = T('''
        def $fn(rows, cols):
            """Cria uma grade rows x cols de zeros; as linhas sao independentes."""
            $m
    ''', fn=fn, m=MARKER)
    bug = "return [[0] * cols] * rows"
    fix = "return [[0] * cols for _ in range(rows)]"
    base = [[0] * cols for _ in range(rows)]
    edited = [row[:] for row in base]
    edited[0][0] = 5
    tests = [Test("blank_shape", [Eq(call(fn, rows, cols), base)]),
             Test("rows_are_independent", [Let("grid", call(fn, rows, cols)), Do("grid[0][0] = 5"),
                                           Eq("grid", edited)])]
    return mk(FAM, "shared_rows", d, f"{d.item}_grid", chunk, bug, fix, tests, fn,
              behavior="criar linhas independentes: alterar uma celula nao pode afetar as outras linhas",
              cause="[[0] * cols] * rows repete a mesma lista interna em todas as linhas",
              fix_desc="criar uma lista nova por linha com compreensao", api="compreensao de lista e range nativos",
              lesson="Multiplicar uma lista de listas repete a referencia; use compreensao para criar cada linha.",
              edge=f"grade {rows}x{cols} e alteracao de uma celula da primeira linha", perf="O(rows * cols), sem mudanca.",
              state="As linhas da grade sao a mesma lista repetida, entao uma escrita aparece em todas.")


# --------------------------------------------------------------------------------------------
# Familia 10: initial_value
# --------------------------------------------------------------------------------------------
FAM = "initial_value"


@shape(FAM, "assign_instead_of_add")
def _init_assign(d, r, k):
    fn = f"grand_total_{d.num}"
    groups = [r.ints(2, 1, 20) for _ in range(3)]
    chunk = T('''
        def $fn(groups):
            """Soma o $num de todos os grupos."""
            total = 0
            for group in groups:
                subtotal = sum(group)
                $m
            return total
    ''', fn=fn, num=d.num, m=MARKER)
    bug, fix = "total = subtotal", "total += subtotal"
    tests = [Test("all_groups", [Eq(call(fn, groups), sum(sum(g) for g in groups))]),
             Test("one_group", [Eq(call(fn, groups[:1]), sum(groups[0]))]),
             Test("no_groups", [Eq(call(fn, []), 0)])]
    return mk(FAM, "assign_instead_of_add", d, f"{d.item}_groups", chunk, bug, fix, tests, fn,
              behavior=f"somar o {d.num} de todos os grupos",
              cause="total = subtotal substitui o acumulado e so o ultimo grupo conta",
              fix_desc="acumular com += em vez de atribuir", api="operador += nativo",
              lesson="Dentro de um laco acumulador, = sobrescreve e += acumula.",
              edge="tres grupos, um grupo so e nenhum grupo", perf="O(n), sem mudanca.")


@shape(FAM, "extreme_start")
def _init_extreme(d, r, k):
    use_max = k % 2 == 0
    fn = f"best_{d.num}" if use_max else f"lowest_{d.num}"
    if use_max:
        vals = [-r.int(2, 9) for _ in range(4)]
        vals = [v - i for i, v in enumerate(vals)]
        var, op, bug_init, ref = "best", ">", "best = 0", max(vals)
        doc = f"Maior {d.num} da lista; os valores podem ser negativos."
        behavior = f"devolver o maior {d.num} mesmo quando todos sao negativos"
        cause = "best comeca em 0 e nenhum valor negativo consegue superar esse inicio"
        edge = "lista so com valores negativos e lista de um valor"
    else:
        vals = [r.int(5, 80) for _ in range(4)]
        var, op, bug_init, ref = "lowest", "<", "lowest = 0", min(vals)
        doc = f"Menor {d.num} da lista; os valores sao positivos."
        behavior = f"devolver o menor {d.num} da lista de valores positivos"
        cause = "lowest comeca em 0 e nenhum valor positivo consegue ser menor que esse inicio"
        edge = "lista so com valores positivos e lista de um valor"
    chunk = T('''
        def $fn(values):
            """$doc"""
            $m
            for v in values:
                if v $op $var:
                    $var = v
            return $var
    ''', fn=fn, doc=doc, op=op, var=var, m=MARKER)
    fix = f"{var} = values[0]"
    tests = [Test("extreme_value", [Eq(call(fn, vals), ref)]),
             Test("single_value", [Eq(call(fn, vals[:1]), vals[0])])]
    return mk(FAM, "extreme_start", d, f"{d.item}_extremes", chunk, bug_init, fix, tests, fn,
              behavior=behavior, cause=cause, fix_desc="iniciar com o primeiro elemento da lista",
              api="indexacao de lista nativa",
              lesson="Inicie maximos e minimos com um elemento real da lista, nao com 0.",
              edge=edge, perf="O(n), sem mudanca.")


@shape(FAM, "product_start")
def _init_product(d, r, k):
    fn = f"product_{d.num}"
    vals = [r.int(2, 5) for _ in range(3)]
    chunk = T('''
        def $fn(factors):
            """Produto dos fatores de $num."""
            $m
            for f in factors:
                result *= f
            return result
    ''', fn=fn, num=d.num, m=MARKER)
    bug, fix = "result = 0", "result = 1"
    p = 1
    for v in vals:
        p *= v
    tests = [Test("product", [Eq(call(fn, vals), p)]),
             Test("single_factor", [Eq(call(fn, vals[:1]), vals[0])]),
             Test("no_factors", [Eq(call(fn, []), 1)])]
    return mk(FAM, "product_start", d, f"{d.item}_factors", chunk, bug, fix, tests, fn,
              behavior=f"multiplicar todos os fatores de {d.num}, devolvendo 1 para a lista vazia",
              cause="o acumulador comeca em 0 e qualquer produto fica zero",
              fix_desc="iniciar o acumulador em 1, o elemento neutro da multiplicacao", api="aritmetica inteira nativa",
              lesson="O valor inicial de um acumulador e o elemento neutro da operacao: 0 para soma, 1 para produto.",
              edge="tres fatores, um fator e nenhum fator", perf="O(n), sem mudanca.")


# --------------------------------------------------------------------------------------------
# Familia 11: boolean_logic
# --------------------------------------------------------------------------------------------
FAM = "boolean_logic"


@shape(FAM, "or_with_literal")
def _bool_or_literal(d, r, k):
    s1, s2, other = r.pick([("open", "pending", "closed"), ("new", "review", "done"),
                            ("draft", "queued", "sent"), ("active", "trial", "expired")])
    fn = f"is_{d.item}_live"
    chunk = T('''
        def $fn(status):
            """Verdadeiro para $items nos estados "$s1" e "$s2"."""
            $m
    ''', fn=fn, items=d.items, s1=s1, s2=s2, m=MARKER)
    bug = f'return status == "{s1}" or "{s2}"'
    fix = f'return status in ("{s1}", "{s2}")'
    tests = [Test("first_state", [Eq(call(fn, s1), True)]),
             Test("second_state", [Eq(call(fn, s2), True)]),
             Test("other_state", [Eq(call(fn, other), False)])]
    return mk(FAM, "or_with_literal", d, f"{d.item}_status", chunk, bug, fix, tests, fn,
              behavior=f"aceitar so os estados '{s1}' e '{s2}'",
              cause=f"status == '{s1}' or '{s2}' avalia a string '{s2}' como verdadeira para qualquer estado",
              fix_desc="testar a pertinencia com in (...)", api="operador in nativo",
              lesson="x == 'a' or 'b' e sempre verdadeiro; compare cada lado ou use x in ('a', 'b').",
              edge=f"os dois estados validos e um estado invalido ({other})", perf="O(1), sem mudanca.")


@shape(FAM, "and_or_swap")
def _bool_and_or(d, r, k):
    fn = f"in_range_{d.num}"
    lo = r.int(5, 20)
    hi = lo + r.int(10, 30)
    chunk = T('''
        def $fn(value, low, high):
            """Verdadeiro quando o $num esta entre low e high, inclusive."""
            $m
    ''', fn=fn, num=d.num, m=MARKER)
    bug, fix = "return low <= value or value <= high", "return low <= value and value <= high"
    tests = [Test("inside", [Eq(call(fn, lo + 1, lo, hi), True)]),
             Test("below", [Eq(call(fn, lo - 1, lo, hi), False)]),
             Test("above", [Eq(call(fn, hi + 1, lo, hi), False)])]
    return mk(FAM, "and_or_swap", d, f"{d.item}_ranges", chunk, bug, fix, tests, fn,
              behavior=f"aceitar so o {d.num} dentro do intervalo, rejeitando valores fora dele",
              cause="com or, basta uma das duas desigualdades valer e todo valor passa",
              fix_desc="trocar or por and", api="operadores logicos nativos",
              lesson="Intervalo exige as duas condicoes ao mesmo tempo: and, nao or.",
              edge="valor dentro, um abaixo e um acima do intervalo", perf="O(1), sem mudanca.")


@shape(FAM, "de_morgan")
def _bool_demorgan(d, r, k):
    fn = f"is_free_{d.item}"
    chunk = T('''
        def $fn(taken, blocked):
            """Verdadeiro quando o $item nao esta ocupado e tambem nao esta bloqueado."""
            $m
    ''', fn=fn, item=d.item, m=MARKER)
    bug, fix = "return not taken or not blocked", "return not taken and not blocked"
    tests = [Test("free", [Eq(call(fn, False, False), True)]),
             Test("taken", [Eq(call(fn, True, False), False)]),
             Test("blocked", [Eq(call(fn, False, True), False)]),
             Test("both", [Eq(call(fn, True, True), False)])]
    return mk(FAM, "de_morgan", d, f"{d.item}_slots", chunk, bug, fix, tests, fn,
              behavior="exigir que o item nao esteja ocupado e nao esteja bloqueado",
              cause="not a or not b e verdadeiro se qualquer um dos dois estiver livre; era preciso exigir os dois",
              fix_desc="trocar or por and (De Morgan)", api="operadores logicos nativos",
              lesson="not (a or b) equivale a not a and not b; trocar o conectivo muda o significado.",
              edge="as quatro combinacoes de ocupado e bloqueado", perf="O(1), sem mudanca.")


# --------------------------------------------------------------------------------------------
# Familia 12: loop_control
# --------------------------------------------------------------------------------------------
FAM = "loop_control"


@shape(FAM, "any_instead_of_all")
def _loop_any_all(d, r, k):
    fn = f"all_{d.items}_within"
    limit = r.int(20, 60)
    chunk = T('''
        def $fn(values, limit):
            """Verdadeiro quando todos os valores de $num estao dentro do limite."""
            $m
    ''', fn=fn, num=d.num, m=MARKER)
    bug = "return any(v <= limit for v in values)"
    fix = "return all(v <= limit for v in values)"
    tests = [Test("all_within", [Eq(call(fn, [limit - 5, limit - 9, limit], limit), True)]),
             Test("one_above", [Eq(call(fn, [limit - 5, limit + 1], limit), False)]),
             Test("empty", [Eq(call(fn, [], limit), True)])]
    return mk(FAM, "any_instead_of_all", d, f"{d.item}_limits", chunk, bug, fix, tests, fn,
              behavior=f"exigir que todos os valores de {d.num} estejam no limite",
              cause="any() basta um valor dentro do limite, mas o contrato exige todos",
              fix_desc="trocar any por all", api="all nativo",
              lesson="any = pelo menos um; all = todos. Teste um caso com um unico valor fora e a lista vazia.",
              edge="todos dentro (inclusive no limite), um acima e lista vazia", perf="O(n), sem mudanca.")


@shape(FAM, "break_instead_of_continue")
def _loop_break(d, r, k):
    fn = f"valid_{d.num}_values"
    vals = [r.int(2, 30), -r.int(1, 9), r.int(2, 30), r.int(2, 30)]
    chunk = T('''
        def $fn(values):
            """Mantem so os valores de $num nao negativos, descartando os negativos."""
            out = []
            for v in values:
                if v < 0:
                    $m
                out.append(v)
            return out
    ''', fn=fn, num=d.num, m=MARKER)
    bug, fix = "break", "continue"
    tests = [Test("skips_negative", [Eq(call(fn, vals), [v for v in vals if v >= 0])]),
             Test("all_valid", [Eq(call(fn, [1, 2, 3]), [1, 2, 3])])]
    return mk(FAM, "break_instead_of_continue", d, f"{d.item}_valid", chunk, bug, fix, tests, fn,
              behavior="descartar so os valores negativos e continuar com os demais",
              cause="break encerra o laco no primeiro negativo e os valores seguintes se perdem",
              fix_desc="trocar break por continue", api="continue nativo",
              lesson="break sai do laco; continue pula so a iteracao atual.",
              edge="um negativo no meio da lista e lista sem negativos", perf="O(n), sem mudanca.")


@shape(FAM, "early_true")
def _loop_early_true(d, r, k):
    fn = f"every_{d.item}_ok"
    limit = r.int(30, 60)
    chunk = T('''
        def $fn(values, limit):
            """Verdadeiro quando todos os valores de $num cabem no limite."""
            $m
    ''', fn=fn, num=d.num, m=MARKER)
    bug = "for v in values:\n    if v <= limit:\n        return True\nreturn False"
    fix = "for v in values:\n    if v > limit:\n        return False\nreturn True"
    tests = [Test("all_fit", [Eq(call(fn, [limit - 3, limit - 8], limit), True)]),
             Test("last_does_not_fit", [Eq(call(fn, [limit - 3, limit + 4], limit), False)]),
             Test("empty", [Eq(call(fn, [], limit), True)])]
    return mk(FAM, "early_true", d, f"{d.item}_checks", chunk, bug, fix, tests, fn,
              behavior=f"devolver True so quando todos os valores de {d.num} cabem no limite",
              cause="o laco devolve True no primeiro valor que cabe, sem olhar os demais",
              fix_desc="devolver False no primeiro valor que nao cabe e True so ao fim do laco",
              api="for e return nativos",
              lesson="Para 'todos', retorne cedo no primeiro contra-exemplo e confirme so depois do laco.",
              edge="todos cabem, so o ultimo nao cabe e lista vazia", perf="O(n) com saida antecipada, sem mudanca.")


# --------------------------------------------------------------------------------------------
# Familia 13: falsy_zero
# --------------------------------------------------------------------------------------------
FAM = "falsy_zero"


@shape(FAM, "or_default")
def _zero_or_default(d, r, k):
    fn = f"effective_{d.num}"
    default = r.int(5, 50)
    chunk = T('''
        def $fn(value, default):
            """Usa o padrao so quando $num e None; zero e um valor valido."""
            $m
    ''', fn=fn, num=d.num, m=MARKER)
    bug, fix = "return value or default", "return default if value is None else value"
    v = r.int(2, 40)
    tests = [Test("none_uses_default", [Eq(call(fn, None, default), default)]),
             Test("zero_is_kept", [Eq(call(fn, 0, default), 0)]),
             Test("value_is_kept", [Eq(call(fn, v, default), v)])]
    return mk(FAM, "or_default", d, f"{d.item}_defaults", chunk, bug, fix, tests, fn,
              behavior=f"manter o {d.num} igual a zero e usar o padrao so para None",
              cause="value or default trata 0 como ausente e troca zero pelo padrao",
              fix_desc="comparar explicitamente com None", api="operador is nativo",
              lesson="'x or padrao' troca 0, '' e [] pelo padrao; use 'is None' quando esses valores sao validos.",
              edge="None, zero e um valor positivo", perf="O(1), sem mudanca.")


@shape(FAM, "not_value_check")
def _zero_not_value(d, r, k):
    fn = f"describe_{d.items}_count"
    chunk = T('''
        def $fn(count):
            """Descreve a contagem de $items; None significa sem dados e 0 e um valor valido."""
            $m
                return "sem dados"
            return f"{count} $items"
    ''', fn=fn, items=d.items, m=MARKER)
    bug, fix = "if not count:", "if count is None:"
    n = r.int(2, 30)
    tests = [Test("none_means_no_data", [Eq(call(fn, None), "sem dados")]),
             Test("zero_is_a_count", [Eq(call(fn, 0), f"0 {d.items}")]),
             Test("positive_count", [Eq(call(fn, n), f"{n} {d.items}")])]
    return mk(FAM, "not_value_check", d, f"{d.items}_report", chunk, bug, fix, tests, fn,
              behavior=f"mostrar '0 {d.items}' para zero e 'sem dados' so para None",
              cause="if not count trata 0 como ausente e responde 'sem dados'",
              fix_desc="testar count is None", api="operador is nativo",
              lesson="'if not x' mistura None com 0; quando zero e valido, compare com None.",
              edge="None, zero e uma contagem positiva", perf="O(1), sem mudanca.")


# --------------------------------------------------------------------------------------------
# Familia 14: exception_handling
# --------------------------------------------------------------------------------------------
FAM = "exception_handling"


@shape(FAM, "missing_raise")
def _exc_missing_raise(d, r, k):
    fn = f"check_{d.num}"
    limit = r.int(20, 90)
    exc = r.pick(["ValueError", "OverflowError", "ArithmeticError"])
    chunk = T('''
        def $fn(value, limit):
            """Levanta $exc quando o $num passa do limite; senao devolve o valor."""
            if value > limit:
                $m
            return value
    ''', fn=fn, num=d.num, exc=exc, m=MARKER)
    bug = f'{exc}(f"{d.num} {{value}} acima do limite {{limit}}")'
    fix = f'raise {exc}(f"{d.num} {{value}} acima do limite {{limit}}")'
    tests = [Test("raises_above_limit", [Raises(call(fn, limit + 1, limit), exc)]),
             Test("limit_is_allowed", [Eq(call(fn, limit, limit), limit)]),
             Test("below_limit", [Eq(call(fn, 3, limit), 3)])]
    return mk(FAM, "missing_raise", d, f"{d.item}_validation", chunk, bug, fix, tests, fn,
              behavior=f"levantar {exc} quando o {d.num} passa do limite",
              cause=f"{exc}(...) so cria a excecao e a descarta; sem raise nada e levantado",
              fix_desc="acrescentar raise antes da excecao", api=f"{exc} nativa e a palavra raise",
              lesson="Criar a excecao nao a levanta: falta o raise.",
              edge="valor acima do limite, exatamente no limite e abaixo do limite", perf="O(1), sem mudanca.")


@shape(FAM, "wrong_exception")
def _exc_wrong(d, r, k):
    kinds = [
        ("parse", "ValueError", ("KeyError", "TypeError", "IndexError"), "return int(text)", "text", "13", "abc", "o texto nao numerico"),
        ("lookup", "KeyError", ("ValueError", "IndexError", "TypeError"), "return table[key]", "table, key", None, None, "a chave ausente"),
        ("index", "IndexError", ("KeyError", "ValueError", "TypeError"), "return items[position]", "items, position", None, None, "a posicao fora da lista"),
        ("divide", "ZeroDivisionError", ("ValueError", "KeyError", "TypeError"), "return total / count", "total, count", None, None, "a contagem zero"),
    ]
    kind, good, bads, expr, params, _a, _b, what = kinds[(r.below(len(kinds)) + k) % len(kinds)]
    bad = bads[r.below(len(bads))]
    fn = f"safe_{kind}_{d.num}"
    default = r.int(1, 9) * 100
    if kind == "parse":
        ok_call, ok_val = call(fn, "42", default), 42
        bad_call = call(fn, "abc", default)
    elif kind == "lookup":
        ok_call, ok_val = call(fn, {"a": 5}, "a", default), 5
        bad_call = call(fn, {"a": 5}, "z", default)
    elif kind == "index":
        ok_call, ok_val = call(fn, [7, 8], 1, default), 8
        bad_call = call(fn, [7, 8], 5, default)
    else:
        ok_call, ok_val = call(fn, 10, 4, default), 2.5
        bad_call = call(fn, 10, 0, default)
    sig = params + ", fallback"
    chunk = T('''
        def $fn($sig):
            """Devolve o resultado, ou fallback quando ha $what."""
            try:
                $expr
            $m
                return fallback
    ''', fn=fn, sig=sig, what=what, expr=expr, m=MARKER)
    bug, fix = f"except {bad}:", f"except {good}:"
    tests = [Test("normal_result", [Eq(ok_call, ok_val)]),
             Test("falls_back", [Eq(bad_call, default)])]
    return mk(FAM, "wrong_exception", d, f"{d.item}_safe", chunk, bug, fix, tests, fn,
              behavior=f"devolver o fallback quando houver {what}, sem deixar a excecao escapar",
              cause=f"o except trata {bad}, mas a operacao levanta {good}, que escapa sem tratamento",
              fix_desc=f"capturar {good}", api=f"{good} nativa",
              lesson="O except precisa nomear a excecao que a operacao realmente levanta; descubra-a com um caso de falha.",
              edge=f"resultado normal e o caso de falha ({what})", perf="O(1), sem mudanca.")


# --------------------------------------------------------------------------------------------
# Familia 15: rounding_format
# --------------------------------------------------------------------------------------------
FAM = "rounding_format"


@shape(FAM, "round_half_up")
def _fmt_half_up(d, r, k):
    fn = f"rounded_{d.num}"
    chunk = T('''
        def $fn(value):
            """Arredonda o $num para o inteiro mais proximo; metades arredondam para cima."""
            $m
    ''', fn=fn, num=d.num, m=MARKER)
    bug, fix = "return round(value)", "return int(value + 0.5)"
    tests = [Test("halves_round_up", [Eq(call(fn, 0.5), 1), Eq(call(fn, 2.5), 3)]),
             Test("regular_values", [Eq(call(fn, 3.2), 3), Eq(call(fn, 4.7), 5)])]
    return mk(FAM, "round_half_up", d, f"{d.item}_rounding", chunk, bug, fix, tests, fn,
              behavior="arredondar metades para cima (0.5 vira 1, 2.5 vira 3)",
              cause="round() usa arredondamento bancario e leva 0.5 e 2.5 para o inteiro par (0 e 2)",
              fix_desc="somar 0.5 e truncar com int", api="int nativo",
              lesson="round() em Python arredonda metades para o par mais proximo; para meia-para-cima use int(x + 0.5) em positivos.",
              edge="metades (0.5 e 2.5) e valores comuns (3.2 e 4.7)", perf="O(1), sem mudanca.")


@shape(FAM, "decimal_places")
def _fmt_places(d, r, k):
    places = r.pick([2, 3])
    fn = f"format_{d.num}"
    prefix = r.pick(["R$ ", "USD ", "", "~"])
    chunk = T('''
        def $fn(value):
            """Texto do $num com $places casas decimais$pfx_doc."""
            $m
    ''', fn=fn, num=d.num, places=places, pfx_doc=(f', precedido de "{prefix}"' if prefix else ""), m=MARKER)
    wrong = places - 1
    bug = f'return f"{prefix}{{value:.{wrong}f}}"'
    fix = f'return f"{prefix}{{value:.{places}f}}"'
    v1, v2 = r.pick([3.14159, 2.71828, 1.41421]), r.pick([10.0, 7.5, 100.25])
    tests = [Test("pads_decimals", [Eq(call(fn, v2), f"{prefix}{v2:.{places}f}")]),
             Test("rounds_decimals", [Eq(call(fn, v1), f"{prefix}{v1:.{places}f}")])]
    return mk(FAM, "decimal_places", d, f"{d.item}_format", chunk, bug, fix, tests, fn,
              behavior=f"mostrar o {d.num} com exatamente {places} casas decimais",
              cause=f"o formato .{wrong}f mostra {wrong} casa(s) em vez de {places}",
              fix_desc=f"usar o formato .{places}f", api="f-string com especificador de formato nativa",
              lesson="O numero depois do ponto no formato e a quantidade de casas: confira-o com um valor de 1 casa e outro de 5.",
              edge=f"valor redondo ({v2}) e valor com muitas casas ({v1})", perf="O(1), sem mudanca.")


@shape(FAM, "percent_text")
def _fmt_percent(d, r, k):
    fn = f"{d.item}_rate_text"
    chunk = T('''
        def $fn(ratio):
            """Texto da taxa de $item como porcentagem com uma casa, ex.: 0.256 vira "25.6%"."""
            $m
    ''', fn=fn, item=d.item, m=MARKER)
    bug = 'return f"{ratio:.1f}%"'
    fix = 'return f"{ratio * 100:.1f}%"'
    tests = [Test("fraction_to_percent", [Eq(call(fn, 0.256), "25.6%"), Eq(call(fn, 0.5), "50.0%")]),
             Test("zero", [Eq(call(fn, 0.0), "0.0%")])]
    return mk(FAM, "percent_text", d, f"{d.item}_rates", chunk, bug, fix, tests, fn,
              behavior="converter a fracao em porcentagem antes de formatar",
              cause="a fracao e formatada sem multiplicar por 100 e 0.256 aparece como 0.3%",
              fix_desc="multiplicar a fracao por 100 antes de formatar", api="f-string nativa",
              lesson="Formatar com sufixo % nao converte a escala: multiplique por 100 antes.",
              edge="0.256, 0.5 e zero", perf="O(1), sem mudanca.")


@shape(FAM, "zero_pad")
def _fmt_pad(d, r, k):
    fn = f"{d.item}_code"
    width = r.int(3, 6)
    n = r.int(2, 98)
    chunk = T('''
        def $fn(number, width):
            """Codigo do $item com zeros a esquerda ate a largura width."""
            $m
    ''', fn=fn, item=d.item, m=MARKER)
    bug, fix = 'return str(number).ljust(width, "0")', "return str(number).zfill(width)"
    tests = [Test("left_padding", [Eq(call(fn, n, width), str(n).zfill(width))]),
             Test("already_wide", [Eq(call(fn, 123456, 3), "123456")])]
    return mk(FAM, "zero_pad", d, f"{d.item}_codes", chunk, bug, fix, tests, fn,
              behavior="completar com zeros a esquerda, sem alterar o valor numerico do codigo",
              cause="ljust preenche a direita, e 7 com largura 3 vira '700' em vez de '007'",
              fix_desc="usar zfill, que preenche a esquerda", api="str.zfill nativo",
              lesson="ljust/rjust escolhem o lado do preenchimento; zfill completa com zeros a esquerda.",
              edge=f"numero curto (largura {width}) e numero que ja passa da largura", perf="O(width), sem mudanca.")


# --------------------------------------------------------------------------------------------
# Familia 16: unit_conversion
# --------------------------------------------------------------------------------------------
FAM = "unit_conversion"


@shape(FAM, "scale_factor")
def _unit_scale(d, r, k):
    src, dst, factor, up = r.pick([
        ("kb", "mb", 1024, False), ("seconds", "minutes", 60, False), ("cm", "m", 100, False),
        ("m", "km", 1000, False), ("hours", "minutes", 60, True), ("kg", "g", 1000, True),
        ("minutes", "seconds", 60, True), ("km", "m", 1000, True)])
    fn = f"{d.item}_{src}_to_{dst}"
    chunk = T('''
        def $fn($src):
            """Converte $src em $dst para um $item."""
            $m
    ''', fn=fn, src=src, dst=dst, item=d.item, m=MARKER)
    if up:
        good, bad = f"{src} * {factor}", f"{src} / {factor}"
        ref = lambda x: x * factor  # noqa: E731
    else:
        good, bad = f"{src} / {factor}", f"{src} * {factor}"
        ref = lambda x: x / factor  # noqa: E731
    xs = [factor * 2, factor * 5]
    tests = [Test("converts", [Eq(call(fn, x), ref(x)) for x in xs]),
             Test("zero", [Eq(call(fn, 0), 0)])]
    return mk(FAM, "scale_factor", d, f"{d.item}_units", chunk, "return " + bad, "return " + good, tests, fn,
              behavior=f"converter {src} em {dst}",
              cause=f"a conversao usa o operador oposto: de {src} para {dst} e preciso {'multiplicar' if up else 'dividir'} por {factor}",
              fix_desc=f"{'multiplicar' if up else 'dividir'} por {factor}", api="aritmetica nativa",
              lesson="Antes de converter unidades, diga em voz alta se o numero deve crescer ou diminuir.",
              edge=f"dois multiplos de {factor} e o valor zero", perf="O(1), sem mudanca.")


@shape(FAM, "clock_format")
def _unit_clock(d, r, k):
    fn = f"{d.item}_clock"
    chunk = T('''
        def $fn(seconds):
            """Formata uma duracao de $item em segundos como HH:MM."""
            hours = seconds // 3600
            $m
            return f"{hours:02d}:{minutes:02d}"
    ''', fn=fn, item=d.item, m=MARKER)
    bug, fix = "minutes = seconds // 60", "minutes = (seconds % 3600) // 60"
    h, m = r.int(1, 9), r.int(5, 55)
    secs = h * 3600 + m * 60 + r.int(0, 59)
    tests = [Test("hours_and_minutes", [Eq(call(fn, secs), f"{h:02d}:{m:02d}")]),
             Test("under_one_hour", [Eq(call(fn, 125), "00:02")])]
    return mk(FAM, "clock_format", d, f"{d.item}_clock", chunk, bug, fix, tests, fn,
              behavior="mostrar so os minutos que sobram depois das horas completas",
              cause="seconds // 60 conta todos os minutos desde o inicio, sem descontar as horas ja mostradas",
              fix_desc="tirar o resto em horas com % 3600 antes de dividir por 60", api="operadores // e % nativos",
              lesson="Ao decompor uma duracao, cada unidade usa o resto da maior: (segundos % 3600) // 60.",
              edge=f"duracao de {h} h e {m} min com segundos soltos e duracao abaixo de uma hora", perf="O(1), sem mudanca.")


# --------------------------------------------------------------------------------------------
# Familia 17: string_methods
# --------------------------------------------------------------------------------------------
FAM = "string_methods"


@shape(FAM, "replace_swapped")
def _str_replace(d, r, k):
    old, new, old_pt = r.pick([("-", " ", "hifen"), ("_", " ", "sublinhado"), (".", " ", "ponto")])
    fn = f"readable_{d.txt}"
    a, b = r.sample(WORDS, 2)
    chunk = T('''
        def $fn(text):
            """Troca cada $old_pt do $txt por um espaco."""
            $m
    ''', fn=fn, old_pt=old_pt, txt=d.txt, m=MARKER)
    bug = f'return text.replace("{new}", "{old}")'
    fix = f'return text.replace("{old}", "{new}")'
    tests = [Test("replaces_marker", [Eq(call(fn, f"{a}{old}{b}"), f"{a} {b}")]),
             Test("nothing_to_replace", [Eq(call(fn, a), a)])]
    return mk(FAM, "replace_swapped", d, f"{d.txt}_display", chunk, bug, fix, tests, fn,
              behavior=f"trocar cada {old_pt} por espaco",
              cause="os dois argumentos de replace estao invertidos: ele troca o espaco pelo caractere",
              fix_desc="inverter os argumentos de replace", api="str.replace nativo",
              lesson="str.replace(antigo, novo): o primeiro argumento e o que sai.",
              edge="texto com o caractere e texto sem ele", perf="O(n), sem mudanca.")


@shape(FAM, "startswith_endswith")
def _str_ends(d, r, k):
    ext = r.pick([".csv", ".json", ".txt", ".log", ".xml"])
    fn = f"is_{ext[1:]}_{d.item}_file"
    a = r.pick(WORDS)
    chunk = T('''
        def $fn(filename):
            """Verdadeiro quando o arquivo do $item termina com $ext."""
            $m
    ''', fn=fn, item=d.item, ext=ext, m=MARKER)
    bug = f'return filename.startswith("{ext}")'
    fix = f'return filename.endswith("{ext}")'
    tests = [Test("matching_suffix", [Eq(call(fn, a + ext), True)]),
             Test("other_suffix", [Eq(call(fn, a + ".bin"), False)])]
    return mk(FAM, "startswith_endswith", d, f"{d.item}_files", chunk, bug, fix, tests, fn,
              behavior=f"reconhecer arquivos que terminam com {ext}",
              cause=f"startswith('{ext}') olha o inicio do nome, e a extensao fica no fim",
              fix_desc="trocar startswith por endswith", api="str.endswith nativo",
              lesson="Extensao e sufixo: endswith, nao startswith.",
              edge="nome com a extensao certa e nome com outra extensao", perf="O(1), sem mudanca.")


@shape(FAM, "split_once")
def _str_split_once(d, r, k):
    sep = r.pick(["=", ":"])
    fn = f"parse_{d.txt}_pair"
    a, b, c = r.sample(WORDS, 3)
    chunk = T('''
        def $fn(line):
            """Separa "chave${sep}valor" no primeiro "$sep"; o valor pode conter "$sep"."""
            $m
            return key.strip(), value.strip()
    ''', fn=fn, sep=sep, m=MARKER)
    bug = f'key, value = line.split("{sep}")'
    fix = f'key, value = line.split("{sep}", 1)'
    tests = [Test("value_with_separator", [Eq(call(fn, f"{a}{sep}{b}{sep}{c}"), (a, f"{b}{sep}{c}"))]),
             Test("simple_pair", [Eq(call(fn, f"{a} {sep} {b}"), (a, b))])]
    return mk(FAM, "split_once", d, f"{d.txt}_pairs", chunk, bug, fix, tests, fn,
              behavior=f"separar so no primeiro '{sep}', mantendo o resto no valor",
              cause=f"split('{sep}') quebra em todos os separadores e o desempacotamento falha com ValueError",
              fix_desc="limitar a um corte com maxsplit=1", api="str.split com maxsplit, nativo",
              lesson="split(sep) corta em toda ocorrencia; use split(sep, 1) para chave/valor.",
              edge="valor que contem o separador e par simples com espacos", perf="O(n), sem mudanca.")


@shape(FAM, "join_separator")
def _str_join(d, r, k):
    fn = f"{d.items}_summary"
    words = r.sample(WORDS, 3)
    sep = r.pick([", ", " | ", "; "])
    chunk = T('''
        def $fn(names):
            """Junta os nomes de $items separados por "$sep"."""
            $m
    ''', fn=fn, items=d.items, sep=sep, m=MARKER)
    bug = f'return "{sep.strip()}".join(names)'
    fix = f'return "{sep}".join(names)'
    tests = [Test("joined_with_separator", [Eq(call(fn, words), sep.join(words))]),
             Test("single_name", [Eq(call(fn, words[:1]), words[0])])]
    return mk(FAM, "join_separator", d, f"{d.items}_summary", chunk, bug, fix, tests, fn,
              behavior=f"separar os nomes com {sep!r}, incluindo os espacos",
              cause=f"o separador {sep.strip()!r} perdeu os espacos de {sep!r}",
              fix_desc="usar o separador completo", api="str.join nativo",
              lesson="O separador do join e literal, espacos incluidos.",
              edge="tres nomes e um nome so", perf="O(n), sem mudanca.")


# --------------------------------------------------------------------------------------------
# Familia 18: recursion_base
# --------------------------------------------------------------------------------------------
FAM = "recursion_base"


@shape(FAM, "sum_to_n")
def _rec_sum(d, r, k):
    fn = f"sum_up_to_{d.item}"
    chunk = T('''
        def $fn(n):
            """Soma de 1 ate n, de forma recursiva."""
            $m
            return n + $fn(n - 1)
    ''', fn=fn, m=MARKER)
    bug = "if n == 1:\n    return 0"
    fix = "if n <= 0:\n    return 0"
    tests = [Test("small_values", [Eq(call(fn, 1), 1), Eq(call(fn, 3), 6)]),
             Test("bigger_value", [Eq(call(fn, 6), 21)]),
             Test("zero", [Eq(call(fn, 0), 0)])]
    return mk(FAM, "sum_to_n", d, f"{d.item}_recursion", chunk, bug, fix, tests, fn,
              behavior="somar 1 ate n incluindo o proprio n",
              cause="o caso base n == 1 devolve 0 e perde o valor 1; n == 0 nunca chega ao caso base",
              fix_desc="usar n <= 0 como caso base", api="recursao nativa",
              lesson="O caso base da soma e o valor neutro (0) no menor n valido; teste 0 e 1.",
              edge="n = 0, 1, 3 e 6", perf="O(n) chamadas, sem mudanca.")


@shape(FAM, "factorial_base")
def _rec_factorial(d, r, k):
    fn = f"factorial_of_{d.items}"
    chunk = T('''
        def $fn(n):
            """Fatorial de n, recursivo (fatorial de 0 e 1)."""
            if n == 0:
                $m
            return n * $fn(n - 1)
    ''', fn=fn, m=MARKER)
    bug, fix = "return 0", "return 1"
    tests = [Test("zero", [Eq(call(fn, 0), 1)]),
             Test("small_values", [Eq(call(fn, 1), 1), Eq(call(fn, 4), 24), Eq(call(fn, 5), 120)])]
    return mk(FAM, "factorial_base", d, f"{d.item}_factorial", chunk, bug, fix, tests, fn,
              behavior="devolver 1 para n = 0 e os fatoriais corretos acima disso",
              cause="o caso base devolve 0 e anula todo o produto",
              fix_desc="devolver 1, o elemento neutro da multiplicacao, no caso base", api="recursao nativa",
              lesson="O caso base de um produto e 1, nao 0.",
              edge="n = 0, 1, 4 e 5", perf="O(n) chamadas, sem mudanca.")


@shape(FAM, "countdown_list")
def _rec_countdown(d, r, k):
    fn = f"countdown_{d.items}"
    chunk = T('''
        def $fn(n):
            """Lista n, n-1, ..., 1 (vazia quando n e zero)."""
            $m
                return []
            return [n] + $fn(n - 1)
    ''', fn=fn, m=MARKER)
    bug, fix = "if n < 0:", "if n <= 0:"
    tests = [Test("three", [Eq(call(fn, 3), [3, 2, 1])]),
             Test("one", [Eq(call(fn, 1), [1])]),
             Test("zero_is_empty", [Eq(call(fn, 0), [])])]
    return mk(FAM, "countdown_list", d, f"{d.item}_countdown", chunk, bug, fix, tests, fn,
              behavior="listar de n ate 1, sem incluir o zero",
              cause="o caso base n < 0 deixa passar n == 0 e o zero entra na lista",
              fix_desc="parar em n <= 0", api="recursao e concatenacao de listas nativas",
              lesson="Confirme o caso base com n = 0: ele decide se o zero entra no resultado.",
              edge="n = 3, n = 1 e n = 0", perf="O(n^2) pela concatenacao, sem mudanca.")


# --------------------------------------------------------------------------------------------
# Enumeracao
# --------------------------------------------------------------------------------------------
VARIANTS = 2


def candidates() -> list[tuple[str, str, int, int]]:
    """Todas as combinacoes (familia, forma, indice do dominio, variante), em ordem estavel."""
    out = []
    for family in sorted(SHAPES):
        for name, _fn in SHAPES[family]:
            for di in range(len(DOMAINS)):
                for k in range(VARIANTS):
                    out.append((family, name, di, k))
    return out


def build_draft(family: str, shape_name: str, di: int, k: int) -> Draft:
    fn = dict(SHAPES[family])[shape_name]
    rng = Rng("simplicio-v2", family, shape_name, di, k)
    return fn(DOMAINS[di], rng, k)


def top_level_names(source: str) -> set[str]:
    return {n.name for n in ast.parse(source).body if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
