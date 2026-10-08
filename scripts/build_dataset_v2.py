#!/usr/bin/env python3
"""Gerador do dataset v2 do Simplicio: tarefas de correcao de bug sinteticas e VERIFICADAS.

Fontes: somente locais e sinteticas (nenhum repositorio externo, nenhuma rede, nenhum modelo, nenhuma
GPU). Cada exemplo e um arquivo Python pequeno com um bug, um teste pytest que falha antes e passa
depois, e um patch SEARCH/REPLACE (marcadores de 4 caracteres) que corrige o bug, embrulhado no
envelope de 50 pontos do simplicio-loop (generate_dataset.py, scripts/loop_check.py).

Um exemplo so e emitido se TODAS as condicoes valem (senao e contado em "rejected" no manifest):
  1. o teste roda no harness do eval (benchmarks/harness/harness.py, mesma sandbox) e FALHA no
     arquivo original (pelo menos um teste, sem erro de importacao) e PASSA no arquivo corrigido;
  2. o patch que esta DENTRO do envelope final, lido com harness.parse_blocks e aplicado com
     harness.apply_blocks ao original, reproduz o arquivo corrigido que foi testado; reaplicar o
     patch ao corrigido falha com search_not_found (idempotencia);
  3. o envelope completo passa em scripts/loop_check.py --complete (os 50 pontos, uma vez cada);
  4. nenhum string do exemplo compartilha um 13-grama com as tarefas do eval
     (benchmarks/harness/tasks) nem com data/unseen_eval_120.json (benchmarks/harness/decontam_tasks.py).
Os fatos do envelope (contagens do pytest, sha256, linhas, assinaturas) vem de execucao e de ast, nao
de texto escrito a mao; nada depende de tempo, de versao do Python ou de PYTHONHASHSEED.

Determinismo: sem o modulo random. Toda escolha vem de sha256 (dataset_v2_families.Rng). Duas
execucoes com os mesmos argumentos produzem arquivos identicos byte a byte.

Uso (precisa de pytest, como o harness: python3 -m pip install -r benchmarks/harness/requirements.txt):
    python3 scripts/build_dataset_v2.py                      # escreve data/v2/{train,val}.jsonl e manifest.json
    python3 scripts/build_dataset_v2.py --out /tmp/x --limit 24   # modo pequeno
    python3 scripts/build_dataset_v2.py --verify data/v2 --sample 50   # reverifica a partir dos arquivos
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
import os
import py_compile
import re
import sys
import tempfile
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


fam = _load("simplicio_dataset_v2_families", HERE / "dataset_v2_families.py")
loop_check = _load("simplicio_loop_check", HERE / "loop_check.py")
harness = _load("simplicio_eval_harness", ROOT / "benchmarks/harness/harness.py")
decontam = _load("simplicio_decontam_tasks", ROOT / "benchmarks/harness/decontam_tasks.py")

SEED = "simplicio-v2"
TRAIN_CAP = 2000
VAL_MODULUS = 10          # um grupo em cada 10 vai para val
BATCH = 32
EVAL_TASKS = ROOT / "benchmarks/harness/tasks"
UNSEEN_EVAL = ROOT / "data/unseen_eval_120.json"
DEFAULT_OUT = ROOT / "data/v2"
FAILED_RE = re.compile(r"^FAILED \S+?::(test_\w+)", re.M)


# --------------------------------------------------------------------------------------------
# Montagem do arquivo, do teste e do patch
# --------------------------------------------------------------------------------------------
@dataclass
class Case:
    draft: fam.Draft
    domain: fam.Domain
    group: str
    stem: str
    edit_file: str
    test_file: str
    original: str
    fixed: str
    search: str
    replace: str
    tests: list
    test_source: str
    key: str


def sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def make_patch(template: str, bug: str, fix: str):
    """Troca o marcador pelo trecho com defeito e pelo corrigido e devolve (original, fixed, search,
    replace). O SEARCH e o menor trecho de linhas que difere, ampliado para cima ate ser unico e, se
    preciso, para baixo ate deixar de existir no arquivo corrigido."""
    lines = template.split("\n")
    at = [i for i, line in enumerate(lines) if line.strip() == fam.MARKER]
    if len(at) != 1:
        return None
    i0 = at[0]
    indent = lines[i0][: len(lines[i0]) - len(lines[i0].lstrip())]
    b = [indent + x if x else x for x in bug.split("\n")]
    f = [indent + x if x else x for x in fix.split("\n")]
    orig_lines = lines[:i0] + b + lines[i0 + 1:]
    fixed_lines = lines[:i0] + f + lines[i0 + 1:]
    p = 0
    while p < min(len(b), len(f)) and b[p] == f[p]:
        p += 1
    s = 0
    while s < min(len(b), len(f)) - p and b[-1 - s] == f[-1 - s]:
        s += 1
    start = i0 + p
    search_lines = b[p: len(b) - s]
    replace_lines = f[p: len(f) - s]
    original = "\n".join(orig_lines)
    fixed = "\n".join(fixed_lines)
    tries = 0
    while not (search_lines and original.count("\n".join(search_lines)) == 1):
        if start == 0 or tries >= 3:
            return None
        start -= 1
        prev = orig_lines[start]
        search_lines = [prev] + search_lines
        replace_lines = [prev] + replace_lines
        tries += 1
    # Idempotencia: depois do patch o SEARCH nao pode mais existir. Isso falha quando o SEARCH e prefixo
    # da linha corrigida ("return a - b" dentro de "return a - b + 1") ou quando o patch so insere
    # linhas. Entao o SEARCH desce ate incluir a linha seguinte, e a correcao fica entre as duas.
    tries = 0
    while "\n".join(search_lines) in fixed:
        end = start + len(search_lines)
        if tries >= 5 or end >= len(orig_lines):
            return None
        search_lines = search_lines + [orig_lines[end]]
        replace_lines = replace_lines + [orig_lines[end]]
        tries += 1
    # Prefere terminar numa linha com texto (a proxima definicao) a terminar em linhas em branco.
    if not search_lines[-1].strip():
        end = start + len(search_lines)
        j = end
        while j < len(orig_lines) and j - end < 3 and not orig_lines[j].strip():
            j += 1
        if j < len(orig_lines) and orig_lines[j].strip():
            search_lines = search_lines + orig_lines[end: j + 1]
            replace_lines = replace_lines + orig_lines[end: j + 1]
    search, replace = "\n".join(search_lines), "\n".join(replace_lines)
    if original.count(search) != 1 or original.replace(search, replace, 1) != fixed or search in fixed:
        return None
    return original, fixed, search, replace


def patch_block(search: str, replace: str) -> str:
    return f"<<<< SEARCH\n{search}\n====\n{replace}\n>>>> REPLACE"


def assemble(family: str, shape_name: str, di: int, k: int) -> Case | None:
    draft = fam.build_draft(family, shape_name, di, k)
    domain = fam.DOMAINS[di]
    r = fam.Rng(SEED, "assemble", family, shape_name, di, k)
    helper_fns = r.sample(fam.HELPERS, 2)[: 1 + r.below(2)]
    chunks = [draft.chunk]
    tests = list(draft.tests)
    taken = {draft.fn}
    for h in helper_fns:
        name, code, htests = h(domain, r)
        if name in taken:
            continue
        taken.add(name)
        chunks.append(code)
        tests.extend(htests)
    order = r.sample(range(len(chunks)), len(chunks))
    patch = None
    for rot in range(len(order)):  # se o alvo e o ultimo e o patch nao fecha, tenta outra ordem
        rotated = order[rot:] + order[:rot]
        template = f'"""Funcoes de {domain.items}."""\n\n\n' + "\n\n\n".join(chunks[i] for i in rotated) + "\n"
        patch = make_patch(template, draft.bug, draft.fix)
        if patch is not None:
            break
    if patch is None:
        return None
    original, fixed, search, replace = patch
    names = [t.name for t in tests]
    if len(set(names)) != len(names):
        return None
    top = fam.top_level_names(original)
    rendered = [fam.render_test(t) for t in tests]
    used = sorted({n.id for src in rendered for n in ast.walk(ast.parse(src)) if isinstance(n, ast.Name)} & top)
    header = ("import pytest\n" if any(s.kind == "raises" for t in tests for s in t.steps) else "")
    stem = draft.stem
    test_source = header + f"from {stem} import {', '.join(used)}\n\n\n" + "\n\n\n".join(rendered) + "\n"
    key = sha("\0".join((original, test_source, search, replace)))
    return Case(draft=draft, domain=domain, group=f"{family}/{shape_name}/{domain.item}", stem=stem,
                edit_file=f"{stem}.py", test_file=f"test_{stem}.py", original=original, fixed=fixed,
                search=search, replace=replace, tests=tests, test_source=test_source, key=key)


# --------------------------------------------------------------------------------------------
# Fatos medidos com ast
# --------------------------------------------------------------------------------------------
def defs(source: str) -> dict[str, tuple[int, str]]:
    lines = source.split("\n")
    out: dict[str, tuple[int, str]] = {}
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            out[node.name] = (node.lineno, lines[node.lineno - 1].strip().rstrip(":"))
    return out


def docstrings(source: str) -> list[tuple[str, str | None]]:
    out = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.ClassDef)):
            doc = ast.get_docstring(node, clean=False)
            if doc is not None:
                out.append((getattr(node, "name", "<module>"), doc))
    return out


def comment_lines(source: str) -> list[str]:
    return [x.strip() for x in source.split("\n") if x.strip().startswith("#")]


def imports(source: str) -> list[str]:
    return sorted(ast.unparse(n) for n in ast.walk(ast.parse(source)) if isinstance(n, (ast.Import, ast.ImportFrom)))


def arg_names(source: str, name: str) -> list[str] | None:
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return [a.arg for a in node.args.args]
    return None


# --------------------------------------------------------------------------------------------
# Execucao real: harness do eval
# --------------------------------------------------------------------------------------------
def run_harness(case: Case):
    """Roda o teste no harness (mesma sandbox do eval) no original e no corrigido. Devolve
    (before, after, patched_source, reapply_error) ou levanta Reject."""
    blocks = harness.parse_blocks(patch_block(case.search, case.replace))
    patched = harness.apply_blocks(case.original, blocks)
    if patched != case.fixed:
        raise Reject("patch_differs_from_fix")
    try:
        harness.apply_blocks(patched, blocks)
    except harness.PatchError as exc:
        if "search_not_found" not in str(exc):
            raise Reject("reapply_other_error") from exc
    else:
        raise Reject("patch_not_idempotent")
    with tempfile.TemporaryDirectory(prefix="simplicio-v2-") as tmp:
        task_dir = Path(tmp) / "task"
        (task_dir / "files").mkdir(parents=True)
        (task_dir / "tests").mkdir()
        (task_dir / "files" / case.edit_file).write_text(case.original, encoding="utf-8")
        (task_dir / "tests" / case.test_file).write_text(case.test_source, encoding="utf-8")
        task = {"dir": task_dir, "edit_file": case.edit_file}
        before = harness.check_source(task, case.original)
        after = harness.check_source(task, patched)
        py_compile.compile(str(task_dir / "files" / case.edit_file), cfile=str(Path(tmp) / "x.pyc"), doraise=True)
    return before, after, patched


class Reject(Exception):
    """Candidato descartado; o texto e a razao contada no manifest."""


# --------------------------------------------------------------------------------------------
# Envelope de 50 pontos
# --------------------------------------------------------------------------------------------
def build_trajectory(case: Case, observed_before: dict, patched: str) -> str:
    dr = case.draft
    fn, file, tfile = dr.fn, case.edit_file, case.test_file
    total = len(case.tests)
    failed = [n for n, msg in observed_before.items() if msg is not None]
    n_failed, n_passed = len(failed), total - len(failed)
    first = failed[0]
    # A mensagem observada so entra na trajetoria como leitura do codigo (sem verbo de execucao
    # e sem texto de excecao): a trajetoria nao pode apresentar saida de comando como vista.
    first_msg = " ".join(str(observed_before[first]).split())
    first_msg = re.sub(r" levantou (\w+):.*$", r" levanta \1", first_msg).replace(" devolveu ", " devolve ")
    src_lines = case.original.split("\n")
    n_lines = len(src_lines) - (1 if src_lines[-1] == "" else 0)
    ln = case.original[: case.original.index(case.search)].count("\n") + 1
    removed, added = len(case.search.split("\n")), len(case.replace.split("\n"))
    changed = removed + added
    economy = max(0, round(100 * (1 - changed / n_lines)))
    d_b, d_a = defs(case.original), defs(patched)
    top_defs = sorted(((v[0], k) for k, v in d_b.items()), key=lambda x: x[0])
    graph = ", ".join(f"{name} (linha {line})" for line, name in top_defs)
    imported = re.search(r"^from \S+ import (.+)$", case.test_source, re.M).group(1).split(", ")
    sig_b, sig_a = d_b[fn][1], d_a[fn][1]
    docs = len(docstrings(case.original))
    comments = len(comment_lines(case.original))
    names = ", ".join(sorted(d_b))
    added_syms = sorted(set(d_a) - set(d_b))
    changed_sigs = sorted(n for n in d_b if n in d_a and d_b[n][1] != d_a[n][1])
    if sig_b == sig_a:
        sig_text = f"Assinatura de {fn} inalterada: {sig_b}."
    else:
        sig_text = f"Assinatura de {fn} muda de '{sig_b}' para '{sig_a}' so no valor padrao; os nomes dos parametros ficam iguais."
    if not changed_sigs and not added_syms:
        iface = f"Simbolos do modulo ({names}) com as mesmas assinaturas antes e depois."
    else:
        parts = []
        if changed_sigs:
            parts.append("assinatura alterada so em " + ", ".join(changed_sigs))
        if added_syms:
            parts.append("simbolo novo " + ", ".join(added_syms) + f" dentro de {fn}")
        iface = "Interface: " + "; ".join(parts) + "; os demais simbolos ficam iguais."
    imps = imports(case.original)
    imp_text = ("Nenhum import no arquivo, antes ou depois do patch." if not imps
                else "Imports identicos antes e depois: " + "; ".join(imps) + ".")
    runtime = "python3 e pytest; o modulo so usa builtins." if not imps else "python3 e pytest; imports: " + "; ".join(imps) + "."
    doc_text = f"{docs} docstring(s) e {comments} comentario(s) de linha, todos identicos antes e depois do patch."
    block = patch_block(case.search, case.replace)

    orient = [
        ("Identificacao de Raiz", f"Raiz em /workspace/{case.stem} com {file} e tests/{tfile}; sem pyproject.toml, requirements ou configs."),
        ("Grafo de Simbolos", f"{file} define {graph}; tests/{tfile} importa {', '.join(imported)}."),
        ("Assinaturas de Tipos", f"Alvo: {sig_b} (linha {d_b[fn][0]}), sem anotacoes de tipo."),
        ("Isolamento de Estado", dr.state),
        ("Custo de Leitura", f"Lidos {file} inteiro ({n_lines} linhas) e tests/{tfile}; nenhum outro arquivo existe."),
        ("Regras Locais", "Sem AGENTS.md, CLAUDE.md ou linter configurado; vale o estilo do arquivo: indentacao de 4 espacos."),
        ("Runtimes & Deps", runtime),
        ("Camadas", f"Camada unica de logica pura em {file}; sem I/O, rede ou banco."),
        ("Ambiguidade", f"Requisito sem ambiguidade: {dr.behavior}. O teste {first} fixa o esperado; por leitura do codigo, {first_msg}."),
        ("Baseline Handle", f"Plano: antes de editar, rodar sha256sum {file} e pytest -q tests e registrar a saida real depois; nenhuma saida e mostrada aqui. Espera-se {n_failed} falha(s) neste baseline."),
    ]
    plan = [
        ("Decomposicao Atomica", f"Passo 1: {dr.fix_desc} (linha {ln}). Passo 2: rodar pytest tests e conferir que os {total} testes ficam verdes."),
        ("Roteamento", f"simplicio_edit com um bloco SEARCH/REPLACE (marcadores de 4 caracteres) em {file}."),
        ("Plano Linear", f"Ler {file}, aplicar o patch, rodar py_compile e rodar pytest tests, nessa ordem."),
        ("Barreira Fan-Out", f"Somente {file} muda; tests/{tfile} fica intocado."),
        ("Efeito Colateral", f"Efeito restrito a {fn}; espera-se que os {n_passed} testes que ja passam continuem verdes depois do patch."),
        ("Zero Fantasma", f"Nenhum import novo e nenhum simbolo inexistente: o patch usa so {dr.api}."),
        ("Hierarquia de Restricoes", f"Teste do usuario > contrato de {fn} ({sig_b}) > estilo do arquivo."),
        ("Condicao de Parada", f"Parar quando pytest tests nao tiver falhas: espera-se {total} testes verdes e 0 falhas depois do patch."),
        ("Rollback", f"Restaurar {file} ao conteudo original (sha256 registrado no baseline) se algum teste regredir."),
        ("Rationale", f"Causa raiz: {dr.cause}."),
    ]
    patch = [
        ("Diff Cirurgico", None),
        ("Preservacao Adjacente", f"Das {n_lines} linhas de {file}, {removed} saem e {added} entram; as outras {n_lines - removed} ficam identicas."),
        ("Preservacao de Comentarios", doc_text),
        ("Minimo Suficiente", f"{removed} linha(s) no SEARCH e {added} no REPLACE, so o trecho com o defeito (a partir da linha {ln})."),
        ("Alinhamento de Assinatura", sig_text),
        ("Importacoes Nao-Destrutivas", imp_text),
        ("Codegen Estruturado", f"Um bloco SEARCH/REPLACE; o SEARCH aparece 1 vez em {file}, entao a troca e inequivoca."),
        ("Isolamento de Configs", f"Nenhum arquivo de configuracao existe; so {file} foi editado."),
        ("Idempotencia", "O SEARCH nao existe mais no arquivo corrigido; reaplicar o patch deve falhar com search_not_found em vez de duplicar a correcao."),
        ("Validacao Sintatica", f"Rodar ast.parse de {file} depois do patch e esperar que nao haja SyntaxError."),
    ]
    validate = [
        ("Validacao Estatica", f"Rodar python3 -m py_compile {file} depois do patch e esperar exit 0."),
        ("Testes Direcionados", f"Rodar pytest tests/{tfile}::{first}: espera-se falha antes do patch e sucesso depois."),
        ("Leitura Cirurgica", f"Falha esperada pela leitura do codigo: {first_msg}; a causa esta na linha {ln}: {dr.cause}."),
        ("Loop Guiado por Falha", "Ciclo guiado pela falha: ler a falha, aplicar o patch e rodar de novo; espera-se verde em 1 ciclo, sem falha nova."),
        ("Inibicao de Loop", "Limite de 1 tentativa de patch; se o mesmo erro voltar, parar e reler em vez de repetir o patch."),
        ("Anti-Placebo", f"Espera-se {n_failed} falha(s) antes do patch e {total} testes verdes, 0 falhas, depois; um teste que ja passa antes do patch nao prova a correcao."),
        ("Regressao Cruzada", f"Rodar pytest tests (diretorio inteiro): espera-se {total} testes verdes, incluindo os {n_passed} que ja passam."),
        ("Shell Sanitizado", "Usar pytest -q -p no:cacheprovider tests, sem pipes nem redirecionamentos, para a saida do pytest aparecer completa."),
        ("Warnings Silenciosos", "Conferir que a saida do pytest nao tem linha de warning."),
        ("Casos de Borda", f"Cobertos por tests/{tfile}: {dr.edge}."),
    ]
    deliver = [
        ("Poda de Tokens", "Resposta curta: um bloco de patch e so os fatos que a execucao real confirmar, sem prosa extra."),
        ("Convergencia Deterministica", 'simplicio_deliver(status="VERIFIED_GREEN")'),
        ("Resumo Explicativo", f"{fn}: {dr.fix_desc}; confirmar com pytest tests depois de rodar."),
        ("Limpeza", "Nao deixar arquivo temporario no repo; usar pytest sem cache (-p no:cacheprovider) e sem bytecode."),
        ("Performance", dr.perf),
        ("Metricas de Economia", f"Diff de {changed} linhas (SEARCH+REPLACE) contra {n_lines} linhas do arquivo: {economy}% de economia."),
        ("Validacao de Interface", iface),
        ("Aprendizado Persistido", dr.lesson),
        ("Zero Alucinacao", "Afirmar so o que a execucao real mostrar: pytest antes e depois, ast.parse, py_compile e sha256; nada neste texto e saida observada."),
        ("Selo de Entrega", f"SELO SIMPLICIO: COMMIT_READY (sha256 final de {file} a registrar a partir da execucao real)"),
    ]
    out = ["<simplicio_loop>"]
    number = 1
    for tag, points in (("orient", orient), ("plan", plan), ("patch", patch), ("validate", validate), ("deliver", deliver)):
        out.append(f"<{tag}>")
        for name, text in points:
            if text is None:
                out.append(f"[Ponto {number}: {name}]")
                out.append(block)
            else:
                out.append(f"[Ponto {number}: {name}] {text}")
            number += 1
        out.append(f"</{tag}>")
    out.append("</simplicio_loop>")
    return "\n".join(out)


# --------------------------------------------------------------------------------------------
# Processamento de um candidato
# --------------------------------------------------------------------------------------------
_EVAL_GRAMS: set | None = None


def eval_grams() -> set:
    """13-gramas das tarefas do eval (criterio de decontam_tasks.py) e de data/unseen_eval_120.json."""
    global _EVAL_GRAMS
    if _EVAL_GRAMS is None:
        grams: set = set()
        for task_dir in decontam.task_dirs(EVAL_TASKS):
            grams |= decontam.ngrams(decontam.task_text(task_dir))
        for entry in json.loads(UNSEEN_EVAL.read_text(encoding="utf-8")):
            texts = list(decontam.strings(entry))
            grams |= decontam.ngrams("\n".join(texts))
            for text in texts:
                grams |= decontam.ngrams(text)
        _EVAL_GRAMS = grams
    return _EVAL_GRAMS


def split_of(group: str) -> str:
    return "val" if int(sha(f"{SEED}:split:{group}")[:8], 16) % VAL_MODULUS == 0 else "train"


def process(cand: tuple[str, str, int, int]) -> dict | str:
    """Devolve o registro verificado ou a razao (str) da rejeicao."""
    family, shape_name, di, k = cand
    try:
        case = assemble(family, shape_name, di, k)
        if case is None:
            return "assemble_failed"
        observed_before = fam.observe(case.original, case.tests)
        observed_after = fam.observe(case.fixed, case.tests)
        failed_obs = {n for n, m in observed_before.items() if m is not None}
        if not failed_obs:
            return "bug_not_observed"
        if any(m is not None for m in observed_after.values()):
            return "fix_not_observed"
        before, after, patched = run_harness(case)
        if before["timed_out"] or after["timed_out"]:
            return "timeout"
        rb, ra = before["report"], after["report"]
        total = len(case.tests)
        if before["passed"] or rb["errors"] or rb["skipped"] or rb["tests"] != total or rb["failures"] != len(failed_obs):
            return "before_not_a_clean_failure"
        if set(FAILED_RE.findall(before["stdout_tail"])) != failed_obs:
            return "failed_names_mismatch"
        if not after["passed"] or ra["tests"] != total or ra["failures"] or ra["errors"] or ra["skipped"]:
            return "after_not_green"
        if "warning" in after["stdout_tail"].lower():
            return "warnings_present"
        if docstrings(case.original) != docstrings(patched) or comment_lines(case.original) != comment_lines(patched):
            return "docstring_or_comment_changed"
        if imports(case.original) != imports(patched):
            return "imports_changed"
        if arg_names(case.original, case.draft.fn) != arg_names(patched, case.draft.fn):
            return "parameter_names_changed"
        trajectory = build_trajectory(case, observed_before, patched)
        if not trajectory.isascii():
            return "non_ascii"
        if loop_check.check(trajectory, complete=True):
            return "loop_check_failed"
        if harness.parse_blocks(trajectory) != harness.parse_blocks(patch_block(case.search, case.replace)):
            return "envelope_patch_differs"
        if harness.apply_blocks(case.original, harness.parse_blocks(trajectory)) != patched:
            return "envelope_patch_does_not_fix"
        record = {
            "instruction": f"Corrija {case.draft.fn} ({case.edit_file}): o comportamento esperado e {case.draft.behavior}.",
            "context": f"Arquivo: {case.edit_file}\nCodigo Atual:\n{case.original}",
            "simplicio_trajectory": trajectory,
            "id": f"v2-{family}-{shape_name}-{case.key[:10]}",
            "family": family,
            "shape": shape_name,
            "domain": case.domain.item,
            "edit_file": case.edit_file,
            "test_file": f"tests/{case.test_file}",
            "test_source": case.test_source,
            "tests_total": total,
            "tests_failed_before": len(failed_obs),
            "original_sha256": sha(case.original),
            "patched_sha256": sha(patched),
        }
        grams = eval_grams()
        for text in decontam.strings(record):
            if decontam.ngrams(text) & grams:
                return "decontamination_overlap"
        record["_group"] = case.group
        record["_key"] = case.key
        return record
    except Reject as exc:
        return str(exc)
    except Exception as exc:  # noqa: BLE001 - um candidato com defeito nao derruba a geracao; fica no manifest
        return f"internal_error:{type(exc).__name__}:{' '.join(str(exc).split())[:80]}"


# --------------------------------------------------------------------------------------------
# Geracao
# --------------------------------------------------------------------------------------------
def interleaved_candidates() -> list[tuple[str, str, int, int]]:
    """Round-robin entre as formas; dentro de cada forma, ordem por sha256 (estavel)."""
    per_shape: dict[tuple[str, str], list] = {}
    for cand in fam.candidates():
        per_shape.setdefault((cand[0], cand[1]), []).append(cand)
    queues = [sorted(items, key=lambda c: sha(f"{SEED}:order:{c}")) for _, items in sorted(per_shape.items())]
    out = []
    for i in range(max(len(q) for q in queues)):
        for q in queues:
            if i < len(q):
                out.append(q[i])
    return out


def generate(train_cap: int, limit: int | None, workers: int, log=None):
    order = interleaved_candidates()
    eval_grams()  # constroi o indice uma vez, antes das threads
    seen: set[str] = set()
    seen_ids: set[str] = set()
    seen_prompts: set[tuple[str, str]] = set()
    records: list[dict] = []
    rejected: Counter = Counter()
    counts = {"train": 0, "val": 0}
    pos = 0
    done = False
    with ThreadPoolExecutor(max_workers=workers) as pool:
        while pos < len(order) and not done:
            batch = order[pos: pos + BATCH]
            pos += BATCH
            for result in pool.map(process, batch):
                if isinstance(result, str):
                    rejected[result] += 1
                    continue
                prompt = (result["instruction"], result["context"])
                if result["_key"] in seen or result["id"] in seen_ids:
                    rejected["duplicate_example"] += 1
                    continue
                if prompt in seen_prompts:
                    rejected["duplicate_prompt"] += 1
                    continue
                seen.add(result["_key"])
                seen_ids.add(result["id"])
                seen_prompts.add(prompt)
                split = split_of(result.pop("_group"))
                result.pop("_key")
                result["_split"] = split
                records.append(result)
                counts[split] += 1
                if counts["train"] >= train_cap or (limit is not None and len(records) >= limit):
                    done = True
                    break
            if log:
                log(f"  {len(records)} aceitos ({counts['train']} train, {counts['val']} val), "
                    f"{sum(rejected.values())} rejeitados, {min(pos, len(order))}/{len(order)} candidatos")
    return records, rejected, len(order)


def order_key(record: dict) -> str:
    return sha(f"{SEED}:file-order:{record['id']}")


def write_outputs(records: list[dict], rejected: Counter, enumerated: int, out: Path, train_cap: int) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    files = {}
    for split in ("train", "val"):
        rows = sorted((r for r in records if r["_split"] == split), key=order_key)
        with open(out / f"{split}.jsonl", "w", encoding="utf-8", newline="\n") as fh:
            for row in rows:
                fh.write(json.dumps({k: v for k, v in row.items() if k != "_split"}, ensure_ascii=False) + "\n")
        files[split] = rows
    fams: dict[str, dict] = {}
    for split, rows in files.items():
        for row in rows:
            entry = fams.setdefault(row["family"], {"train": 0, "val": 0, "shapes": {}})
            entry[split] += 1
            entry["shapes"][row["shape"]] = entry["shapes"].get(row["shape"], 0) + 1
    manifest = {
        "generator": "scripts/build_dataset_v2.py",
        "sources": "somente locais e sinteticas: nenhum repositorio externo, nenhuma rede, nenhum modelo",
        "seed": SEED,
        "train_cap": train_cap,
        "split": f"grupo (familia/forma/dominio) -> sha256('{SEED}:split:<grupo>')[:8] % {VAL_MODULUS} == 0 vai para val",
        "candidates_enumerated": enumerated,
        "files": {
            f"{split}.jsonl": {
                "examples": len(rows),
                "sha256": hashlib.sha256((out / f"{split}.jsonl").read_bytes()).hexdigest(),
            }
            for split, rows in files.items()
        },
        "families": {k: {**v, "shapes": dict(sorted(v["shapes"].items()))} for k, v in sorted(fams.items())},
        "rejected": dict(sorted(rejected.items())),
        "decontamination": decontamination(out),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return manifest


# --------------------------------------------------------------------------------------------
# Decontaminacao do arquivo escrito
# --------------------------------------------------------------------------------------------
def decontamination(data_dir: Path) -> dict:
    """Aplica o criterio de decontam_tasks.py (13-gramas) aos .jsonl de data_dir contra as tarefas do
    eval e contra cada entrada de data/unseen_eval_120.json. Nao escreve nada nas tarefas."""
    report = decontam.build_report(tasks_dir=EVAL_TASKS, data_dir=data_dir, root=data_dir)
    entries = json.loads(UNSEEN_EVAL.read_text(encoding="utf-8"))
    index: dict[tuple, list[int]] = {}
    for i, entry in enumerate(entries):
        texts = list(decontam.strings(entry))
        grams = decontam.ngrams("\n".join(texts))
        for text in texts:
            grams |= decontam.ngrams(text)
        for gram in grams:
            index.setdefault(gram, []).append(i)
    hit: set[int] = set()
    for path in decontam.corpus_files(data_dir):
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                for text in decontam.strings(json.loads(line)):
                    for gram in decontam.ngrams(text) & index.keys():
                        hit.update(index[gram])
    bad_tasks = [t["id"] for t in report["tasks"] if t["overlaps"]]
    return {
        "criterion": f"{decontam.N}-gramas de {decontam.TOKENIZER}, como benchmarks/harness/decontam_tasks.py",
        "eval_tasks": {"checked": len(report["tasks"]), "with_overlap": len(bad_tasks), "ids_with_overlap": bad_tasks},
        "unseen_eval_120": {"checked": len(entries), "with_overlap": len(hit), "indexes_with_overlap": sorted(hit)},
    }


# --------------------------------------------------------------------------------------------
# Reverificacao a partir dos arquivos escritos
# --------------------------------------------------------------------------------------------
def verify_record(record: dict) -> list[str]:
    """Reverifica um registro so com o que esta nele: envelope completo, patch aplicado ao codigo do
    contexto, hashes e o teste real no harness (falha antes, passa depois)."""
    problems = [f"loop_check: {p}" for p in loop_check.check(record["simplicio_trajectory"], complete=True)]
    prefix = f"Arquivo: {record['edit_file']}\nCodigo Atual:\n"
    if not record["context"].startswith(prefix):
        return problems + ["contexto: formato inesperado"]
    original = record["context"][len(prefix):]
    blocks = harness.parse_blocks(record["simplicio_trajectory"])
    try:
        patched = harness.apply_blocks(original, blocks)
    except harness.PatchError as exc:
        return problems + [f"patch: {exc}"]
    if len(blocks) != 1:
        problems.append(f"patch: {len(blocks)} blocos em vez de 1")
    if sha(original) != record["original_sha256"] or sha(patched) != record["patched_sha256"]:
        problems.append("hash: sha256 do original ou do corrigido nao confere com o registro")
    with tempfile.TemporaryDirectory(prefix="simplicio-v2-verify-") as tmp:
        task_dir = Path(tmp) / "task"
        (task_dir / "files").mkdir(parents=True)
        (task_dir / "tests").mkdir()
        (task_dir / "files" / record["edit_file"]).write_text(original, encoding="utf-8")
        (task_dir / "tests" / Path(record["test_file"]).name).write_text(record["test_source"], encoding="utf-8")
        task = {"dir": task_dir, "edit_file": record["edit_file"]}
        before = harness.check_source(task, original)
        after = harness.check_source(task, patched)
    if before["passed"] or before["report"]["failures"] != record["tests_failed_before"] or before["report"]["errors"]:
        problems.append("teste: o original nao falha como registrado")
    if not after["passed"] or after["report"]["tests"] != record["tests_total"]:
        problems.append("teste: o arquivo corrigido nao passa")
    return problems


def verify_dir(path: Path, sample: int | None) -> int:
    bad = total = 0
    for name in ("train.jsonl", "val.jsonl"):
        rows = [json.loads(x) for x in (path / name).read_text(encoding="utf-8").splitlines() if x.strip()]
        step = max(1, len(rows) // sample) if sample else 1
        for row in rows[::step]:
            total += 1
            problems = verify_record(row)
            if problems:
                bad += 1
                print(f"FALHA {row['id']}: " + " | ".join(problems))
    print(f"verify: {total} registros reverificados, {bad} com problema")
    return 1 if bad else 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Gera o dataset v2 (exemplos sinteticos verificados no formato do loop).")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT, help="diretorio de saida (padrao data/v2)")
    ap.add_argument("--train-cap", type=int, default=TRAIN_CAP, help="maximo de exemplos no train (padrao 2000)")
    ap.add_argument("--limit", type=int, help="modo pequeno: para depois de N exemplos aceitos")
    ap.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1))
    ap.add_argument("--verify", type=Path, metavar="DIR", help="reverifica os .jsonl de DIR em vez de gerar")
    ap.add_argument("--sample", type=int, help="com --verify: reverifica cerca de N registros por arquivo")
    args = ap.parse_args(argv)
    if args.verify:
        return verify_dir(args.verify, args.sample)
    if not importlib.util.find_spec("pytest"):
        print("erro: pytest nao esta instalado (o harness precisa dele): "
              "instale as dependencias de benchmarks/harness/requirements.txt", file=sys.stderr)
        return 2
    records, rejected, enumerated = generate(args.train_cap, args.limit, args.workers, log=lambda m: print(m, file=sys.stderr))
    manifest = write_outputs(records, rejected, enumerated, args.out, args.train_cap)
    deco = manifest["decontamination"]
    print(f"train: {manifest['files']['train.jsonl']['examples']}  val: {manifest['files']['val.jsonl']['examples']}  "
          f"rejeitados: {dict(rejected)}")
    print(f"decontaminacao: eval_tasks {deco['eval_tasks']['with_overlap']}/{deco['eval_tasks']['checked']}, "
          f"unseen_eval_120 {deco['unseen_eval_120']['with_overlap']}/{deco['unseen_eval_120']['checked']}")
    bad = deco["eval_tasks"]["with_overlap"] + deco["unseen_eval_120"]["with_overlap"]
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
