# Dataset v2: estatisticas

Gerado por `python3 scripts/build_dataset_v2.py`. Nao edite a mao.

- Seed: 27
- Teto por repo: 300
- Repo de validacao: python-poetry/poetry
- Filtros: commit sem merge, 1 a 60 linhas alteradas em um arquivo .py fora de testes, docs, vendor, benchmarks/harness/tasks e data; arquivo pai <= 400 linhas e <= 16000 caracteres; mudanca de codigo (AST sem docstrings); round-trip SEARCH/REPLACE exato; sem credencial literal; 13-gram contra benchmarks/harness/tasks (instrucao, codigo e oracle) e contra data/unseen_eval_120.json (criterio de benchmarks/harness/decontam_tasks.py)
- Instrucao: assunto do commit quando ele toca 1 arquivo e comeca com um verbo no imperativo; texto vetado contra o diff (`curated_from_commit`) nos outros casos
- Linhas: train 1, val 0
- Origem: 1 de commit real, 0 sinteticos
- Marco de treino: 2000 linhas em train; faltam 1999
- Caracteres por linha: max 15292, media 15292

Texto de cada licenca: `data/v2/LICENSES/<owner>__<repo>.txt`.

| Repo | Licenca | Commit fixado | Commits | Candidatos | Validados | Contaminados | Usados | Split |
|---|---|---|---|---|---|---|---|---|
| simpletibr/simplicio-27b | Apache-2.0 | `b61dccd400457c5112a2c2c1c74b94a90edc8376` | 61 | 47 | 1 | 0 | 1 | train |

Descartes por motivo (candidatos que nao viraram linha):

- simpletibr/simplicio-27b: bad_message 3, multi_file_commit 43

## Pendencias

Repositorios nao processados nesta geracao (rode `python3 scripts/build_dataset_v2.py --external --min-train 2000`). Pins e licencas vem da issue #28 e nao foram conferidos na rede.

| Repo | Licenca | Commit fixado | Arquivo de licenca | Split |
|---|---|---|---|---|
| django/django | BSD-3-Clause | `135d7c606fc66a9f47f690e5f242ced386524954` | LICENSE | train |
| ipython/ipython | BSD-3-Clause | `93dde62b5b76275dbd1a1cfda328969fa59caa08` | LICENSE | train |
| scrapy/scrapy | BSD-3-Clause | `eba114d98ecd3986f23af6facb524fc4ac4d8f71` | LICENSE | train |
| pypa/pip | MIT | `a7002c9771a6c3f0317a4e6b9fbdcd22e643f7b6` | LICENSE.txt | train |
| conan-io/conan | MIT | `e85a86e84b75a8bc80ba17e19be8df7e5c6df1fa` | LICENSE.md | train |
| mitmproxy/mitmproxy | MIT | `5253dcbd1d8f0522de097bfe56918fe12a0f267a` | LICENSE | train |
| dask/dask | BSD-3-Clause | `6ce3740fa79bfa051ecc74c3641f8c6a46dfdc30` | LICENSE.txt | train |
| pytest-dev/pytest | MIT | `59200df778adf8079df6125d60662908ac0a0646` | LICENSE | train |
| pallets/werkzeug | BSD-3-Clause | `594452f6a4fe4de38a544962fbf04bfc9d37fbc2` | LICENSE.txt | train |
| python-poetry/poetry | MIT | `63bba274f8ea6bc9a064f5328fa6c67bce617cc1` | LICENSE | val |
