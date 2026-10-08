# Eval harness

Runs the patch an OpenAI-compatible endpoint returns against hidden pytest tests, locally on CPU. Standard library only, plus `pytest` for the tests of each task.

## Task format

Each task is a directory `tasks/<id>/` with:

- `task.json`: required keys `id` (equal to the directory name), `instruction` (the expected behavior, never the answer), `context` (may be `""`) and `edit_file` (path relative to `files/`). Extra keys such as `category` and `source_ids` are allowed and ignored.
- `files/`: a copy of the project. Only the content of `files/<edit_file>` goes into the prompt. `files/tests` must not exist.
- `tests/`: hidden pytest tests (`test_*.py`), never sent to the model. They import the edited module by name (the working directory is the cwd and is on `sys.path`) or read the file at `os.environ["TASK_SRC"]`.
- `oracle.txt`: the reference answer in SEARCH/REPLACE blocks. It must pass the tests, and the tests must fail without it.

Only subdirectories that contain `task.json` count. Two tasks with identical content make `load_tasks` fail.

## Prompt

The system prompt is the one used in training. The user template is the one from the `BENCHMARK REAL` cell of `Simplicio_27B_Training_Colab.ipynb`, the same one that produced the 56/120 run: `Contexto`, `Arquivo`, `Codigo Atual`, `Tarefa`. Training itself used only `Contexto` and `Tarefa`, without `Arquivo` or `Codigo Atual`.

- `--system none` drops the system prompt (useful for the base model).
- `--format-hint` appends an explicit description of the SEARCH/REPLACE format.
- `temperature` is always 0.
- `--chat-template-kwargs` is sent as `chat_template_kwargs` and works on vLLM. Ollama ignores that field; use `--extra-body '{"reasoning_effort": "none"}'` there.

## Answer format

Markers of 4 to 7 characters are accepted (`<<<< SEARCH` / `====` / `>>>> REPLACE`, and the Git/Aider 7-character form). Only the text after the last `</think>` is parsed. Only the first occurrence of each SEARCH text is replaced, blocks apply in order, and an unclosed block is ignored.

## Sandbox

Tests run in a fresh temporary copy with a minimal environment (the API key is not passed), proxies pointing to `127.0.0.1:9`, sockets blocked, and a 60 s timeout. On Linux, `RLIMIT_AS` is 2 GiB, `RLIMIT_CPU` 70 s and `RLIMIT_FSIZE` 64 MiB. This is best effort, not security isolation (see "Limites do sandbox" below). `HOME` is changed, so pytest must be installed outside the user site.

### Verdict

A task passes only if pytest exits with 0 **and** the JUnit report that pytest writes at the end of its session (`--junitxml`, in a temporary directory the harness creates and reads) exists, can be parsed, has at least one executed case, and has no failed, errored or skipped case. The exit code alone is not evidence: the model's code is imported in the same process as pytest, so a module that runs `import os; os._exit(0)` ends the process with exit code 0 before any test runs. pytest writes the report only after collection and all tests finished, so that process leaves no report and the task is a FAIL (`report.reason` is `report_missing`). The `report` field of each JSONL row has `ok`, `reason`, `tests`, `failures`, `errors` and `skipped`.

## Limites do sandbox

What the sandbox does not do. Read this before trusting a number, and before running a model you do not control.

- The network is blocked only by monkeypatching `socket` inside the same process that runs the model's code (plus proxy variables pointing to `127.0.0.1:9`). Code in the patched module can undo the patch or go around it: for example `_socket.socket().connect(...)` is not patched and connects (checked), and a spawned process is not patched either. This is **not a security boundary**.
- There is no file isolation. The tests run as the same user, with the temporary copy as the working directory, but nothing stops the code from reading or writing other paths that user can reach. Only `HOME` and the environment are replaced, so the API key is not passed in the environment.
- The JUnit report protects against lazy or accidental early exits (`os._exit(0)`, crashes, a module skipped on import). It does not protect against code written to forge the report: that code runs in the same process, has no file isolation, and can write a well-formed XML file itself. The harness measures patches that are honestly trying to pass, not hostile ones.
- On Windows, `proc.kill()` on timeout ends only the direct child. Processes it started can survive the kill. On POSIX the whole process group is killed. The `RLIMIT_*` limits are Linux only.

To run an untrusted model, put the whole harness in a container or VM without network and without credentials.

## Outputs

For each run, `results/<run>.jsonl` has one line per task (`task`, `task_sha256`, `error`, `latency_s`, `raw_output`, `reasoning`, `finish_reason`, `prompt_tokens`, `completion_tokens`, `applied`, `apply_error`, `blocks`, `passed`, `returncode`, `timed_out`, `stdout_tail`, `report`). A task whose request fails, or whose answer is not a chat completion (`OSError`, `HTTPException`, `KeyError`, `IndexError`, `ValueError`, `AttributeError`, `TypeError`), is written as a failed row with `error` set and counts in `errors`. A tasks directory with no task is an error (exit code 2) and writes nothing. `results/<run>.summary.json` has the run configuration, counts, `pass_rate`, the 95% Wilson interval (`wilson_95`), `prompt_sha256`, `git_commit` and timestamps. `results/` is git-ignored.

## Tasks

The 40 eval tasks (`tasks/cat1_*` to `tasks/cat4_*`, 10 per category) come from the 40 unique tasks of [`data/unseen_eval_120.json`](https://github.com/simpletibr/simplicio-27b/blob/b8567df/data/unseen_eval_120.json), which lists each of them 3 times. `source_ids` keeps the 3 original ids and `source_prompt` keeps the old prompt. `instruction` is the new statement: it describes the behavior and never the answer. `context` is empty. Each task has real code in `files/`, hidden pytest tests in `tests/` that run the patched module (a few use the AST or a spy where the behavior cannot be observed from the outside) and an `oracle.txt` that passes them. The 3 `ex_*` format examples live in `tests/fixtures/harness_tasks/`, so that `tasks/` holds only the 40 eval tasks.

The tests need `pydantic`, `fastapi`, `pandas` and `SQLAlchemy`, all pinned in `requirements.txt`, and the harness runs them with the same Python that runs `harness.py`. `tasks/ruff.toml` excludes `files/` from ruff, because the stubs there keep unused imports on purpose.

Verification: `harness.py --oracle` (every task must print `OK`) and `decontam_tasks.py`, which writes `tasks/decontamination.json` and exits with 1 if instruction, code and oracle of a task share a 13-gram with `data/*.jsonl`.

## Commands

```bash
python3 -m pip install -r benchmarks/harness/requirements.txt
python3 benchmarks/harness/harness.py --oracle   # the 40 eval tasks
python3 benchmarks/harness/harness.py --tasks tests/fixtures/harness_tasks --oracle   # the 3 format examples
python3 benchmarks/harness/decontam_tasks.py
python3 benchmarks/harness/harness.py --base-url http://127.0.0.1:8000/v1 --model simplicio-27b --chat-template-kwargs '{"enable_thinking": false}'
python3 benchmarks/harness/harness.py --base-url http://127.0.0.1:11434/v1 --model wesleysimplicio/simplicio-27b --extra-body '{"reasoning_effort": "none"}'
```
