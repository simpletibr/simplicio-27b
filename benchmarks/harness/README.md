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

Tests run in a fresh temporary copy with a minimal environment (the API key is not passed), proxies pointing to `127.0.0.1:9`, sockets blocked, and a 60 s timeout. On Linux, `RLIMIT_AS` is 2 GiB, `RLIMIT_CPU` 70 s and `RLIMIT_FSIZE` 64 MiB. This is best effort, not security isolation. `HOME` is changed, so pytest must be installed outside the user site.

## Outputs

For each run, `results/<run>.jsonl` has one line per task (`task`, `task_sha256`, `error`, `latency_s`, `raw_output`, `reasoning`, `finish_reason`, `prompt_tokens`, `completion_tokens`, `applied`, `apply_error`, `blocks`, `passed`, `returncode`, `timed_out`, `stdout_tail`). `results/<run>.summary.json` has the run configuration, counts, `pass_rate`, the 95% Wilson interval (`wilson_95`), `prompt_sha256`, `git_commit` and timestamps. `results/` is git-ignored.

## Commands

```bash
python3 -m pip install -r benchmarks/harness/requirements.txt
python3 benchmarks/harness/harness.py --oracle
python3 benchmarks/harness/harness.py --base-url http://127.0.0.1:8000/v1 --model simplicio-27b --chat-template-kwargs '{"enable_thinking": false}'
python3 benchmarks/harness/harness.py --base-url http://127.0.0.1:11434/v1 --model wesleysimplicio/simplicio-27b --extra-body '{"reasoning_effort": "none"}'
```
