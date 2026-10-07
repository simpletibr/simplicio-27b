# Local gate. There is no CI: run `make check` before every push.
# Recipes use the one-line `target: ; command` form, so no TAB is needed.
.PHONY: check test lint shellcheck phplint vllm-args

SH_FILES := $(shell git ls-files '*.sh') .githooks/pre-push
PHP_FILES := $(shell git ls-files '*.php')

check: test lint shellcheck phplint vllm-args

test: ; python3 -m unittest discover -s tests
lint: ; ruff check --select F,E9 .
shellcheck: ; shellcheck $(SH_FILES)
phplint: ; @for f in $(PHP_FILES); do php -l "$$f" || exit 1; done
vllm-args: ; python3 scripts/check_vllm_args.py
