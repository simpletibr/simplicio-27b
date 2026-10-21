"""F2b gate GT (issue #75): does Qwen3.8-27B with thinking beat the best one-pass decision model?

Modules, in the order of a run:
  `sample`   deterministic sample of the public knowledge rows + the private pack (question text, gold, the
             decision models' per-option probabilities) + `gt-sample-manifest.json` (ids and hashes only);
  `prompt`   what the thinker is asked, `answers` how its reply is read;
  `thinker`  the runner: an OpenAI-compatible endpoint or transformers with NF4, resumable row by row;
  `compare`  accuracy, chance-corrected skill, paired bootstrap, the GT verdict line, the offline gate curve;
  `trace`    the `Raciocínio:` block that the hybrid appends to a decider's state (a proposal);
  `serve`    the `vllm serve` command for the thinker on a rented or own GPU (not the Kaggle T4);
  `common`   constants and file helpers; `mutation_check` plants one-line bugs and expects the tests to fail.

The sample is evaluation data: nothing made from it goes into a training corpus (ADR 0004, item 6).
Standard library only, except `thinker --backend transformers` (torch, transformers, bitsandbytes).
"""
__version__ = "1"
