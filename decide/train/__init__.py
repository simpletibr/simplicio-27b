"""Training code for Simplicio-Decide-27B (issue #69, F4).

Distillation of a teacher ensemble into a pplx-decider-v1.2 style joint-readout model (LoRA or full fine-tuning),
seed soup, and the builder that joins corpus, teacher labels and a target combiner.  Parts adapted from
Perplexity's Apache-2.0 autojev code (see NOTICE).  The modules that need torch import it lazily; ``codes``,
``permute``, ``targets``, ``schedule`` and ``protocol`` are pure Python.
"""
