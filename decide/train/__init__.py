"""Training code for Simplicio-Decide-27B (issue #69, F4).

Fine-tuning of a pplx-decider-v1.2 style joint-readout model (LoRA or full) on the gold labels mixed with the labels of ONE
teacher, the starting model itself (self-distillation, ADR 0004, decisions 1 and 3: no ensemble; several teachers only
through an explicit combiner, for experiments), seed soup, temperature fit, and the builder that joins the decontaminated
corpus, the teacher labels, the thinker traces and the combiner into the targets dataset.  Parts adapted from Perplexity's
Apache-2.0 autojev code (see NOTICE).  The modules that need torch import it lazily; ``codes``, ``permute``, ``targets``,
``schedule`` and ``protocol`` are pure Python.
"""
