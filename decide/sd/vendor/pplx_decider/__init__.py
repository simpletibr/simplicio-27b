# Vendored from https://huggingface.co/perplexity-ai/pplx-decider-v1.2 (source/src/pplx_decider/__init__.py),
# upstream sha256 48c26c24432476f2e4a43486d60e24ee0a6e124093a38fb8803686e99fecb5b3.
# Apache-2.0, (c) Perplexity AI; see LICENSE and NOTICE in this directory.
# Modifications by simplicio-decide: the eager imports were replaced by lazy ones (module __getattr__), so that
# `pplx_decider.protocol` can be imported without transformers; the public names are unchanged.
"""Inference for the pplx joint decider (protocol v3: one sequence per request, one readout per question)."""

from __future__ import annotations

import importlib

_EXPORTS = {
    "DEFAULT_MAX_LENGTH": "model",
    "Decider": "model",
    "FastOptions": "model",
    "MAX_OPTIONS": "protocol",
    "LengthError": "protocol",
    "RequestError": "protocol",
    "answer": "protocol",
    "options": "protocol",
    "Answer": "types",
    "Question": "types",
    "Request": "types",
    "Response": "types",
    "Usage": "types",
}

__all__ = sorted(_EXPORTS)
__version__ = "0.3.0"


def __getattr__(name: str):
    module = _EXPORTS.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(importlib.import_module(f".{module}", __name__), name)
