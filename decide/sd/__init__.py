"""simplicio-decide: Kaggle edition.

Turns Simplicio-27B into a Jev-style decision model (merge-on-load of Qwen3.8-27B decision
backbones + the Simplicio LoRA delta), evaluates it in NF4 on a stratified public sample,
calibrates it and tests option-order TTA.  Heavy dependencies (torch, transformers,
bitsandbytes) are imported lazily so the pure-math/state modules run anywhere.
"""

__version__ = "0.1.0"
