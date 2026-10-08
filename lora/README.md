# lora/

Layout of the LoRA adapter. The weights are not in git (`.gitignore` skips `*.safetensors`); this folder only says where they go.

```text
lora/
  adapter_config.json          # base_model_name_or_path points at the 4-bit base used for training
  adapter_model.safetensors    # the LoRA tensors
```

On Hugging Face the adapter lives in `lora/` of `wesleysimplicio/Simplicio-27B`, not at the repository root. A root `adapter_config.json` makes the Hub list the repo as an adapter and makes `transformers` swap the merged BF16 weights for the 4-bit base plus adapter.

To get the adapter, download `lora/*` from the Hub (for example `snapshot_download(..., allow_patterns=["lora/*"])`) and put the two files here.

`scripts/hf_publish.py` is the only way these files reach the Hub. It uploads only `lora/adapter_config.json` and `lora/adapter_model.safetensors` from this folder; this README and any other file here stay local. When the Hub still has the adapter at the root, the script moves it into `lora/` inside the same commit. The default run is a dry run; see the docstring of the script for `--publish`.
