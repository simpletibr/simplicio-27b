# lora/

Layout of the LoRA adapter. The weights are not in git (`.gitignore` skips `*.safetensors`); this folder only says where they go.

```text
lora/
  adapter_config.json          # base_model_name_or_path points at the 4-bit base used for training
  adapter_model.safetensors    # the LoRA tensors
```

On Hugging Face the adapter lives in `lora/` of `wesleysimplicio/Simplicio-27B`, not at the repository root. A root `adapter_config.json` makes the Hub list the repo as an adapter and makes `transformers` swap the merged BF16 weights for the 4-bit base plus adapter.

To get the adapter, put the two files here. While the Hub still has them at the repository root, download them from there, for example `hf download wesleysimplicio/Simplicio-27B adapter_config.json adapter_model.safetensors --local-dir lora`; later, download `lora/*` from the Hub instead.

`scripts/hf_publish.py` is the only way these files reach the Hub. It uploads only `lora/adapter_config.json` and `lora/adapter_model.safetensors` from this folder; this README and any other file here stay local. The script never deletes or copies anything on the Hub: a file on the Hub that the allowlist does not cover shows up in the plan as a warning ("órfão no Hub, não removido"). So the move out of the root takes three steps: put the adapter here, run the script with `--publish` (it uploads `lora/`), and only after checking `lora/` on the Hub, delete `adapter_config.json` and `adapter_model.safetensors` at the root by hand. The default run is a dry run; see the docstring of the script for `--publish`.
