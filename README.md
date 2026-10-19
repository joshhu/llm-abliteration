# LLM abliteration on DGX Spark (heretic)

Scripts used to build an **abliterated** (refusal-removed) version of
[`Qwen/Qwen3.8-27B`](https://huggingface.co/Qwen/Qwen3.8-27B).

| | |
|---|---|
| Tool | [p-e-w/heretic](https://github.com/p-e-w/heretic) — PyPI `heretic-llm` **1.4.0** |
| Method | Directional ablation / abliteration ([Arditi et al. 2024](https://arxiv.org/abs/2406.11717)) + Optuna parameter search, pure `transformers` inference (no vLLM needed) |
| Host | `JoshDGX` — NVIDIA DGX Spark, `NVIDIA GB10`, 121 GB unified memory, CUDA 13.0 |
| Image | `vllm/vllm-openai:v0.27.1` (arm64, torch 2.13.0+cu130, **transformers 5.15** — needed for `model_type: qwen3_5`) |
| Input | `Qwen/Qwen3.8-27B` — 52 GB, 18 safetensors shards, `Qwen3_5ForConditionalGeneration`, 64 layers, alternating `linear_attention` / `full_attention` |
| Output | merged bf16 model → `~/heretic/out/Qwen3.8-27B-abliterated` (5 GB shards) |
| Baseline | heretic measured **90/100 refusals** on `mlabonne/harmful_behaviors` (`test[:100]`) before ablation |

## Why heretic (and not the other tools)

* PyPI `abliterated` no longer exists (404), and PyPI `heretic` is an unrelated
  placeholder (version `0.0.0`, no deps). The real package is **`heretic-llm`**.
* heretic's README explicitly supports *hybrid* models such as Qwen3.5, which this
  architecture needs; it also handles multimodal wrappers
  (`AutoModelForImageTextToText`) like `Qwen3_5ForConditionalGeneration`.
* Alternatives considered: `Orion-zhen/abliteration`, `Tsadoq/ErisForge`,
  `p-e-w/heretic`, `huihui-ai/Huihui-Qwen3.8-27B-abliterated` (existing HF version,
  ablates layers 18–51 only, leaves MTP + visual untouched).

## Files

* `config.toml` — heretic config. heretic reads `config.toml` **from the current
  working directory**, so run it from the model's work dir.
* `run_heretic.py` — thin headless shim. Released heretic 1.4.0 asks *interactive*
  `questionary` questions at the end (which trial → what to do → folder path).
  This monkey-patches `prompt_select` / `prompt_path` / `prompt_text` so an
  unattended run: picks the best Pareto-front trial, chooses *Save to local folder*,
  saves to `$HERETIC_SAVE_DIR`, then exits.
* `dgx-run.sh` — the exact container / install / download / run commands (idempotent).

## Run

```bash
# 1) container (GPU + shared HF cache + work dir)
docker run -d --name heretic-qwen38 --gpus all --ipc=host --shm-size=16g \
  -v $HOME/.cache/huggingface:/root/.cache/huggingface \
  -v $HOME/heretic:/heretic \
  --entrypoint bash vllm/vllm-openai:v0.27.1 -c 'sleep infinity'

# 2) tool + weights (52 GB)
docker exec heretic-qwen38 bash -lc 'pip install -U heretic-llm'
docker exec heretic-qwen38 bash -lc 'hf download Qwen/Qwen3.8-27B'

# 3) headless run (config.toml must be in cwd)
docker exec -d -e HERETIC_SAVE_DIR=/heretic/out/Qwen3.8-27B-abliterated \
  heretic-qwen38 bash -lc \
  'cd /heretic/qwen3.8-27b && python3 run_heretic.py Qwen/Qwen3.8-27B < /dev/null > run.log 2>&1'

# 4) watch
tail -c 2000 ~/heretic/qwen3.8-27b/run.log | tr '\r' '\n' | tail -20
```

## Gotchas (all hit in practice)

1. **heretic 1.4.0 config is `extra="forbid"`**: `model_action` and `save_directory`
   exist only on `master`, so putting them in `config.toml` fails with
   *"Extra inputs are not permitted"*. The save path therefore comes from
   `HERETIC_SAVE_DIR` + the shim.
2. **Set `export_strategy = "merge"`** — otherwise heretic prompts for merge-vs-adapter,
   and an adapter export would not give standalone weights.
3. **Headless prompts need `< /dev/null`** and the shim; otherwise the run hangs at the
   end of a many-hour optimization waiting for TTY input.
4. **Auto batch size** settled on `128` at ~70 tok/s aggregate (batch 1 → 4 tok/s,
   32 → 52 tok/s). Expect roughly 5–6 min per trial → 100 trials ≈ 8–10 h on a GB10.
5. **Memory is tight**: the working model lives on the GPU and the KL reference model on
   CPU (~52 GB each) inside 121 GB of unified memory; keep other GPU services off.
6. **Checkpoints are per-trial** (Optuna journal storage → `checkpoints/*.jsonl`), so you
   can kill the run any time and later re-run to export the best trial so far.

## Status

* 2026-10-19 — run started on `JoshDGX`; original model baseline 90/100 refusals;
  100 trials in progress, output pending.
* Next: export best trial → optionally push to `joshhu/Qwen3.8-27B-abliterated` →
  serve (vLLM / llama.cpp) and register in `pi`.

## License / credits

heretic is AGPL-3.0 (p-e-w). `Qwen/Qwen3.8-27B` weights are under Qwen's own license.
This repo only contains the small glue scripts and config used for the run.
