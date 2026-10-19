# 在 DGX Spark 上用 heretic 做 abliteration

把 [`Qwen/Qwen3.8-27B`](https://huggingface.co/Qwen/Qwen3.8-27B) 做成一個
**abliterated（移除拒答傾向）** 版本所用的程式、設定與踩坑記錄。

| 項目 | 內容 |
|---|---|
| 工具 | [p-e-w/heretic](https://github.com/p-e-w/heretic) — PyPI 套件名 `heretic-llm`，版本 **1.4.0** |
| 方法 | 方向性消融 / abliteration（[Arditi et al. 2024](https://arxiv.org/abs/2406.11717)）＋ Optuna 參數搜尋；推理只用 `transformers`，**不需要 vLLM** |
| 主機 | `JoshDGX` — NVIDIA DGX Spark、`NVIDIA GB10`、121 GB 統一記憶體、CUDA 13.0 |
| 映像檔 | `vllm/vllm-openai:v0.27.1`（arm64、torch 2.13.0+cu130、**transformers 5.15**，因為 `model_type: qwen3_5` 需要新版 transformers）|
| 輸入模型 | `Qwen/Qwen3.8-27B` — 52 GB、18 個 safetensors 分片、`Qwen3_5ForConditionalGeneration`、64 層、`linear_attention` 與 `full_attention` 交錯 |
| 輸出 | 合併後的 bf16 模型 → `~/heretic/out/Qwen3.8-27B-abliterated`（每個分片 5 GB）|
| 消融前基線 | heretic 實測 `mlabonne/harmful_behaviors`（`test[:100]`）原始模型 **90/100 拒答** |

## 為什麼選 heretic

* PyPI 上的 `abliterated` 已經不存在（404）；PyPI 上的 `heretic` 是不相干的佔位包
  （版本 `0.0.0`、沒有相依套件）。真正的套件名是 **`heretic-llm`**。
* heretic 的 README 明確提到支援 Qwen3.5 這類 **hybrid** 架構，也正是
  `qwen3_5`（交錯線性注意力／全域注意力）需要的；它也處理多模態包裝類別
  （`AutoModelForImageTextToText`），所以 `Qwen3_5ForConditionalGeneration` 可直接載入。
* 其他考慮過的方案：`Orion-zhen/abliteration`、`Tsadoq/ErisForge`，以及現成的
  `huihui-ai/Huihui-Qwen3.8-27B-abliterated`（只消融 18–51 層，MTP 與視覺模組不處理）。

## 檔案說明

* `config.toml` — heretic 設定檔。heretic 是**從目前工作目錄**讀 `config.toml`，
  所以一定要在模型的工作目錄下執行。
* `run_heretic.py` — 無頭（headless）執行用的小 shim。正式版 heretic 1.4.0 在最佳化結束後
  會用 `questionary` 丢出互動問題（要選哪個 trial → 要對模型做什麼 → 存到哪個資料夾）。
  這裡把 `prompt_select` / `prompt_path` / `prompt_text` 換掉，讓無人值守的執行自動：
  選 Pareto 前端最好的 trial → 選「存到本機資料夾」→ 存到 `$HERETIC_SAVE_DIR` → 結束。
* `dgx-run.sh` — 封裝 `start / deps / download / run / status / stop` 六個子命令，
  建立 container、裝套件、下載權重、啟動、看進度都靠它。

## 執行流程

```bash
# 1) 開 container（要 GPU、共用 HF cache、掛載工作目錄）
docker run -d --name heretic-qwen38 --gpus all --ipc=host --shm-size=16g \
  -v $HOME/.cache/huggingface:/root/.cache/huggingface \
  -v $HOME/heretic:/heretic \
  --entrypoint bash vllm/vllm-openai:v0.27.1 -c 'sleep infinity'

# 2) 裝工具 + 下載權重（52 GB）
docker exec heretic-qwen38 bash -lc 'pip install -U heretic-llm'
docker exec heretic-qwen38 bash -lc 'hf download Qwen/Qwen3.8-27B'

# 3) 無頭執行（config.toml 必須在 cwd）
docker exec -d -e HERETIC_SAVE_DIR=/heretic/out/Qwen3.8-27B-abliterated \
  heretic-qwen38 bash -lc \
  'cd /heretic/qwen3.8-27b && python3 run_heretic.py Qwen/Qwen3.8-27B < /dev/null > run.log 2>&1'

# 4) 看進度
./dgx-run.sh status
# 或：tail -c 2000 ~/heretic/qwen3.8-27b/run.log | tr '\r' '\n' | tail -20
```

## 實際跑出來的搜尋參數（範例）

Optuna 每個 trial 會給一組消融核參數，長這樣：

```text
* direction_index = per layer
  * attn.o_proj.max_weight = 1.01
  * attn.o_proj.max_weight_position = 40.26
  * attn.o_proj.min_weight = 0.69
  * attn.o_proj.min_weight_distance = 17.20
  * mlp.down_proj.max_weight = 0.89
  * mlp.down_proj.max_weight_position = 50.28
  * mlp.down_proj.min_weight = 0.03
  * mlp.down_proj.min_weight_distance = 34.46
```

評估端則同時最小化兩件事：拒答率（越低越好）與相對原模型的 KL 散度（越低越好）。

## 踩過的坑（全部實際遇過）

1. **heretic 1.4.0 的設定是 `extra="forbid"`**：`model_action` 與 `save_directory`
   只有 `master` 分支有，寫進正式版 `config.toml` 會報
   *"Extra inputs are not permitted"*。所以儲存路徑改由 `HERETIC_SAVE_DIR` ＋ shim 提供。
2. **要設 `export_strategy = "merge"`**：否則 heretic 會再問一次「merge 還是 adapter」，
   而 adapter 輸出不是完整權重。
3. **無人執行一定要 `< /dev/null` 並搭配 shim**：不然幾個小時的搜尋跑完後，
   程式會卡在等 TTY 輸入而不會自己結束。
4. **自動 batch size 落在 128**，約 70 tok/s（batch 1 只有 4 tok/s、batch 32 約 52 tok/s）。
   在 GB10 上大約每 trial 5–6 分鐘，100 trials ≈ 8–10 小時。
5. **記憶體很緊**：工作模型放 GPU、KL 參考模型放 CPU（各約 52 GB），
   全在 121 GB 統一記憶體裡，跑的時候其他 GPU 服務先關掉。
6. **checkpoint 是逐 trial 寫入**（Optuna journal storage → `checkpoints/*.jsonl`），
   所以隨時可以中止；之後重跑選「顯示上次結果」就能把當下最好的 trial 匯出。

## 目前進度

* 2026-10-19 — 在 `JoshDGX` 開跑；原模型基線 90/100 拒答；100 trials 進行中，模型尚未匯出。
* 後續：匯出最好的 trial → 上傳 `joshhu/Qwen3.8-27B-abliterated` → 用 vLLM 或 llama.cpp 起服務，
  並加進 `pi` 的模型選單。

## 授權／備註

heretic 為 AGPL-3.0（p-e-w）；`Qwen/Qwen3.8-27B` 權重採 Qwen 自有授權。
本 repo 只放這次執行用到的 glue 程式與設定，不含模型權重與大份 log。
