# Qwen3.8-27B Abliteration

使用 [heretic](https://github.com/p-e-w/heretic) 對 [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) 進行方向性消融（abliteration）的執行範例。專案提供設定檔、無互動執行腳本與 Optuna 試驗結果摘要工具；**不包含模型權重**。

## 環境需求

- 支援 CUDA 的 NVIDIA GPU；建議使用 NVIDIA DGX Spark 或具備足夠記憶體的主機。
- Docker、NVIDIA Container Toolkit，以及可使用 GPU 的容器執行環境。
- 可存放模型權重、Hugging Face 快取和輸出模型的磁碟空間。
- 預設容器映像：`vllm/vllm-openai:v0.27.1`（此流程透過映像提供的 PyTorch 與 Transformers 執行 heretic，不使用 vLLM 進行推論）。

## 安裝與執行

在 **DGX 主機**上執行以下指令。腳本預設將工作目錄放在 `~/heretic/qwen3.8-27b`，模型輸出放在 `~/heretic/out/Qwen3.8-27B-abliterated`。

```bash
mkdir -p ~/heretic
git clone https://github.com/joshhu/llm-abliteration.git ~/heretic/qwen3.8-27b
cd ~/heretic/qwen3.8-27b

./dgx-run.sh start      # 啟動 GPU 容器
./dgx-run.sh deps       # 安裝 heretic-llm
./dgx-run.sh download   # 下載 Qwen/Qwen3.8-27B
./dgx-run.sh run        # 在背景執行
./dgx-run.sh status     # 查看執行記錄與輸出目錄
```

`run` 從工作目錄讀取 `config.toml`。預設設定進行 100 次試驗、將最佳化結果合併為完整權重，並以每個檔案最多 5 GB 的方式儲存。執行結束後，`run_heretic.py` 會自動選取 Pareto 前緣中**拒答數最低**的試驗，將模型存入 `$HERETIC_SAVE_DIR`（預設為上述輸出目錄）。可在執行前修改 `config.toml` 調整設定。

如需停止並釋放 GPU 記憶體：

```bash
./dgx-run.sh stop
```

`stop` 會移除容器；掛載在主機上的 Hugging Face 快取、記錄及 Optuna checkpoint 不會因此刪除。重新執行 `start` 和 `deps` 可建立新容器。

## 產生試驗摘要

執行中的試驗會寫入 `checkpoints/*.jsonl`。摘要工具會讀取 Optuna journal，列出已完成試驗的 Pareto 前緣及所選試驗的參數：

```bash
./dgx-run.sh summary
# 輸出：results/best-trial.md
```

摘要需要在已安裝 heretic 相依套件的容器內產生；此檔案為本機產物，不納入版本控制。如需在整次執行結束後自動產生摘要，可另執行 `./dgx-run.sh watch`。

## 專案檔案

| 檔案 | 說明 |
|---|---|
| `config.toml` | heretic 搜尋與匯出設定 |
| `run_heretic.py` | 回答 heretic 的互動式選單，支援無人值守執行與儲存模型 |
| `dgx-run.sh` | 容器、下載、執行及摘要管理指令 |
| `tools/summarize.py` | 從 Optuna journal 產生 Markdown 試驗摘要 |

本專案程式依賴 [heretic](https://github.com/p-e-w/heretic)；使用模型權重前，請確認 [Qwen/Qwen3.8-27B 的授權條款](https://huggingface.co/Qwen/Qwen3.8-27B)。
