# grid-dispatch-dpo

Preference alignment (DPO) on top of **grid-dispatch-A**, a Qwen3-4B-Instruct-2507 model QLoRA-SFT'd to turn Chinese grid-dispatch operator notes into constraint JSON and refuse SLA-violating instructions. It does not make dispatch decisions; outputs still need safety validation and human approval.

Part of Hugging Face smol-course, Unit 2 (DPO). All training runs on Kaggle (T4).

## Upstream assets

| Asset | Hub ID |
|---|---|
| Base model | `Qwen/Qwen3-4B-Instruct-2507` |
| SFT adapter (QLoRA) | `yoyu001/grid-dispatch-A-instruct-qlora-v1` |
| SFT merged model | `yoyu001/grid-dispatch-A-instruct-merged-v1` |
| SFT dataset (synthetic, private) | `yoyu001/grid-dispatch-notes-zh` |
| GSM8K regression check | `yoyu001/details_yoyu001__grid-dispatch-A-instruct-merged-v1` |

SFT results (exact match, base → SFT): test 19.6% → 98.6%, unseen phrasings 19.8% → 81.2%, 20 human-written questions 30% → 75%. GSM8K 0.670 → 0.710.

The gap between the test set and unseen/human phrasings is what DPO targets.

## Roadmap

| # | Notebook | What it does | GPU |
|---|---|---|---|
| 00 | `00_inspect_sft_assets.ipynb` | Print SFT dataset schema, samples, SLA-refusal examples, LoRA config, eval splits | None |
| 01 | `01_build_preference_pairs.ipynb` | Build `{prompt, chosen, rejected}` pairs: rule-based corruptions + on-policy failures from the SFT model | T4 |
| 02 | `02_dpo_train.ipynb` | DPOTrainer with the SFT adapter as policy start and SFT model as reference | T4 |
| 03 | `03_eval_before_after.ipynb` | Same three eval sets as SFT + GSM8K regression; SFT vs DPO | T4 |

## Run on Kaggle

1. Kaggle → Create → Import Notebook → GitHub tab → `YOYU001/grid-dispatch-dpo`
2. Add-ons → Secrets → `HF_TOKEN` (read access for 00, write access for 02)
3. Run all

## Log

| Date | Run | Notes |
|---|---|---|
| 2026-11-19 | 00 | Created inspection notebook |
