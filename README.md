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

## SFT facts (from notebook 00)

**Dataset splits**: train 3,954 · validation 500 · test 500 · test_unseen_phrasing 394. The 20 human-written questions are kept outside the dataset.

**Columns**: `id`, `category` (multi 1085 · single 725 · grid 638 · cancel 624 · refuse 485 · no_action 292 · off_topic 105 in train), `held_out_phrasing`, `messages` (system/user/assistant), `note`, `trace`.

**Output JSON**:
```json
{"intent": "set_constraints|no_action|off_topic|refuse",
 "constraints": [{"workload_id": "...", "field": "flex_tier|earliest_start|allow_pause",
                  "value": "...", "action": "set|clear", "valid_until": "YYYY-MM-DD HH:MM|null"}],
 "explanation": "..."}
```

**Refuse rules (whole note refused, `constraints: []`)**: violates a customer contract · adjusts `emergency-inference` · skips safety checks or human confirmation.

**LoRA config**: r=16, alpha=32, dropout=0.05, target_modules = q/k/v/o/gate/up/down_proj, bias none.

**Data rule for DPO**: preference pairs are built from `train` only. `validation` may be used for DPO eval loss; `test`, `test_unseen_phrasing` and the 20 human questions are never used to build pairs.

## Roadmap

| # | Notebook | What it does | GPU |
|---|---|---|---|
| 00 | `00_inspect_sft_assets.ipynb` | Print SFT dataset schema, samples, SLA-refusal examples, LoRA config, eval splits | None |
| 01a | `01a_probe_sft_errors.ipynb` | Run the SFT model on 500 train prompts (greedy + sampling) and measure the error rate per category | T4 |
| 01b | `01b_build_preference_pairs.ipynb` | Build `{prompt, chosen, rejected}` pairs: on-policy failures first, rule-based near-miss corruptions to fill gaps (tagged by `source`) | None |
| 02 | `02_dpo_train.ipynb` | DPOTrainer with the SFT adapter as policy start and SFT model as reference | T4 |
| 03 | `03_eval_before_after.ipynb` | Same three eval sets as SFT + GSM8K regression; SFT vs DPO | T4 |

## Run on Kaggle

1. Kaggle → Create → Import Notebook → GitHub tab → `YOYU001/grid-dispatch-dpo`
2. Add-ons → Secrets → `HF_TOKEN` (read access for 00, write access for 02)
3. Run all

## Log

| Date | Run | Notes |
|---|---|---|
| 2026-11-19 | 00 | Inspected SFT assets; fixed keyword search to skip the system prompt |
