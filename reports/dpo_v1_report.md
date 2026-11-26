# Grid-Dispatch-A：QLoRA-SFT + DPO v1 實驗報告

2026-11-26 · 劉宥羽（Yoyo）

> PDF 版（含圖表）：[dpo_v1_report.pdf](dpo_v1_report.pdf)

## 摘要

DPO v1 讓模型在**沒看過的說法**上進步了 3.5 個百分點（unseen 80.5% → 84.0%，p = 0.044），但在**和訓練同分布的說法**上退步了 2.0 個百分點（test 98.4% → 96.4%，p = 0.021）。這是一個典型的對齊取捨：泛化變好，原本會的能力付出一點代價。

- **任務**：把中文電網調度備註轉成約束 JSON，並拒絕違反 SLA 的要求。
- **起點**：Qwen3-4B-Instruct-2507 加上 QLoRA-SFT adapter（grid-dispatch-A）。
- **做法**：在 SFT adapter 上繼續做 DPO。偏好資料共 1,692 組訓練對、201 組驗證對，來源是模型自己的錯誤加上 8 種規則產生的「差一點就對」的錯誤答案。
- **算力**：全程只用 Kaggle 免費 T4（16 GB），推論用 vLLM 0.34.0，訓練用 TRL 1.18.2。
- **評估**：三份互相隔離的考卷（test 500 題、unseen 394 題、真人手寫 20 題），用 exact McNemar 檢定判斷差異是否顯著。
- **主要發現**：DPO 減少了漏拒（4 → 2），沒有增加誤拒；但出現「多寫限制」和「時間往半夜偏」兩種新錯誤，很可能來自偏好資料只懲罰「少寫」、沒有懲罰「多寫」。

程式碼：[YOYU001/grid-dispatch-dpo](https://github.com/YOYU001/grid-dispatch-dpo)。模型：yoyu001/grid-dispatch-A-instruct-qlora-dpo-v1（私人）。

## 背景與任務定義

模型的工作是把調度員寫的中文備註，翻譯成 AI 資料中心排程器看得懂的約束 JSON。它不做調度決策，輸出仍需要安全驗證與人工確認。

舉例來說，備註「今晚客服不能降速」應該輸出一條 `flex_tier` 約束，有效期限到今天 23:59。備註「把 emergency-inference 先暫停」則必須整則拒絕。

**輸出格式**

```json
{"intent": "set_constraints|no_action|off_topic|refuse",
 "constraints": [{"workload_id": "...", "field": "flex_tier|earliest_start|allow_pause",
                  "value": "...", "action": "set|clear", "valid_until": "YYYY-MM-DD HH:MM|null"}],
 "explanation": "..."}
```

**必須拒絕的三種情況**（整則拒絕，`constraints` 為空）

- 違反客戶合約（SLA）
- 調整 `emergency-inference`
- 跳過安全檢查或人工確認

**評分方式**：`intent` 加上整組 `constraints` 完全正確才算對（exact match），約束的順序不影響。`explanation` 不計分。答錯時會標上錯誤類型，例如漏拒（missed_refuse）、誤拒（false_refuse）、少寫約束（missing_constraint）、多寫約束（extra_constraint）、時間錯誤（wrong_valid_until）。

## SFT 基線

SFT 已經把和訓練同分布的說法練到 98% 以上，但換個說法就掉到約 80%，真人手寫題只有 75%。這個落差就是 DPO 要處理的目標。

**模型與 LoRA 設定**

- 底座：`Qwen/Qwen3-4B-Instruct-2507`，以 4-bit NF4（double quant，compute dtype fp16）載入，也就是 QLoRA。
- LoRA：r = 16、alpha = 32、dropout = 0.05，掛在全部 7 個投影層（q/k/v/o/gate/up/down_proj），bias 不訓練。
- 優化器：`paged_adamw_8bit`，`max_grad_norm` 0.5（DPO 沿用同一組設定）。
- adapter 不合併回底座，所以 base、SFT、DPO 三個版本可以在同一個底座上隨時切換。

**資料切分**（合成資料 `yoyu001/grid-dispatch-notes-zh`，私人）

| 切分 | 題數 | 用途 |
| --- | --- | --- |
| train | 3,954 | SFT 訓練；DPO 偏好資料也只從這裡取 |
| validation | 500 | SFT 驗證；DPO 只拿來算 eval loss |
| test | 500 | 和訓練同分布的說法（41% 只換了 workload 名稱） |
| test_unseen_phrasing | 394 | 刻意保留、訓練時沒出現過的說法 |
| 真人手寫 | 20 | 放在資料集之外的私人檔案 |

train 的題型分布：multi 1,085、single 725、grid 638、cancel 624、refuse 485、no_action 292、off_topic 105。

**基線成績**（exact match）

| 考卷 | Base | SFT（model card） | SFT（03 重跑） |
| --- | --- | --- | --- |
| test（500） | 19.6% | 98.6% | 98.4% |
| unseen（394） | 19.8% | 81.2% | 80.5% |
| 真人手寫（20） | 30% | 75% | 75% |

03 重跑的 SFT 成績和 model card 只差 0.2–0.7 個百分點，代表評估流程本身可信，後面 DPO 的比較才有意義。另外 SFT 沒有傷到一般推理能力：GSM8K 從 0.670 升到 0.710。

## 偏好資料設計（01a / 01b）

SFT 模型在 train 題上幾乎不犯錯，所以絕大多數的「錯誤答案」必須用規則製造。最後做出 1,692 組訓練對、201 組驗證對，推到 `yoyu001/grid-dispatch-dpo-pairs`（私人）。

### 01a：先找出 SFT 模型會錯在哪

從 train 分層抽 530 題，每題跑 1 次 greedy 加 4 次 sampling（temperature 0.8）。

- greedy 正確率 99.6%，sampling 99.4%；530 題裡只有 6 題曾經答錯。
- 沒有 JSON 格式錯誤、沒有被截斷、沒有拒絕錯誤。
- 錯誤集中在四類：過半夜的時間進位（包括寫成 `24:00`）、「今晚」和「今天」的有效期限混淆、彈性程度詞（對應 `flex_tier`）、`clear` 和 `set` 搞反。
- 速度：vLLM 約 364–437 tok/s，transformers 約 73 tok/s，快了 5–6 倍。之後所有推論都改用 vLLM。

這個結果代表：只靠模型自己的錯誤（on-policy）只能湊出 6 組資料，其餘要靠「差一點就對」的規則式錯誤來補。

### 01b：做成 {prompt, chosen, rejected}

每組資料的 chosen 是標準答案，規則產生的 rejected 只改一個地方。例如備註「今晚客服不能降速」，chosen 的 `valid_until` 是今天 23:59，rejected 可能只把時間改成別的值，其他欄位完全相同。這樣模型才會把注意力放在「哪個細節錯了」。例外：on_policy 和 refuse_partial 的 rejected 是模型自己寫的，refuse_comply 則是整段照做違規要求，都不只差一個地方。

| 規則 | 實際數量 | 製造的錯誤 |
| --- | --- | --- |
| on_policy | 6 | 01a 抽考中 SFT 模型實際答錯的答案 |
| time_rollover | 250 | 過半夜的時間進位錯誤 |
| tonight_swap | 200 | 「今晚」和「今天」的有效期限對調 |
| flex_shift | 200 | `flex_tier` 等級偏一級 |
| time_shift | 150 | 時間值偏移 |
| cancel_misread | 150 | 取消類備註把 `clear` 讀成 `set` |
| missing_constraint | 150 | 少寫一條約束 |
| false_refuse | 150 | 應該執行卻拒絕 |
| intent_confusion | 80 | `intent` 分類錯誤 |
| refuse_comply | 56 | 應該拒絕卻整段照做（沒有上限，train 裡純違規的 refuse 題全收） |
| refuse_partial | 300 | 把違規部分拿掉後照做其他部分（由 SFT 模型生成） |
| **合計** | **1,692** | 有上限的規則全部達到上限 |

- 每個 prompt 做 2 組資料；eval 從 validation 取 12%。
- 每一個 rejected 都用評分程式確認「真的是錯的」，避免不小心懲罰到正確答案。
- refuse_partial 的 300 個模型答案全部保留，沒有丟棄。
- 最長一組資料 654 tokens。
- 拒絕類佔全部資料的 21%，比 SFT 資料的 12% 高，所以 03 要特別觀察會不會變成什麼都拒絕。

**設計上的缺口**：有 150 組懲罰「少寫約束」，卻沒有任何一組懲罰「多寫約束」。這是後面錯誤分析的主要線索。

## DPO 訓練

在 4-bit Qwen3-4B 上接著訓練 SFT adapter，共 106 步、102 分鐘（一張 T4）。訓練本身很順，但曲線顯示題目太簡單，而且模型對正確答案的信心下降了。

### 設定與用途

| 參數 | 值 | 白話用途 |
| --- | --- | --- |
| `beta` | 0.1 | 繩子的鬆緊：越小越允許模型離開 ref 模型。 0.1 是常見預設 |
| `learning_rate` | 1e-5 | 是 SFT 學習率（1e-4）的十分之一，只做微調，避免把 SFT 學到的東西洗掉 |
| `num_train_epochs` | 1 | 每組資料只看一次 |
| batch × grad_accum | 2 × 8 | 每次更新等於看 16 組資料；1,692 ÷ 16 ≈ 106 步 |
| `max_length` | 1536 | 最長一組只有 654 tokens，不會被截斷 |
| `fp16` | True | T4 不支援 bf16，只能用 fp16 混合精度 |
| `optim` | paged_adamw_8bit | 8-bit 優化器省記憶體，和 SFT 一樣 |
| `max_grad_norm` | 0.5 | 每步更新力道的上限，和 SFT 一樣 |
| scheduler / warmup | cosine / 0.1 | 前 10% 步數慢慢加速，之後像煞車一樣降到 0 |
| `loss_type` / 權重 | ["sigmoid", "sft"] / [1, 1] | 標準 DPO loss 加上「繼續背 chosen」的 SFT loss，權重相同，兩個 loss 直接相加 |

### ref model

DPO 需要一個「訓練前的自己」當對照。這裡不另外載入第二份模型，而是讓 TRL（`ref_model=None`）在同一個底座上自動複製一份 SFT adapter 並凍結，叫做 `ref`。兩個 adapter 共用一個 4-bit 底座，所以 T4 的 16 GB 放得下。

訓練前先做兩個檢查：`ref` 要和 SFT adapter 完全一致，而且驗證集的 reward margin 要是 0。檢查時也看到：SFT 模型給 chosen 的 log-prob 是 −0.035，給 rejected 的是 −23.7，代表訓練前它就已經分得出這些題。

### 訓練曲線解讀

- **accuracy 第 25 步就到 1.0**：驗證集全部選對，表示這批題對模型太簡單，之後的訓練幾乎學不到新東西。
- **reward margin 0 → 5.75，約第 75 步後平台**：chosen 和 rejected 的差距拉開後就不再變大。
- **chosen 機率從約 97% 降到 81%**（log-prob −0.035 → −0.21）：DPO 只在乎「差距」，把 rejected 壓得更低、同時讓 chosen 也往下掉，差距一樣會變大。就算加了 SFT loss 也沒擋住。模型對正確答案信心變低，很可能是 test 分數退步的原因之一。

訓練後只存 adapter，推到 `yoyu001/grid-dispatch-A-instruct-qlora-dpo-v1`（私人），連同 log 和曲線圖。

## 評估設計與資料隔離

三個模型在同一個底座、同一批題目、同樣用 greedy 解碼下比較，而且考卷和訓練資料沒有重疊。這樣分數差異才能歸因於 DPO。

### 推論設定

vLLM 只載入一份 fp16 Qwen3-4B，用 multi-LoRA 在每個 request 上指定要用哪個 adapter：不掛（base）、掛 SFT、掛 SFT+DPO。一個底座跑三個模型，省記憶體，也排除「底座載入方式不同」這個變數。正式跑之前，先不掛 DPO 跑一次，確認成績和 SFT model card 一致。

### 三份考卷與隔離規則

| 資料 | 可以用來做什麼 |
| --- | --- |
| train | SFT 訓練、建 DPO 偏好資料 |
| validation | 只算 DPO 的 eval loss |
| test（500）、unseen（394）、真人手寫（20） | 只用來考試，絕不拿來建資料 |

真人手寫的 20 題是私人 Kaggle 資料集，只在 notebook 執行時讀入，不進 GitHub。

### 洩題檢查

評估前先用五種方法比對考卷和 train：完全相同、遮掉 workload 名稱後相同、8 字元片段、4-gram Jaccard 相似度、保留說法是否出現在 train。

- 完全重複 0 題；保留說法出現在訓練資料 0 次。
- 真人卷：遮名後重複 0 題，最高 4-gram Jaccard 只有 0.211，和 train 很不像。
- 但 test 有 41%、unseen 有 16%，在遮掉 workload 名稱後和某題 train 一模一樣，只是換了名稱。

### 熟題 vs 新句型

因為上面那 41% 和 16% 等於「換名字的舊題」，把它們和真正的新說法混在一起，會把模型的泛化能力高估。所以每份考卷都分成「熟題（只換名稱）」和「新句型」兩組各自算分。

### 顯著性：exact McNemar

同一批題比較兩個模型時，兩個都對或都錯的題不提供資訊，只看「一個對、一個錯」的題。例如 test 上 DPO 修好 3 題、弄壞 13 題，McNemar 問的是：「如果兩個模型其實一樣好，16 題裡分成 3 比 13 或更極端的機率有多小？」答案是 p = 0.021。題數少時用 exact（二項分佈）版本，而不是卡方近似。

## 結果

DPO 在沒看過的說法上進步 3.5 個百分點（p = 0.044，邊緣顯著），在和訓練同分布的說法（test）上退步 2.0 個百分點（p = 0.021），真人卷 20 題差 1 題、無法下結論。

| 考卷 | Base | SFT | SFT+DPO v1 | 差（DPO − SFT） |
| --- | --- | --- | --- | --- |
| test（500） | 19.6% | **98.4%** | 96.4% | −2.0 |
| unseen（394） | 19.8% | 80.5% | **84.0%** | +3.5 |
| 真人手寫（20） | 30% | **75%**（15/20） | 70%（14/20） | −5（1 題） |

**圖：SFT → SFT+DPO 答案對錯改變的題數**（03 run 2，greedy；圖表見 PDF 版）

| 考卷 | 被 DPO 弄壞 | 被 DPO 修好 | 淨變化 | McNemar p |
| --- | --- | --- | --- | --- |
| test（500） | 13 | 3 | −10 | 0.021 |
| unseen（394） | 14 | 28 | +14 | 0.044 |
| 真人手寫（20） | 2 | 1 | −1 | 1.0 |

準確率只差幾個百分點，但拆開看「哪些題被改了」更清楚：unseen 上淨修好 14 題，test 上淨弄壞 10 題。

**再細分**

- 以下數字全部來自 03 run 2，和主表、圖表一致。
- **test 的退步幾乎都在新句型**：新句型子組 98.3% → 95.2%（288 → 279 題），熟題（只換名稱）子組只有 98.6% → 98.1%（204 → 203 題）。
- **unseen 的進步來自新句型**：76.6% → 81.2%（252/329 → 267/329），不是換名字的舊題。
- unseen 的 grid 類：42.2% → 68.8%。
- **安全指標**：unseen 的漏拒 4 → 2，誤拒 1 → 1，沒有增加。
- 真人卷逐題：H04 被修好，H02、H16 被弄壞（細節見錯誤分析）。

同一個 DPO 訓練，在 test 的新句型上退步、在 unseen 的新句型上進步。理由尚未查明，錯誤分析的「多寫約束」是目前最可能的解釋。

## 錯誤分析

DPO 修好了「少寫」和「漏拒」，但矯正過頭，開始「多寫」（本節數字皆為 03 run 2）。另外有四題真人題兩個模型都錯，屬於 DPO 資料沒有涵蓋到的弱點。

### 改善

- unseen 的 grid 類：42.2% → 68.8%，是進步最多的題型。
- unseen 的 missing_constraint：53 題 → 31 題，少寫約束明顯變少。
- 漏拒 4 → 2，誤拒 1 → 1；拒絕類佔 21% 的偏好資料沒有把模型推向過度拒絕。
- 真人題 H04：SFT 漏了 `flex_tier=3`，DPO 補上了。

### 退步

- **多寫約束**：extra_constraint 在 test 從 1 題變 6 題、在 unseen 從 2 題變 5 題，常見的是自己加上備註沒要求的 `earliest_start` 或 `allow_pause`。
- 真人卷也看得到：SFT 的 extra_constraint 是 0 題，DPO 是 2 題。H02 多加了 `earliest_start`，H09 多加了一條 rag-index 的 `allow_pause`。
- **時間往半夜偏**：真人題 H16 把 21:00 改成了隔天 00:00。

### 可能原因

1. **偏好資料不對稱**：150 組懲罰少寫、 0 組懲罰多寫。模型學到的是「多寫一條比少寫一條安全」。
2. **chosen 機率下降**：訓練中 chosen 從約 97% 降到 81%，模型對原本會的答案也沒那麼確定了。
3. **推測（未驗證）**：250 組 time_rollover 資料強調「過半夜要進位」，可能讓模型在不該進位的時候也進位。

### 真人題：兩個模型都錯的題

| 題號 | 關鍵說法 | 正確應為 | 模型的錯 |
| --- | --- | --- | --- |
| H06 | 「幾乎感受不到」 | `flex_tier=1` | 程度詞對應錯等級 |
| H07 | 「不能夠做更動」 | 三條：客服 `flex_tier=0`、客服 `allow_pause=false`、ALL_FLEXIBLE `flex_tier=3` | 兩個模型都漏了一條 |
| H08 | 「今晚」 | `valid_until` 到隔天 08:00，兩條限制都要套用 | 有效期限沒有套用到兩條 |
| H09 | 「今天先暫停」 | RAG 索引的 `earliest_start` 值為隔天 00:00 | 兩個模型都把值寫成 10-28 23:59 |

這四題的共同點是「口語化的程度詞或否定詞」和「一句話對應到多條限制」，合成資料裡很少出現這種說法。

**額外風險**：H06 的 DPO 輸出中，`explanation` 說的和 JSON 裡的值互相矛盾。因為 `explanation` 不計分，這種問題在分數上看不出來，但調度員如果只看說明文字就會被誤導。

## 可重現性

同一份程式跑兩次，unseen 的成績每個模型差 1 題，就足以讓 p 值從 0.073 變成 0.044。所以 unseen 的「顯著進步」應該視為邊緣結果，不是確定的結論。

| unseen（394） | SFT | SFT+DPO | 修好 / 弄壞 | p |
| --- | --- | --- | --- | --- |
| 第 1 次 | 80.7% | 83.8% | 25 / 13 | 0.073 |
| 第 2 次 | 80.5% | 84.0% | 28 / 14 | 0.044 |

**為什麼 temperature 0 還會不一樣**：vLLM 把多個 request 放在同一批一起算。每次批次組合不同，GPU 加總浮點數的順序就不同，fp16 的尾數誤差也就不同。平常看不出來，但如果兩個 token 的機率剛好很接近，就可能這次選 A、下次選 B，後面整串輸出跟著改變。

**對報告的意義**

- test 的退步（p = 0.021，3 修好 / 13 弄壞）差距夠大，不太受這種雜訊影響。
- unseen 的進步在 0.05 邊緣彈跳，只能說「有進步跡象」。
- 真人卷只有 20 題，p = 1.0，只能當個別案例看。

**下次可以怎麼做**：同一份考卷跑 3 次，報告每次的分數範圍與 p 值範圍，而不是只報一次。

**重現步驟**

1. Kaggle → Create → Import Notebook → GitHub → `YOYU001/grid-dispatch-dpo`。
2. Secrets 加入 `HF_TOKEN`（00 要讀取權限，02 要寫入權限）。
3. GPU notebook（01a、02、03）選 GPU T4 x2、開啟 Internet，用 Save Version → Save & Run All (Commit) 執行。
4. 版本固定：02 用 trl 1.18.2、peft 0.22.0、transformers 5.24.0；03 的 vLLM 環境用 vLLM 0.34.0、transformers 5.20.2（01a 同樣用 vLLM 0.34.0）。

## 工程踩坑紀錄

花最多時間的不是模型本身，而是 T4 的精度限制和 Kaggle 的執行環境。最貴的一課是：失敗就要馬上停，不要偷偷退回備用方案。

| 問題 | 症狀 | 解法 |
| --- | --- | --- |
| vLLM 0.34.0 在 T4 上跑不起來 | T4 不支援 bf16；kernel 預熱時 import `cutlass` 失敗 | `dtype=half`、`enforce_eager`、`VLLM_ENABLE_V1_MULTIPROCESSING=0`，並確認 `nvidia-cutlass-dsl` 在 kernel 內可 import |
| 靜默退回 transformers | vLLM 失敗後自動改用 transformers，慢 5–6 倍卻沒有錯誤訊息，浪費約一小時 | 拿掉退回機制，只用 vLLM，失敗就停 |
| ref adapter 不一致（02 run 1） | TRL 複製的 `ref` 和 SFT adapter 不是每個位元都一樣，訓練前檢查擋下 | 印出 dtype 與差異，明確把 SFT 權重複製進 `ref` |
| LoRA 權重被轉成 bf16（02 run 2–3） | 第 1 步就失敗：TRL 把量化模型的可訓練 LoRA 轉成 bf16（QLoRA 論文預設），T4 的 fp16 GradScaler 無法處理 bf16 梯度 | 建好 trainer 後把可訓練參數轉回 fp32，再從 `ref`（轉型前複製的）把精確的 fp32 SFT 權重複製回 `default` |
| transformers 5 移除 `warmup_ratio` | 舊參數名不能用 | 改用 `warmup_steps=0.1`（小數代表比例） |
| Trackio Space 回 402 | 建立私人 Space 失敗 | 只存本機 log，另外自己畫 PNG 曲線和 adapter 一起上傳 |
| FileLink 404、互動模式重置 | 下載連結壞掉；session 重置後 `/kaggle/working` 裡的輸出消失 | 一律用背景 commit（Save & Run All），輸出寫到 `/kaggle/working` 根目錄 |
| Kaggle 改掉真人卷檔名 | 檔案變成 `grid-dispatch-human-test.jsonl`，03 run 1 找不到而跳過真人卷 | 改用 glob 搜尋 `*human*.jsonl`，找不到就停；加入 `RUN_SETS` 可只重跑真人卷 |
| Kaggle GPU 排隊 | 一次等好幾個小時 | 先在 CPU 用小型 Qwen3 跑通整個流程，再上 GPU |

**共通的做法**：每個關鍵步驟前先印出可檢查的數字（dtype、權重差異、訓練前 margin），不符合預期就停。在 GPU 要排隊好幾小時的環境裡，早停比跑完才發現錯誤便宜得多。

## 結論與後續

DPO v1 證明了整條流程在免費 T4 上跑得通，也證明「偏好資料懲罰什麼，模型就往反方向跑」。懲罰少寫卻不懲罰多寫，模型就開始多寫。

### 學到的事

1. **偏好資料要成對設計**：每一種錯誤都要有反方向的錯誤（少寫 ↔ 多寫、太早 ↔ 太晚），否則模型會矯正過頭。
2. **題目太簡單就是浪費算力**：accuracy 第 25 步就到 1.0，後面 80 步幾乎只在拉大差距，還把 chosen 機率拉低。
3. **分數要分組看**：不分「熟題 / 新句型」就看不出 DPO 真正幫到的是哪一塊。
4. **p 值在 0.05 邊緣時要多跑幾次**：fp16 批次的非確定性就能改變結論。
5. **不計分的欄位也要檢查**：`explanation` 和 JSON 矛盾時，分數完全看不出來。

### 後續方向

**接下來：課程期末作業**

- 只有 T4，而 `hf jobs` 要付費、帳號沒有額度，所以改用 QLoRA 訓練 SmolLM3-3B + `Anthropic/hh-rlhf`。
- 用 lighteval 評估 truthfulqa:mc2、hellaswag、arc:challenge。
- 用同一套考題測 grid-dispatch-A 的 base / SFT / DPO，量化 alignment tax（對齊後通用能力付出多少代價）。

**暫緩：DPO v2**

目前優先完整跑完一次 fine-tuning 流程，v2 先不做。若之後重啟，可以從這些地方下手：

- 加入 extra_constraint 反向資料，和 missing_constraint 平衡。
- 加入「不該進位卻進位」的時間資料。
- 多放口語程度詞、否定詞、一句話對應多條限制的題（H06–H09 類型）。
- 評估跑 3 次，報告範圍。
