# AI Explainability and "Ask FraudLens AI" - how it works

Open the dashboard and click **AI Explainability** in the sidebar (`http://localhost:8000/#/explain`).

## What the page does
1. You pick a transaction from the real dataset (filter by group, source, set, or type an ID).
2. The backend scores it with the **saved model** (`models/fraud_pipeline.joblib`) and calculates feature contributions for that one transaction.
3. The page shows the details, the **model prediction** (separate from the **actual historical label**), the fraud score against the model's decision threshold, a plain-English explanation, a contribution chart and table, and limitations.
4. The chatbot answers questions about *that* transaction. Changing the transaction starts a new conversation.

## Explanation method (and why not SHAP's TreeExplainer)
* The model's real inputs are `Source`, `TransactionType`, `Amount`, `hour_of_day`. `FraudLabel` is not an input (tested: changing every label does not change any score or contribution).
* I computed **exact Shapley values** over those raw inputs, in score points (probability x 100). `Source` is held fixed as context and the baseline is the average score of up to 400 *training* transactions from the same source, so the model is never asked about an impossible combination (e.g. a PaySim type with a ULB source). With three varied inputs there are only 8 coalitions, so no sampling or approximation is involved.
* Every explanation is checked: `baseline + sum(contributions) = model score` (tolerance 1e-6). If the check fails, no contributions are shown (an honest error appears instead).
* **Why not `shap.TreeExplainer`?** It does run on this model, but it works inside the pipeline's internal one-hot space in log-odds (17 columns), so contributions would have to be regrouped into user-facing inputs and could not be expressed in score points. The exact calculation explains the inputs a user actually sees and adds no heavy runtime dependency.
* **Cross-check:** `tests/verify_against_shap.py` recomputes 24 transactions with the SHAP library's exact explainer on the same value function. Maximum difference: 3.3e-15. Agreement with `TreeExplainer` is only directional (top-contribution sign agreed on 19 of 24), as expected for a different baseline and scale.
* Contributions describe the **model's reasoning**, not real-world causes of fraud.

## Is `Source` a problem? (audit, `python -m backend.source_audit`)
Read-only; the deployed model, split, threshold and metrics were **not** changed. Gradient Boosting, same time split:

| Variant | Test ROC-AUC | PaySim / IEEE-CIS / ULB |
|---|---|---|
| Deployed model (Source + type + amount + hour) | 0.844 | 0.968 / 0.721 / 0.654 |
| Without Source one-hot | 0.843 | 0.969 / 0.718 / 0.658 |
| Without any source-specific input or scaling | 0.850 | 0.969 / 0.728 / 0.673 |
| Amount and hour only (no source, no type) | 0.729 | 0.892 / 0.566 / 0.554 |
| Separate model per source | - | 0.965 / 0.692 / 0.652 |

Finding: removing Source changes nothing meaningful. `TransactionType` values never overlap between sources, so the type column already reveals the source; the model is source-aware either way. Differences of about 0.01 AUC are within the noise of these small test sets (no confidence intervals were computed). The performance gap between sources is real and is shown on the Performance page and in the chatbot.

## Chatbot design
* Browser sends only `tx_id`, the question and the last few turns. The **server rebuilds the evidence from `tx_id`** (transaction, live model score, threshold, contributions, metrics, per-source results) - the client cannot inject facts.
* **Built-in mode (default, no key needed):** deterministic answers assembled from that evidence. Covers the 7 suggested questions plus source performance, base rate, actual label, method, Source audit, unavailable data and "is it definitely fraud?". Unknown questions get an honest "I can only answer..." reply.
* **Optional LLM mode:** active only if `FRAUDLENS_LLM_API_KEY` (or `ANTHROPIC_API_KEY`) is set **on the backend**. The provider receives a minimal verified context (no transaction ID, no raw timestamp, no key in the body) and a strict system prompt. Questions about unavailable data or certainty are always answered by the built-in engine. Any provider error falls back to the built-in answer and says so. Optional `FRAUDLENS_LLM_MODEL` overrides the model name (default `claude-sonnet-5-5`).
* The chatbot is told, and tested, not to: invent facts, call a flag proof, claim to know identity/location/device/merchant/balance/history, or treat the 50% sample share as a real rate.

### Configure an optional LLM key (Windows)
* **Double-click users:** create `llm_key.txt` in the project folder with the key on the first line, then use `run.bat` (it loads the file; never share or commit it).
* **PowerShell:** `$env:FRAUDLENS_LLM_API_KEY="your-key"` then `python -m uvicorn backend.main:app --port 8000`.
* Confirm: the chat panel badge changes from "Built-in answers" to "LLM-assisted answers" (`/api/chat/status`).

## API
`GET /api/explain/{tx_id}` - prediction, threshold, contributions, summary, metrics context.
`POST /api/chat` - body `{tx_id, message (1-500 chars), history (max 8 turns)}`.
`GET /api/chat/status` - `built-in` or `llm-assisted` (never returns the key).

## Known limitations
* The live LLM call was **not tested** against the real provider (no key was available); it is tested with a mocked network layer only (request contents, key handling, error fallback).
* LLM answers cannot be fully verified; the system prompt and context limit risk but do not eliminate it. Built-in answers are numerically audited by a test.
* Scores are not calibrated probabilities; the sample is 50% fraud by design. For rows from the training set the model has already seen the transaction (the page says so).
* Amount units differ by source and are unspecified; `hour_of_day` for IEEE-CIS and ULB comes from timestamps with unknown reference start.
* Only three inputs vary per explanation, so explanations are inherently coarse. No account, device, merchant or location information exists in this dataset.
