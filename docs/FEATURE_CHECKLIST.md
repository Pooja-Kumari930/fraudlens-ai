# Feature checklist (what was and was not verified)
Verified = executed in the build sandbox (Linux, Python 3.12, headless Chromium).

## Completed and tested
- [x] Dashboard pages: overview, transaction explorer, fraud detection, analytics, model performance, reports, dataset/model
- [x] Search, 8 filters, sorting, pagination, page size - compared with pandas ground truth (API) and clicked in a browser (UI)
- [x] CSV export (filtered transactions, fraud cases, model comparison); downloads checked in browser and parsed
- [x] Prediction form -> `/api/predict` -> saved model; valid and invalid inputs (client-side and server-side validation); model responds to inputs
- [x] Metrics, ROC-AUC and confusion matrix recomputed from stored scores and matched to the reported values
- [x] 4 models x 2 evaluation protocols; selection by CV PR-AUC; test used once
- [x] Loading spinners, empty state, error banners, backend-offline message, phone-width layout
- [x] Notebook executed with 0 errors, 7 visualisations
- [x] 50 API checks and 49 browser checks passing
- [x] AI Explainability page: transaction selector, prediction vs actual label, threshold, exact per-transaction contributions, chart, plain-English summary, limitations
- [x] "Ask FraudLens AI" chatbot (built-in grounded answers; optional LLM): 19 pytest tests incl. a numeric grounding audit, SHAP cross-check, error/validation cases
- [x] Source audit (read-only): removing Source does not change results
- [x] Uses the supplied real data (no synthetic stand-in)

## NOT done / limitations - please read
- [ ] **Not tested on Windows.** `setup.bat`, `run.bat` and the PowerShell commands were never executed on Windows (code ran on Linux only).
- [ ] **Small, balanced benchmark.** 5,000 rows; 50 % fraud by construction. Precision/accuracy are not real-world values.
- [ ] **Weak sources.** IEEE-CIS (test AUC about 0.72) and ULB (about 0.65) have only amount and hour available; per-source test sets are small (ULB about 200 rows), so those numbers are uncertain.
- [ ] **Mixed semantics.** IEEE-CIS "type" is the ProductCD code; amounts are in different (unspecified) units per source; PaySim is simulated, not UPI.
- [ ] `Source` is a model input, which is unrealistic outside this mixed benchmark.
- [ ] Real-world precision projection uses approximate public prevalence values and small test samples - illustrative only.
- [ ] 21 duplicate rows and 24 (source, time, amount) keys shared by random-split train/test were left as supplied; the time split has no overlap.
- [ ] No cross-validation of the final test result, no authentication/database/streaming (data held in memory).
- [ ] The older /api/predict form still uses an approximate occlusion explanation; the new AI Explainability page uses exact Shapley values.
- [ ] **Optional LLM mode was never run against a real provider** (no key available); only tested with a mocked network layer.
- [ ] Explanations vary only 3 inputs (type, amount, hour); no account/device/merchant/location data exists.
- [ ] Curves are thinned to about 150 points for the browser.
