# FraudLens AI - Digital Payment Fraud Detection (Mendeley dataset edition)
*Detect patterns. Assess risk. Protect transactions.*

Full-stack fraud-detection dashboard: **Python / Pandas / scikit-learn / FastAPI** backend with a real trained model, and a responsive dashboard (HTML/CSS/JavaScript + bundled Chart.js) served by the same FastAPI app. One command starts everything; there is no Node.js build step.

## Dataset
Uses the **Mendeley "Final 12 files" package**, already placed in `data/mendeley/`. It holds 5,000 transactions sampled from three public datasets:

| Source | Rows | What it is |
|---|---|---|
| PaySim | 2,016 | *Simulated* mobile-money transactions (not real UPI) |
| IEEE-CIS | 2,000 | E-commerce card transactions (`ProductCD` is used as the "type") |
| ULB | 984 | European credit-card transactions |

Columns: `Time, Amount, TransactionType, FraudLabel, Source`. The sample is **balanced 50/50 by design** and time-matched by the package authors, so:
* the 50 % fraud share is a sampling choice, not a real rate (real rates are roughly 0.13 % PaySim, 0.17 % ULB, 3.5 % IEEE-CIS);
* precision and accuracy on this sample are *not* real-world values - the Performance page projects them to realistic base rates;
* only time, amount, type and source are available (no PaySim balances, no card features).

Files used by the app: `train_time_80.csv`, `test_time_20.csv`, `train_random_80.csv`, `test_random_20.csv`, `harmonized_5000_temporally_matched.csv`. The package's own splits are used as-is (nothing is re-split). The other files are kept for reference.

## Run on Windows
1. Install **Python 3.10 - 3.13** from python.org (tick *Add python.exe to PATH*).
2. Double-click **`setup.bat`** (creates `.venv`, installs packages, trains the models - about a minute).
3. Double-click **`run.bat`** - the browser opens `http://localhost:8000`.

PowerShell alternative, inside the project folder:
```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python -m backend.train
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
```
Dashboard: <http://localhost:8000>. API docs: <http://localhost:8000/docs>.

## Project objectives
1. Understand the benchmark's construction, quality and balance.
2. Find which types, amounts and hours relate to fraud, per source.
3. Check split integrity (time vs random) and leakage risks.
4. Encode source-specific time comparably; engineer features without the label.
5. Compare Logistic Regression, Decision Tree, Random Forest and Gradient Boosting (precision, recall, F1, PR-AUC, ROC-AUC, confusion matrices).
6. Select the model by cross-validated PR-AUC on training data only; report test results once.
7. Project balanced-sample results to realistic prevalence; serve the model through an API and dashboard with CSV reports.

## Machine-learning design
| Topic | Decision |
|---|---|
| Evaluation | **Primary:** time-based split (train earlier 80 %, test later 20 % of each source). **Secondary:** random split. Both reported. |
| Selection | Highest 5-fold CV PR-AUC on the training file; threshold maximises F1 on out-of-fold predictions; test file used once |
| Features | Source, type (one-hot), log amount, per-source amount z-score, hour of day |
| Time handling | Raw `Time` has different units per source, so it is converted to hour of day |
| Imbalance | Not needed in the sample (balanced); handled instead by the base-rate projection |
| Models | Logistic Regression, Decision Tree, Random Forest, Gradient Boosting (`HistGradientBoostingClassifier`) |
| Explainability | Global: permutation importance. Per prediction: occlusion against a typical legitimate value of the same source |
| Scores in the table | Train rows use out-of-fold predictions, test rows use the held-out model, so every shown risk score is honest |
| Artifacts | `models/fraud_pipeline.joblib`, `models/model_metrics.json`, `data/transactions.csv` |

## API
`GET /api/health` - `GET /api/summary` - `GET /api/transactions` (search, filters, sort, pagination) - `GET /api/model-metrics` - `POST /api/predict` (validated) - `GET /api/insights` - `GET /api/dataset-info` - `GET /api/examples` - `GET /api/export/transactions | fraud-cases | model-metrics` (CSV)

## How to check it works
1. `http://localhost:8000/api/health` shows `"status":"ok"` and 5000 rows.
2. Overview shows 5,000 transactions and 2,500 fraud cases, with the "balanced by design" warning.
3. Transaction explorer: Source = PaySim, Type = TRANSFER, Label = Fraud gives the same count as the Analytics chart; Next moves to ID 26; sorting and CSV export follow the filters.
4. Fraud detection: click "Real fraud (caught)" then "Assess"; then "Real legitimate"; a negative amount shows a validation message.
5. With the server running: `python tests/api_check.py` (50 checks vs pandas ground truth) and `python tests/ui_check.py` (49 browser checks; needs `pip install playwright` and `playwright install chromium`).


## AI Explainability and chatbot (new)
Sidebar -> **AI Explainability** (`http://localhost:8000/#/explain`). Pick a transaction from the dataset to see the model's prediction versus the actual label, the fraud score against the decision threshold, exact per-transaction feature contributions (chart, table, plain-English summary), model limitations, and a chatbot ("Ask FraudLens AI") that answers about that transaction using only verified model output. It works with no API key; an optional LLM can be enabled with `FRAUDLENS_LLM_API_KEY` (see `docs/EXPLAINABILITY.md`).

Extra tests: `python -m pytest tests/test_explain_chat.py -v` (19 tests; no server needed) and `python tests/verify_against_shap.py` (needs `pip install shap`).

## Folder structure
```
fraudlens/
  backend/    main.py (API)  train.py  data.py  features.py  config.py  explain.py  chatbot.py  source_audit.py
  frontend/   index.html  styles.css  app.js  vendor/chart.umd.js (bundled, works offline)
  data/       mendeley/ (the 12-file package)   transactions.csv (written by training)
  models/     fraud_pipeline.joblib  model_metrics.json  source_audit.json
  notebooks/  01_eda_and_preprocessing.ipynb (executed, 7 plots)
  docs/       EXPLAINABILITY.md  REPORT_OUTLINE.md  SCREENSHOTS_CHECKLIST.md  FEATURE_CHECKLIST.md  screenshots/
  tests/      api_check.py  ui_check.py  test_explain_chat.py  verify_against_shap.py
  setup.bat  run.bat  requirements.txt  README.md
```
See `docs/FEATURE_CHECKLIST.md` for what is and is not done.
