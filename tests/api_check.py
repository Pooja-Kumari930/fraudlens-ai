"""Integration checks against a RUNNING server (python tests/api_check.py). Compares API output to pandas ground truth."""
import io, sys, requests, pandas as pd
B = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"
df = pd.read_csv("data/transactions.csv")
def j(path):
    r = requests.get(B + path); return r.status_code, (r.json() if "json" in r.headers.get("content-type", "") else r.text)
fails = []
def check(name, cond, extra=""):
    print(("PASS " if cond else "FAIL ") + name, extra)
    if not cond: fails.append(name)
N = len(df); F = int(df.FraudLabel.sum())
c, r = j("/api/transactions?page=1&page_size=10"); check("pagination totals", c == 200 and r["total"] == N and len(r["rows"]) == 10, f'pages={r["pages"]}')
c, r2 = j("/api/transactions?page=2&page_size=10"); check("page 2 starts at id 11", r2["rows"][0]["tx_id"] == 11)
c, r = j("/api/transactions?page=999999&page_size=10"); check("out-of-range page clamps", r["page"] == r["pages"])
for q, mask in [("source=PaySim&label=fraud", (df.Source == "PaySim") & (df.FraudLabel == 1)),
                ("type=TRANSFER&label=fraud", (df.TransactionType == "TRANSFER") & (df.FraudLabel == 1)),
                ("source=IEEE-CIS&type=W", (df.Source == "IEEE-CIS") & (df.TransactionType == "W")),
                ("min_amount=100&max_amount=1000", df.Amount.between(100, 1000)),
                ("label=legit&model_flag=flagged", (df.FraudLabel == 0) & (df.predicted_fraud == 1)),
                ("split=test&label=fraud", (df.split == "test") & (df.FraudLabel == 1)),
                ("split=train", df.split == "train")]:
    c, r = j("/api/transactions?" + q + "&page_size=5"); check("filter " + q, c == 200 and r["total"] == int(mask.sum()), f'{r.get("total")} rows')
c, r = j("/api/transactions?search=555"); check("search by tx id", any(x["tx_id"] == 555 for x in r["rows"]))
c, r = j("/api/transactions?search=cash_out"); check("search by type text (case-insensitive)", r["total"] == int((df.TransactionType == "CASH_OUT").sum()))
c, r = j("/api/transactions?search=zzzzzzz"); check("empty search -> 0 rows", r["total"] == 0 and r["rows"] == [])
c, r = j("/api/transactions?sort_by=Amount&sort_dir=desc&page_size=5"); check("sort Amount desc", r["rows"][0]["Amount"] == df.Amount.max())
c, r = j("/api/transactions?sort_by=risk_score&sort_dir=asc&page_size=5"); check("sort risk_score asc", r["rows"][0]["risk_score"] == df.risk_score.min())
for q in ["sort_by=evil", "type=XX", "source=Nope", "min_amount=-1", "min_amount=10&max_amount=5", "page=0", "page_size=1000"]:
    c, _ = j("/api/transactions?" + q); check(f"bad param rejected ({q})", c == 422, f"-> {c}")
r = requests.get(B + "/api/export/transactions?source=PaySim&label=fraud"); e = pd.read_csv(io.StringIO(r.text))
check("CSV export respects filters", len(e) == int(((df.Source == "PaySim") & (df.FraudLabel == 1)).sum()) and (e.Source == "PaySim").all() and (e.FraudLabel == 1).all(), f"{len(e)} rows")
r = requests.get(B + "/api/export/transactions?source=ULB&type=TRANSFER"); check("export with no matches -> 404 message", r.status_code == 404 and "nothing to export" in r.json()["detail"])
r = requests.get(B + "/api/export/fraud-cases"); check("fraud-cases export", len(pd.read_csv(io.StringIO(r.text))) == F)
r = requests.get(B + "/api/export/model-metrics"); mm = pd.read_csv(io.StringIO(r.text)); check("model-metrics export: 4 models x 2 protocols", len(mm) == 8 and mm.selected.sum() == 1)
s = j("/api/summary")[1]; check("summary equals dataset", s["total_transactions"] == N and s["fraud_cases"] == F and abs(s["fraud_share"] - df.FraudLabel.mean()) < 1e-12)
i = j("/api/insights")[1]; check("insights by_type sums to total", sum(x["count"] for x in i["by_type"]) == N); check("insights hourly fraud sums", sum(x["fraud"] for x in i["hourly"]) == F)
check("insights amount bands sum to total", sum(x["count"] for x in i["amount_bands"]) == N)
m = j("/api/model-metrics")[1]; t = m["protocols"]["time"]["models"][m["selected_model"]]
te = df[df.split == "test"]; check("test-set size in metrics matches data", m["protocols"]["time"]["n_test"] == len(te))
cm = t["confusion_matrix"]; check("confusion matrix matches stored predictions", cm["tp"] == int(((te.predicted_fraud == 1) & (te.FraudLabel == 1)).sum()) and cm["fp"] == int(((te.predicted_fraud == 1) & (te.FraudLabel == 0)).sum()))
from sklearn.metrics import roc_auc_score
check("ROC-AUC recomputed from scores matches", abs(roc_auc_score(te.FraudLabel, te.risk_score) - t["roc_auc"]) < 1e-3)
for p in ["/api/health", "/api/dataset-info", "/api/model-metrics", "/api/examples", "/"]:
    check("GET " + p, j(p)[0] == 200)
check("unknown route 404", j("/api/nope")[0] == 404)
P = lambda b: requests.post(B + "/api/predict", json=b)
ok = {"source": "PaySim", "transaction_type": "CASH_OUT", "amount": 1000, "hour_of_day": 10}
r = P(ok); check("predict valid", r.status_code == 200 and 0 <= r.json()["fraud_probability"] <= 1)
ex = j("/api/examples")[1]; fr = ex["fraud_caught"][0]; r = P({k: fr[k] for k in ok}); check("predict real caught-fraud example flagged", r.status_code == 200 and r.json()["is_fraud"], f'p={r.json()["fraud_probability"]:.3f}')
lg = ex["legit_clear"][0]; r = P({k: lg[k] for k in ok}); check("predict real legit example not flagged", r.status_code == 200 and not r.json()["is_fraud"], f'p={r.json()["fraud_probability"]:.3f}')
a = P({**ok, "transaction_type": "TRANSFER", "amount": 300000}).json()["fraud_probability"]; b = P({**ok, "transaction_type": "PAYMENT", "amount": 300000}).json()["fraud_probability"]
check("model responds to inputs (TRANSFER riskier than PAYMENT)", a > b, f"{a:.3f} > {b:.3f}")
for name, bad in [("bad source", {**ok, "source": "X"}), ("type not valid for source", {**ok, "source": "ULB"}), ("negative amount", {**ok, "amount": -1}),
                  ("hour 24", {**ok, "hour_of_day": 24}), ("missing field", {"source": "ULB"}), ("string amount", {**ok, "amount": "abc"}), ("huge amount", {**ok, "amount": 1e12})]:
    check("predict rejects " + name, P(bad).status_code == 422)
print("\nFAILED:" if fails else "\nALL CHECKS PASSED", fails if fails else "")
sys.exit(1 if fails else 0)
