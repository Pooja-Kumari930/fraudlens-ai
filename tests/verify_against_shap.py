"""Cross-check backend/explain.py (exact Shapley) against the SHAP library. Needs:  pip install shap
Runs the SHAP exact explainer on the SAME value function (same-source background, source fixed) and
compares contributions for many transactions. Also reports sign agreement with shap.TreeExplainer
(which works in the model's internal one-hot / log-odds space, so only direction is comparable)."""
import sys, warnings; warnings.filterwarnings("ignore")
sys.path.insert(0, ".")
import joblib, numpy as np, pandas as pd
try:
    import shap
except ImportError:
    print("shap not installed - skipping (pip install shap)"); sys.exit(0)
from backend.explain import explain_row, predict_one, make_background, EXPLAINED
df = pd.read_csv("data/transactions.csv"); b = joblib.load("models/fraud_pipeline.joblib"); raw = b["raw_features"]; pipe = b["pipeline"]
types = sorted(df.TransactionType.unique()); code = {t: i for i, t in enumerate(types)}
rng = np.random.default_rng(0); idx = rng.choice(len(df), 24, replace=False); maxdiff = 0.0; agree = tot = 0
tree = shap.TreeExplainer(pipe[-1])
for i in idx:
    row = df.iloc[i]; res = explain_row(df, b, row, predict_one(b, row)); mine = {c["feature"]: c["contribution"] for c in res["contributions"]}
    bg = make_background(df, row.Source, raw)
    def f(M):                                            # M columns: type code, amount, hour (Source fixed)
        Z = pd.DataFrame({"Source": row.Source, "TransactionType": [types[int(round(c))] for c in M[:, 0]], "Amount": M[:, 1], "hour_of_day": M[:, 2]})
        return pipe.predict_proba(Z[raw])[:, 1]
    bgm = np.column_stack([bg.TransactionType.map(code), bg.Amount, bg.hour_of_day]).astype(float)
    ex = shap.explainers.Exact(f, shap.maskers.Independent(bgm, max_samples=len(bgm)))
    sv = ex(np.array([[code[row.TransactionType], row.Amount, row.hour_of_day]], dtype=float)).values[0]
    theirs = dict(zip(["TransactionType", "Amount", "hour_of_day"], sv)); maxdiff = max(maxdiff, max(abs(mine[k] - theirs[k]) for k in EXPLAINED))
    # direction vs TreeExplainer (grouped one-hot columns, log-odds): compare sign of the largest contribution
    Xt = pipe[:-1].transform(pd.DataFrame([row[raw]])); tv = tree.shap_values(Xt)[0]; names = pipe[1].get_feature_names_out()
    grp = {"TransactionType": sum(v for n, v in zip(names, tv) if "TransactionType" in n), "Amount": sum(v for n, v in zip(names, tv) if n in ("num__log_amount", "num__amount_z")), "hour_of_day": tv[list(names).index("num__hour_of_day")]}
    top = max(EXPLAINED, key=lambda k: abs(mine[k])); tot += 1; agree += int(np.sign(mine[top]) == np.sign(grp[top]))
print(f"Compared {len(idx)} transactions. Max |my exact value - shap Exact value| = {maxdiff:.2e} (score points x100: {maxdiff*100:.4f})")
print(f"Sign of the largest contribution agrees with shap.TreeExplainer (grouped, log-odds): {agree}/{tot}")
sys.exit(0 if maxdiff < 1e-6 else 1)
