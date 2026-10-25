"""Train, compare and save the fraud models on the Mendeley package.   Run:  python -m backend.train

Two evaluation protocols are reported, both using the train/test files supplied in the package:
  * time   (PRIMARY)  - train_time_80 -> test_time_20   (chronological within each source)
  * random (secondary)- train_random_80 -> test_random_20
Model selection and threshold tuning use ONLY 5-fold cross-validation on the training file;
the test file is touched once, for reporting.
"""
from __future__ import annotations
import json, sys, time
import joblib, numpy as np, pandas as pd, sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix, f1_score,
                             precision_recall_curve, precision_score, recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from . import config
from .data import load_package, integrity_report, random_test_flags
from .features import (engineer, CATEGORICAL, NUMERIC, SOURCES, ALL_TYPES, MODEL_FEATURES, RAW_FEATURES, TYPES_BY_SOURCE)

RS = config.RANDOM_STATE


def amount_stats(train: pd.DataFrame) -> dict:
    la = np.log1p(train["Amount"])
    return {s: (float(la[train.Source == s].mean()), float(la[train.Source == s].std() or 1.0)) for s in SOURCES}


def make_pipeline(clf, stats) -> Pipeline:
    pre = ColumnTransformer([
        ("cat", OneHotEncoder(categories=[SOURCES, ALL_TYPES], handle_unknown="ignore", sparse_output=False), CATEGORICAL),
        ("num", StandardScaler(), NUMERIC)])
    return Pipeline([("features", FunctionTransformer(engineer, kw_args={"stats": stats}, validate=False)),
                     ("preprocess", pre), ("model", clf)])


def candidates():
    return {"Logistic Regression": LogisticRegression(max_iter=2000, random_state=RS),
            "Decision Tree": DecisionTreeClassifier(max_depth=6, min_samples_leaf=5, random_state=RS),
            "Random Forest": RandomForestClassifier(n_estimators=300, min_samples_leaf=3, n_jobs=-1, random_state=RS),
            "Gradient Boosting": HistGradientBoostingClassifier(max_iter=150, learning_rate=0.06, random_state=RS)}


def best_f1_threshold(y, p):
    prec, rec, thr = precision_recall_curve(y, p)
    f1 = 2 * prec[:-1] * rec[:-1] / np.clip(prec[:-1] + rec[:-1], 1e-12, None)
    i = int(np.nanargmax(f1)); return float(thr[i]), float(f1[i])


def thin(x, y, n=150):
    idx = np.unique(np.linspace(0, len(x) - 1, min(n, len(x))).astype(int))
    return [[float(x[i]), float(y[i])] for i in idx]


def evaluate(y, p, thr):
    pred = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    fpr, tpr, _ = roc_curve(y, p); pr, rc, _ = precision_recall_curve(y, p)
    return {"threshold": thr, "precision": float(precision_score(y, pred, zero_division=0)), "recall": float(recall_score(y, pred)),
            "f1": float(f1_score(y, pred)), "roc_auc": float(roc_auc_score(y, p)), "pr_auc": float(average_precision_score(y, p)),
            "accuracy": float(accuracy_score(y, pred)), "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
            "roc_curve": thin(fpr, tpr), "pr_curve": thin(rc, pr)}


def run_protocol(name, tr, te):
    """CV-based selection on the training file, single evaluation on the test file."""
    stats = amount_stats(tr)
    Xtr, ytr, Xte, yte = tr[RAW_FEATURES], tr["FraudLabel"].to_numpy(), te[RAW_FEATURES], te["FraudLabel"].to_numpy()
    cv = StratifiedKFold(config.CV_FOLDS, shuffle=True, random_state=RS)
    res, fitted, oof = {}, {}, {}
    for mname, clf in candidates().items():
        t0 = time.time(); pipe = make_pipeline(clf, stats)
        o = cross_val_predict(pipe, Xtr, ytr, cv=cv, method="predict_proba")[:, 1]
        thr, cvf1 = best_f1_threshold(ytr, o)
        fit = make_pipeline(candidates()[mname], stats).fit(Xtr, ytr)
        r = evaluate(yte, fit.predict_proba(Xte)[:, 1], thr)
        r["cv"] = {"pr_auc": float(average_precision_score(ytr, o)), "roc_auc": float(roc_auc_score(ytr, o)), "f1": cvf1}
        r["train_seconds"] = round(time.time() - t0, 1)
        res[mname], fitted[mname], oof[mname] = r, fit, o
        print(f"[{name:6s}] {mname:20s} CV PR-AUC={r['cv']['pr_auc']:.3f} | test P={r['precision']:.3f} R={r['recall']:.3f} "
              f"F1={r['f1']:.3f} PR-AUC={r['pr_auc']:.3f} AUC={r['roc_auc']:.3f}")
    best = max(res, key=lambda n: (round(res[n]["cv"]["pr_auc"], 3), res[n]["cv"]["f1"]))
    return {"models": res, "selected": best, "n_train": int(len(tr)), "n_test": int(len(te)),
            "test_fraud": int(yte.sum())}, fitted, oof, stats


def main():
    t0 = time.time(); config.MODELS_DIR.mkdir(exist_ok=True)
    d = load_package(); integ = integrity_report(d)
    print("[1/5] Loaded package:", {k: len(v) for k, v in d.items()}); print("      integrity:", integ)
    primary, fit_t, oof_t, stats = run_protocol("time", d["train_time"], d["test_time"])
    secondary, *_ = run_protocol("random", d["train_random"], d["test_random"])
    best = primary["selected"]; pipe = fit_t[best]; thr = primary["models"][best]["threshold"]
    print(f"[2/5] Selected (time protocol, CV PR-AUC): {best}, threshold {thr:.4f}")

    tr, te = d["train_time"], d["test_time"]; yte = te["FraudLabel"].to_numpy()
    pte = pipe.predict_proba(te[RAW_FEATURES])[:, 1]
    # per-source test performance + illustrative projection to real-world base rates
    per_source, proj = {}, []
    for s in SOURCES:
        m = (te["Source"] == s).to_numpy(); y, p = yte[m], pte[m]; pred = (p >= thr).astype(int)
        tp = int(((pred == 1) & (y == 1)).sum()); fn = int(((pred == 0) & (y == 1)).sum())
        fp = int(((pred == 1) & (y == 0)).sum()); tn = int(((pred == 0) & (y == 0)).sum())
        tpr, fpr = tp / max(tp + fn, 1), fp / max(fp + tn, 1)
        per_source[s] = {"n_test": int(m.sum()), "fraud_test": int(y.sum()), "roc_auc": float(roc_auc_score(y, p)),
                         "pr_auc": float(average_precision_score(y, p)), "recall": tpr, "false_positive_rate": fpr,
                         "precision": tp / max(tp + fp, 1), "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp}}
        pi = config.REAL_BASE_RATES[s]; flagged = tpr * pi + fpr * (1 - pi)
        proj.append({"source": s, "real_base_rate": pi, "recall": tpr, "false_positive_rate": fpr,
                     "precision_at_real_rate": (tpr * pi / flagged) if flagged > 0 else None,
                     "alerts_per_1000": 1000 * flagged, "false_alarms_per_true_fraud": (fpr * (1 - pi) / (tpr * pi)) if tpr * pi > 0 else None})

    pi_ = permutation_importance(pipe, te[RAW_FEATURES], yte, scoring="average_precision", n_repeats=30, random_state=RS, n_jobs=1)
    importance = sorted([{"feature": f, "importance": float(a), "std": float(b)} for f, a, b in
                         zip(RAW_FEATURES, pi_.importances_mean, pi_.importances_std)], key=lambda x: -x["importance"])

    # baseline "typical legitimate transaction" per source, used by the per-prediction occlusion explanation
    leg = tr[tr.FraudLabel == 0]
    baseline = {s: {"TransactionType": leg[leg.Source == s].TransactionType.mode().iloc[0],
                    "Amount": float(leg[leg.Source == s].Amount.median()), "hour_of_day": int(leg[leg.Source == s].hour_of_day.median())}
                for s in SOURCES}

    # dashboard table: honest scores for every row (out-of-fold for train rows, held-out for test rows)
    dash = pd.concat([tr.assign(split="train", score_kind="out-of-fold", risk_score=oof_t[best]),
                      te.assign(split="test", score_kind="held-out test", risk_score=pte)], ignore_index=True)
    dash["in_random_test"] = random_test_flags(dash, d["test_random"])
    dash["predicted_fraud"] = (dash["risk_score"] >= thr).astype(int); dash["risk_score"] = dash["risk_score"].round(5)
    dash.insert(0, "tx_id", np.arange(1, len(dash) + 1))
    dash = dash[["tx_id", "Source", "TransactionType", "Amount", "Time", "hour_of_day", "FraudLabel", "risk_score",
                 "predicted_fraud", "split", "score_kind", "in_random_test"]]
    dash.to_csv(config.PROCESSED_CSV, index=False)

    joblib.dump({"pipeline": pipe, "threshold": thr, "model_name": best, "baseline": baseline, "raw_features": RAW_FEATURES,
                 "sklearn_version": sklearn.__version__}, config.PIPELINE_PATH)
    h = d["harmonized"]
    meta = {"primary_protocol": "time", "selected_model": best, "threshold": thr,
            "selection_rule": "Highest 5-fold cross-validated PR-AUC on the training file (ties: CV F1). The decision threshold maximises F1 on the "
                              "out-of-fold predictions. The test file is used once, for reporting.",
            "protocols": {"time": primary, "random": secondary}, "per_source": per_source, "base_rate_projection": proj,
            "feature_importance": importance, "model_features": MODEL_FEATURES, "raw_features": RAW_FEATURES, "integrity": integ,
            "dataset": {"name": "Mendeley harmonized fraud package (PaySim + IEEE-CIS + ULB samples)", "rows": int(len(h)),
                        "fraud_rows": int(h.FraudLabel.sum()), "per_source": {s: {"rows": int((h.Source == s).sum()),
                        "fraud": int(h[h.Source == s].FraudLabel.sum())} for s in SOURCES}, "files_used": config.FILES,
                        "balanced_by_design": True, "real_base_rates": config.REAL_BASE_RATES},
            "excluded_or_transformed": {"Time": "raw value has different units per source -> converted to hour_of_day",
                                        "FraudLabel": "target", "tx_id": "row index added by the app"},
            "versions": {"python": sys.version.split()[0], "scikit_learn": sklearn.__version__, "pandas": pd.__version__, "numpy": np.__version__},
            "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"), "total_train_seconds": round(time.time() - t0, 1)}
    config.METRICS_PATH.write_text(json.dumps(meta, indent=2))
    print("[3/5] per-source test AUC:", {s: round(v["roc_auc"], 3) for s, v in per_source.items()})
    print("[4/5] Saved", config.PIPELINE_PATH.name, config.METRICS_PATH.name, config.PROCESSED_CSV.name)
    print(f"[5/5] Done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
