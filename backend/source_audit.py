"""Audit: does using `Source` as a model input change or bias results?   Run: python -m backend.source_audit
Read-only analysis. It does NOT change the deployed model, split, threshold or metrics.
Writes models/source_audit.json. Variants use the same Gradient Boosting settings and the package's time split."""
from __future__ import annotations
import json, warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from . import config
from .data import load_package
from .features import SOURCES, ALL_TYPES, RAW_FEATURES, engineer
from .train import amount_stats, candidates, make_pipeline


def variant_pipeline(name: str, stats: dict):
    def feats(df):
        e = engineer(df, stats=stats if name in ("current", "no_source_onehot") else None)
        return e
    cats = {"current": ["Source", "TransactionType"], "no_source_onehot": ["TransactionType"], "no_source_info": ["TransactionType"], "amount_hour_only": []}[name]
    nums = {"current": ["log_amount", "amount_z", "hour_of_day"], "no_source_onehot": ["log_amount", "amount_z", "hour_of_day"],
            "no_source_info": ["log_amount", "hour_of_day"], "amount_hour_only": ["log_amount", "hour_of_day"]}[name]
    parts = []
    if cats:
        parts.append(("cat", OneHotEncoder(categories=[ALL_TYPES] if cats == ["TransactionType"] else [SOURCES, ALL_TYPES], handle_unknown="ignore", sparse_output=False), cats))
    parts.append(("num", StandardScaler(), nums))
    return Pipeline([("features", FunctionTransformer(feats, validate=False)), ("preprocess", ColumnTransformer(parts)), ("model", candidates()["Gradient Boosting"])])


def main():
    d = load_package(); tr, te = d["train_time"], d["test_time"]; stats = amount_stats(tr)
    ytr, yte = tr.FraudLabel.to_numpy(), te.FraudLabel.to_numpy(); cv = StratifiedKFold(config.CV_FOLDS, shuffle=True, random_state=config.RANDOM_STATE)
    desc = {"current": "Deployed model: Source and TransactionType one-hot, log amount, per-source amount z-score, hour",
            "no_source_onehot": "Source one-hot removed (type, log amount, per-source z-score, hour kept)",
            "no_source_info": "No Source anywhere: no one-hot and no per-source amount scaling (type, log amount, hour)",
            "amount_hour_only": "Amount and hour only: neither Source nor TransactionType"}
    rows = {}
    for name in desc:
        p = variant_pipeline(name, stats); oof = cross_val_predict(p, tr[RAW_FEATURES], ytr, cv=cv, method="predict_proba")[:, 1]
        fit = variant_pipeline(name, stats).fit(tr[RAW_FEATURES], ytr); pt = fit.predict_proba(te[RAW_FEATURES])[:, 1]
        per = {s: float(roc_auc_score(yte[(te.Source == s).to_numpy()], pt[(te.Source == s).to_numpy()])) for s in SOURCES}
        rows[name] = {"description": desc[name], "cv_pr_auc": float(average_precision_score(ytr, oof)), "test_roc_auc": float(roc_auc_score(yte, pt)),
                      "test_pr_auc": float(average_precision_score(yte, pt)), "test_roc_auc_by_source": per}
        print(f"{name:18s} CV PR-AUC={rows[name]['cv_pr_auc']:.3f} test AUC={rows[name]['test_roc_auc']:.3f} " + " ".join(f"{s}={v:.3f}" for s, v in per.items()))
    # one separate model per source (no Source input needed) vs the pooled model
    per_model = {}
    for s in SOURCES:
        a, b = tr[tr.Source == s], te[te.Source == s]
        m = variant_pipeline("no_source_info", stats).fit(a[RAW_FEATURES], a.FraudLabel); per_model[s] = float(roc_auc_score(b.FraudLabel, m.predict_proba(b[RAW_FEATURES])[:, 1]))
    rows["separate_model_per_source"] = {"description": "One model trained per source (type, log amount, hour)", "test_roc_auc_by_source": per_model}
    print("separate per-source models:", {k: round(v, 3) for k, v in per_model.items()})
    types_per_source = pd.concat([tr, te]).groupby("TransactionType").Source.nunique()
    out = {"variants": rows, "type_identifies_source": bool((types_per_source == 1).all()),
           "note": "TransactionType values never overlap between sources, so the type column alone reveals the source."}
    (config.MODELS_DIR / "source_audit.json").write_text(json.dumps(out, indent=2)); print("saved models/source_audit.json")


if __name__ == "__main__":
    main()
