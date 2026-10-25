"""Local (per-transaction) explanations for the deployed fraud model.

Method: EXACT Shapley values over the model's raw user-facing inputs.
  * The model takes four raw inputs: Source, TransactionType, Amount, hour_of_day.
  * Source is held fixed as context. TransactionType values never overlap between sources, so mixing a
    PaySim type with a ULB source would be an impossible transaction; fixing the source avoids that.
  * The baseline is the average model score of TRAINING transactions from the same source.
  * With 3 explained inputs there are only 8 coalitions, so the Shapley values are computed exactly
    (no sampling, no approximation). Values are in score points (probability), and satisfy
        baseline + sum(contributions) == model score      (checked on every call).
This is the same quantity the SHAP library estimates (interventional Shapley values); see
tests/verify_against_shap.py for a cross-check against shap.
FraudLabel is never an input: it is not in the model's raw features.
"""
from __future__ import annotations
from functools import lru_cache
from itertools import combinations
from math import factorial
import numpy as np
import pandas as pd

EXPLAINED = ["TransactionType", "Amount", "hour_of_day"]      # Source is fixed context
LABELS = {"TransactionType": "Transaction type", "Amount": "Amount", "hour_of_day": "Hour of day"}
BACKGROUND_ROWS = 400
MIN_MATERIAL = 0.01          # contributions smaller than 1 score point are treated as negligible in the text


def shapley_exact(predict, x: dict, background: pd.DataFrame, features: list[str]):
    """Exact Shapley values of predict() for one row against a background sample."""
    n = len(features); memo: dict = {}

    def v(S: tuple) -> float:
        key = frozenset(S)
        if key not in memo:
            Z = background.copy()
            for f in S:
                Z[f] = x[f]
            memo[key] = float(np.mean(predict(Z)))
        return memo[key]

    phi = {}
    for f in features:
        others = [g for g in features if g != f]; total = 0.0
        for k in range(n):
            w = factorial(k) * factorial(n - k - 1) / factorial(n)
            for S in combinations(others, k):
                total += w * (v(S + (f,)) - v(S))
        phi[f] = total
    return phi, v(()), v(tuple(features))


def make_background(df: pd.DataFrame, source: str, raw_features: list[str]) -> pd.DataFrame:
    tr = df[(df["split"] == "train") & (df["Source"] == source)][raw_features]
    if len(tr) > BACKGROUND_ROWS:
        tr = tr.sample(BACKGROUND_ROWS, random_state=42)
    return tr.reset_index(drop=True)


def fmt_value(feature: str, value) -> str:
    if feature == "Amount":
        return f"{float(value):,.2f}"
    if feature == "hour_of_day":
        return f"{int(value):02d}:00"
    return str(value)


def predict_one(bundle: dict, row: pd.Series) -> float:
    """Score one transaction with the deployed pipeline. FraudLabel is not among the raw features."""
    raw = bundle["raw_features"]
    return float(bundle["pipeline"].predict_proba(pd.DataFrame([{f: row[f] for f in raw}]))[:, 1][0])


def explain_row(df: pd.DataFrame, bundle: dict, row: pd.Series, prob: float) -> dict:
    """Exact feature contributions for one transaction (raises if the additivity self-check fails)."""
    raw = bundle["raw_features"]; pipe = bundle["pipeline"]
    x = {f: row[f] for f in raw}
    predict = lambda Z: pipe.predict_proba(Z[raw])[:, 1]
    bg = make_background(df, str(row["Source"]), raw)
    phi, base, full = shapley_exact(predict, x, bg, EXPLAINED)
    gap = abs((base + sum(phi.values())) - prob)
    if gap > 1e-6 or abs(full - prob) > 1e-6:       # never show attributions that do not add up to the score
        raise ValueError(f"explanation failed its additivity check (gap {gap:.2e})")
    items = [{"feature": f, "label": LABELS[f], "value": (x[f].item() if isinstance(x[f], np.generic) else x[f]), "display_value": fmt_value(f, x[f]),
              "contribution": phi[f], "direction": "increases" if phi[f] > 0 else "decreases" if phi[f] < 0 else "neutral"}
             for f in EXPLAINED]
    items.sort(key=lambda d: -abs(d["contribution"]))
    return {"method": "Exact Shapley values over the model's raw inputs (interventional, same-source training baseline)",
            "units": "score points (probability x 100); positive values push the fraud score up",
            "baseline_score": base, "baseline_description": f"average model score of {len(bg)} training transactions from {row['Source']}",
            "contributions": items, "additivity_gap": gap, "fixed_context": {"Source": str(row["Source"])}, "background_rows": int(len(bg))}


def _pts(v: float) -> str:
    return f"{abs(v) * 100:.1f}"


def plain_summary(ctx: dict) -> str:
    """Simple-English summary generated from the real prediction and contributions."""
    t, p, e = ctx["transaction"], ctx["prediction"], ctx.get("explanation")
    cls = "Fraud" if p["predicted_fraud"] else "Legitimate"
    rel = "at or above" if p["predicted_fraud"] else "below"
    lines = [f"The model gave this {t['source']} transaction a fraud score of {p['probability'] * 100:.1f}%, "
             f"which is {rel} the decision threshold of {p['threshold'] * 100:.1f}%, so it is classified as {cls}."]
    if not e:
        lines.append("Feature contributions could not be calculated for this transaction, so I cannot say which inputs drove the score.")
    else:
        lines.append(f"For comparison, the average {t['source']} training transaction scores {e['baseline_score'] * 100:.1f}%.")
        ups = [c for c in e["contributions"] if c["contribution"] >= MIN_MATERIAL]
        downs = [c for c in e["contributions"] if c["contribution"] <= -MIN_MATERIAL]
        phrase = lambda c: f"{c['label'].lower()} ({c['display_value']}) {'raised' if c['contribution'] > 0 else 'lowered'} the score by {_pts(c['contribution'])} points"
        if ups:
            lines.append("Pushing the score up: " + "; ".join(phrase(c) for c in ups) + ".")
        if downs:
            lines.append("Pushing the score down: " + "; ".join(phrase(c) for c in downs) + ".")
        if not ups and not downs:
            lines.append("No single input moved the score by more than 1 point compared with the average transaction from this source.")
    a = t.get("actual_label")
    if a is not None:
        outcome = {(1, True): "The historical label also says fraud, so the model's flag matches the record (a true positive).",
                   (0, True): "The historical label says legitimate, so this flag is a false alarm.",
                   (1, False): "The historical label says fraud, but the model did not flag it (a missed fraud case).",
                   (0, False): "The historical label says legitimate and the model agrees."}[(a, p["predicted_fraud"])]
        lines.append(outcome)
    if t.get("in_training_data"):
        lines.append("This transaction was part of the model's training data, so the model has already seen it; its score says less about how the model treats new transactions.")
    lines.append("These contributions describe how the model reached its score, not why fraud happens in the real world, and a flag is a reason to investigate, not proof of fraud.")
    return " ".join(lines)


OUTCOMES = {(1, True): "true_positive", (0, True): "false_positive", (1, False): "false_negative", (0, False): "true_negative"}
UNAVAILABLE = ["customer identity", "account balances", "location", "device", "merchant or receiver", "payment history", "network (IP) data"]


def build_context(df: pd.DataFrame, bundle: dict, metrics: dict, audit: dict | None, tx_id: int) -> dict:
    """Verified evidence for one transaction. Everything the chatbot may say comes from this dict."""
    hit = df[df["tx_id"] == tx_id]
    if hit.empty:
        raise LookupError(f"Transaction {tx_id} not found")
    row = hit.iloc[0]; thr = bundle["threshold"]
    prob = predict_one(bundle, row)                       # real model; model errors propagate to the caller
    pred = bool(prob >= thr); actual = int(row["FraudLabel"])
    ctx = {"transaction": {"tx_id": int(row["tx_id"]), "source": str(row["Source"]), "transaction_type": str(row["TransactionType"]),
                           "amount": float(row["Amount"]), "hour_of_day": int(row["hour_of_day"]), "raw_time": float(row["Time"]),
                           "actual_label": actual, "split": str(row["split"]), "in_training_data": bool(row["split"] == "train")},
           "prediction": {"probability": prob, "threshold": float(thr), "predicted_fraud": pred, "predicted_class": "Fraud" if pred else "Legitimate",
                          "risk_level": "High" if pred else ("Medium" if prob >= 0.6 * thr else "Low"), "margin": prob - thr,
                          "model": bundle["model_name"], "stored_table_score": float(row["risk_score"]), "stored_score_kind": str(row["score_kind"])},
           "outcome": OUTCOMES[(actual, pred)], "explanation": None, "explanation_error": None}
    try:
        ctx["explanation"] = explain_row(df, bundle, row, prob)
    except Exception as e:                                # honest fallback: no fabricated contributions
        ctx["explanation_error"] = f"{type(e).__name__}: {str(e)[:160]}"
    proto = metrics["protocols"][metrics["primary_protocol"]]; sel = proto["models"][metrics["selected_model"]]
    src = ctx["transaction"]["source"]; ps = metrics["per_source"][src]
    proj = next((r for r in metrics["base_rate_projection"] if r["source"] == src), None)
    ctx["metrics"] = {"model": metrics["selected_model"], "protocol": "time-based split: trained on the earlier 80% of each source, tested on the later 20%", "n_test": proto["n_test"],
                      **{k: sel[k] for k in ("precision", "recall", "f1", "roc_auc", "pr_auc")},
                      "selection_rule": metrics["selection_rule"]}
    ctx["source_metrics"] = {"source": src, **{k: ps[k] for k in ("n_test", "fraud_test", "roc_auc", "pr_auc", "recall", "false_positive_rate", "precision")}}
    ctx["all_source_auc"] = {s: {"roc_auc": v["roc_auc"], "n_test": v["n_test"]} for s, v in metrics["per_source"].items()}
    ctx["dataset"] = {"fraud_share": metrics["dataset"]["fraud_rows"] / metrics["dataset"]["rows"], "balanced_by_design": True,
                      "real_base_rates": metrics["dataset"]["real_base_rates"], "projection_for_source": proj, "rows": metrics["dataset"]["rows"]}
    ctx["audit"] = None
    if audit:
        v = audit["variants"]
        ctx["audit"] = {"type_identifies_source": audit["type_identifies_source"], "current_test_auc": v["current"]["test_roc_auc"],
                        "without_source_onehot_auc": v["no_source_onehot"]["test_roc_auc"], "without_any_source_info_auc": v["no_source_info"]["test_roc_auc"],
                        "amount_hour_only_auc": v["amount_hour_only"]["test_roc_auc"]}
    ctx["available_fields"] = list(bundle["raw_features"]); ctx["unavailable_fields"] = UNAVAILABLE
    ctx["summary"] = plain_summary(ctx)
    return ctx
