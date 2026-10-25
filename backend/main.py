"""FraudLens AI - FastAPI backend.   Run:  uvicorn backend.main:app --port 8000"""
from __future__ import annotations
import io, json
from typing import Literal, Optional
import joblib, numpy as np, pandas as pd
from fastapi import Depends, FastAPI, HTTPException, Path, Query
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, model_validator
from . import config
from . import chatbot
from .explain import build_context
from .features import SOURCES, TYPES_BY_SOURCE, ALL_TYPES

app = FastAPI(title="FraudLens AI", version="2.0.0",
              description="Digital payment fraud detection on the Mendeley harmonized dataset (PaySim + IEEE-CIS + ULB samples).")
STATE: dict = {"df": None, "bundle": None, "metrics": None, "error": None, "audit": None, "ctx_cache": {}}
TABLE_COLS = ["tx_id", "Source", "TransactionType", "Amount", "Time", "hour_of_day", "FraudLabel", "risk_score",
              "predicted_fraud", "split", "score_kind"]
SORTABLE = ["tx_id", "Source", "TransactionType", "Amount", "Time", "hour_of_day", "FraudLabel", "risk_score", "predicted_fraud"]


@app.on_event("startup")
def load_artifacts():
    try:
        missing = [p.name for p in (config.PIPELINE_PATH, config.METRICS_PATH, config.PROCESSED_CSV) if not p.exists()]
        if missing:
            raise FileNotFoundError(f"Missing {missing}. Run:  python -m backend.train")
        STATE["df"] = pd.read_csv(config.PROCESSED_CSV)
        STATE["bundle"] = joblib.load(config.PIPELINE_PATH)
        STATE["metrics"] = json.loads(config.METRICS_PATH.read_text())
        audit_path = config.MODELS_DIR / "source_audit.json"
        STATE["audit"] = json.loads(audit_path.read_text()) if audit_path.exists() else None
    except Exception as e:  # keep serving so /api/health can explain
        STATE["error"] = f"{type(e).__name__}: {e}"


def need_ready():
    if STATE["error"] or STATE["df"] is None:
        raise HTTPException(503, detail=STATE["error"] or "Application not ready. Run: python -m backend.train")
    return STATE["df"]


def primary():
    m = STATE["metrics"]; return m["protocols"][m["primary_protocol"]]["models"][m["selected_model"]]


@app.get("/api/health")
def health():
    ok = STATE["error"] is None and STATE["df"] is not None
    return {"status": "ok" if ok else "error", "error": STATE["error"], "model": STATE["bundle"]["model_name"] if ok else None,
            "rows_loaded": int(len(STATE["df"])) if ok else 0}


@app.get("/api/summary")
def summary():
    df = need_ready(); m = STATE["metrics"]; sel = primary()
    src = df.groupby("Source").agg(count=("FraudLabel", "size"), fraud=("FraudLabel", "sum")).reset_index()
    return {"total_transactions": int(len(df)), "fraud_cases": int(df.FraudLabel.sum()), "fraud_share": float(df.FraudLabel.mean()),
            "balanced_by_design": True, "sources": json.loads(src.to_json(orient="records")),
            "total_amount_by_source": {s: float(df[df.Source == s].Amount.sum()) for s in SOURCES},
            "real_base_rates": m["dataset"]["real_base_rates"],
            "model": {"name": m["selected_model"], "protocol": "Time-based split: trained on the earlier 80% of each source, tested on the later 20%",
                      **{k: sel[k] for k in ("precision", "recall", "f1", "roc_auc", "pr_auc")}}}


# ------------------------------------------------------------ transactions
def filter_df(df, q):
    if q.search:
        s = q.search.strip()
        mask = df["Source"].str.contains(s, case=False, regex=False) | df["TransactionType"].str.contains(s, case=False, regex=False)
        if s.isdigit():
            mask |= df["tx_id"] == int(s)
        df = df[mask]
    if q.source != "ALL": df = df[df.Source == q.source]
    if q.type != "ALL": df = df[df.TransactionType == q.type]
    if q.label == "fraud": df = df[df.FraudLabel == 1]
    elif q.label == "legit": df = df[df.FraudLabel == 0]
    if q.model_flag == "flagged": df = df[df.predicted_fraud == 1]
    elif q.model_flag == "not_flagged": df = df[df.predicted_fraud == 0]
    if q.split != "all": df = df[df.split == q.split]
    if q.min_amount is not None: df = df[df.Amount >= q.min_amount]
    if q.max_amount is not None: df = df[df.Amount <= q.max_amount]
    return df


class TxQuery:
    def __init__(self, search: Optional[str] = Query(None, max_length=40),
                 source: Literal["ALL", "PaySim", "IEEE-CIS", "ULB"] = "ALL",
                 type: str = "ALL",
                 label: Literal["all", "fraud", "legit"] = "all",
                 model_flag: Literal["all", "flagged", "not_flagged"] = "all",
                 split: Literal["all", "train", "test"] = "all",
                 min_amount: Optional[float] = Query(None, ge=0), max_amount: Optional[float] = Query(None, ge=0),
                 sort_by: str = "tx_id", sort_dir: Literal["asc", "desc"] = "asc"):
        if type != "ALL" and type not in ALL_TYPES:
            raise HTTPException(422, f"type must be ALL or one of {ALL_TYPES}")
        if sort_by not in SORTABLE:
            raise HTTPException(422, f"sort_by must be one of {SORTABLE}")
        if min_amount is not None and max_amount is not None and min_amount > max_amount:
            raise HTTPException(422, "min_amount cannot be greater than max_amount")
        self.__dict__.update(locals())


def apply_query(q):
    return filter_df(need_ready(), q).sort_values(q.sort_by, ascending=q.sort_dir == "asc", kind="stable")


@app.get("/api/transactions")
def transactions(q: TxQuery = Depends(), page: int = Query(1, ge=1), page_size: int = Query(25, ge=5, le=200)):
    df = apply_query(q); total = len(df); pages = max(1, -(-total // page_size)); page = min(page, pages)
    rows = df.iloc[(page - 1) * page_size: page * page_size][TABLE_COLS]
    return {"page": page, "page_size": page_size, "total": int(total), "pages": pages, "rows": json.loads(rows.to_json(orient="records"))}


def csv_response(df, name):
    buf = io.StringIO(); df.to_csv(buf, index=False)
    return StreamingResponse(iter([buf.getvalue()]), media_type="text/csv", headers={"Content-Disposition": f'attachment; filename="{name}"'})


@app.get("/api/export/transactions")
def export_transactions(q: TxQuery = Depends()):
    df = apply_query(q)
    if len(df) == 0:
        raise HTTPException(404, "No transactions match the current filters - nothing to export.")
    return csv_response(df[TABLE_COLS], "fraudlens_transactions.csv")


@app.get("/api/export/fraud-cases")
def export_fraud():
    df = need_ready(); return csv_response(df[df.FraudLabel == 1][TABLE_COLS], "fraudlens_confirmed_fraud_cases.csv")


@app.get("/api/export/model-metrics")
def export_metrics():
    need_ready(); m = STATE["metrics"]; rows = []
    for proto, pd_ in m["protocols"].items():
        for n, r in pd_["models"].items():
            cm = r["confusion_matrix"]
            rows.append({"protocol": proto, "model": n, "selected": proto == m["primary_protocol"] and n == m["selected_model"],
                         "threshold": r["threshold"], "precision": r["precision"], "recall": r["recall"], "f1": r["f1"], "roc_auc": r["roc_auc"],
                         "pr_auc": r["pr_auc"], "accuracy": r["accuracy"], "cv_pr_auc": r["cv"]["pr_auc"], **cm})
    return csv_response(pd.DataFrame(rows), "fraudlens_model_comparison.csv")


@app.get("/api/model-metrics")
def model_metrics():
    need_ready(); return STATE["metrics"]


@app.get("/api/dataset-info")
def dataset_info():
    df = need_ready(); m = STATE["metrics"]
    return {"dataset": m["dataset"], "integrity": m["integrity"], "excluded_or_transformed": m["excluded_or_transformed"],
            "model_features": m["model_features"], "raw_features": m["raw_features"], "versions": m["versions"], "trained_at": m["trained_at"],
            "types_by_source": TYPES_BY_SOURCE,
            "split": {s: {"rows": int((df.split == s).sum()), "fraud": int(df[df.split == s].FraudLabel.sum())} for s in ("train", "test")},
            "columns": [{"name": c, "dtype": str(t)} for c, t in df[TABLE_COLS].dtypes.items()],
            "caveats": [
                "The sample is balanced 50/50 BY DESIGN. Real fraud rates are far lower, so precision and accuracy here are not real-world values.",
                "PaySim is simulated mobile-money data, not real UPI. IEEE-CIS is e-commerce card data; ULB is European card data.",
                "Only Time, Amount, TransactionType and Source are available. PaySim balances and card features are not part of this package.",
                "IEEE-CIS 'TransactionType' is the ProductCD product code (C/H/R/S/W), a different concept from PaySim transaction types.",
                "Source is a model input; a production system would know the channel, but this makes the task partly 'which dataset is it?'.",
                "Test sets are small (ULB: ~197 rows), so per-source metrics carry wide uncertainty."]}


# ------------------------------------------------------------ insights
@app.get("/api/insights")
def insights():
    df = need_ready(); m = STATE["metrics"]; ex = lambda d: json.loads(d.to_json(orient="records"))
    by_type = df.groupby(["Source", "TransactionType"]).agg(count=("FraudLabel", "size"), fraud=("FraudLabel", "sum")).reset_index()
    by_type["fraud_rate"] = by_type.fraud / by_type["count"]
    hourly = df.groupby("hour_of_day").agg(count=("FraudLabel", "size"), fraud=("FraudLabel", "sum")).reset_index()
    hourly["fraud_rate"] = hourly.fraud / hourly["count"]
    q = df.assign(band=df.groupby("Source").Amount.transform(lambda s: pd.qcut(s.rank(method="first"), 4, labels=["Q1 lowest", "Q2", "Q3", "Q4 highest"]).astype(str)))
    bands = q.groupby(["Source", "band"]).agg(count=("FraudLabel", "size"), fraud=("FraudLabel", "sum")).reset_index()
    bands["fraud_rate"] = bands.fraud / bands["count"]
    med = df.groupby(["Source", "FraudLabel"]).Amount.median().unstack()
    ps = df[df.Source == "PaySim"]
    return {"by_source": ex(df.groupby("Source").agg(count=("FraudLabel", "size"), fraud=("FraudLabel", "sum")).reset_index()),
            "by_type": ex(by_type), "hourly": ex(hourly), "amount_bands": ex(bands), "feature_importance": m["feature_importance"],
            "median_amount": {s: {"legit": float(med.loc[s, 0]), "fraud": float(med.loc[s, 1])} for s in med.index},
            "paysim_zero_fraud_types": sorted(t for t in ps.TransactionType.unique() if ps[ps.TransactionType == t].FraudLabel.sum() == 0),
            "paysim_transfer_fraud_rate": float(ps[ps.TransactionType == "TRANSFER"].FraudLabel.mean())}


# ------------------------------------------------------------ prediction
class PredictIn(BaseModel):
    source: Literal["PaySim", "IEEE-CIS", "ULB"]
    transaction_type: str
    amount: float = Field(ge=0, le=1e9)
    hour_of_day: int = Field(ge=0, le=23)

    @model_validator(mode="after")
    def check(self):
        if self.transaction_type not in TYPES_BY_SOURCE[self.source]:
            raise ValueError(f"transaction_type for {self.source} must be one of {TYPES_BY_SOURCE[self.source]}")
        return self


def _score(rows):
    b = STATE["bundle"]
    df = pd.DataFrame([{"Source": r["source"], "TransactionType": r["transaction_type"], "Amount": r["amount"], "hour_of_day": r["hour_of_day"]} for r in rows])
    return b["pipeline"].predict_proba(df[b["raw_features"]])[:, 1]


@app.post("/api/predict")
def predict(body: PredictIn):
    if STATE["bundle"] is None:
        raise HTTPException(503, STATE["error"] or "Model not loaded. Run: python -m backend.train")
    b = STATE["bundle"]; thr = b["threshold"]; base = b["baseline"][body.source]; row = body.model_dump()
    prob = float(_score([row])[0])
    fields = [("transaction_type", "TransactionType"), ("amount", "Amount"), ("hour_of_day", "hour_of_day")]
    pv = _score([{**row, k: base[bk]} for k, bk in fields])
    infl = sorted([{"feature": k, "influence": float(prob - p)} for (k, _), p in zip(fields, pv)], key=lambda d: -abs(d["influence"]))
    level = "High" if prob >= thr else ("Medium" if prob >= thr * 0.6 else "Low")
    notes = ["The training sample is balanced 50/50, so this probability is not a real-world fraud probability; use it for ranking and comparison."]
    if body.source == "PaySim" and body.transaction_type in ("CASH_IN", "PAYMENT", "DEBIT"):
        notes.append("In this PaySim sample, CASH_IN, PAYMENT and DEBIT transactions are never fraud.")
    if body.source == "ULB":
        notes.append("ULB has only amount and time available here, so predictions for this source are weakly informative (test AUC about 0.65).")
    return {"fraud_probability": prob, "is_fraud": prob >= thr, "risk_level": level, "threshold": thr, "model": b["model_name"],
            "influences": infl, "baseline_used": base, "notes": notes,
            "disclaimer": "Educational model trained on a small balanced sample of public datasets; not a real fraud decision."}


@app.get("/api/examples")
def examples():
    """Real held-out test transactions to pre-fill the prediction form."""
    df = need_ready(); t = df[df.split == "test"]
    cols = ["Source", "TransactionType", "Amount", "hour_of_day", "risk_score", "FraudLabel"]
    def pick(mask, n=3):
        s = t[mask].sample(min(n, int(mask.sum())), random_state=7)
        return [{"source": r.Source, "transaction_type": r.TransactionType, "amount": r.Amount, "hour_of_day": int(r.hour_of_day),
                 "risk_score": r.risk_score, "FraudLabel": int(r.FraudLabel)} for r in s[cols].itertuples()]
    return {"fraud_caught": pick((t.FraudLabel == 1) & (t.predicted_fraud == 1)), "fraud_missed": pick((t.FraudLabel == 1) & (t.predicted_fraud == 0), 2),
            "legit_clear": pick((t.FraudLabel == 0) & (t.predicted_fraud == 0)), "false_alarm": pick((t.FraudLabel == 0) & (t.predicted_fraud == 1), 2)}



# ------------------------------------------------------------ AI explainability + chatbot
def get_context(tx_id: int) -> dict:
    """Verified evidence for one transaction (real model score + exact feature contributions)."""
    df = need_ready()
    if STATE["bundle"] is None:
        raise HTTPException(503, STATE["error"] or "Model not loaded. Run: python -m backend.train")
    cache = STATE["ctx_cache"]
    if tx_id in cache:
        return cache[tx_id]
    try:
        ctx = build_context(df, STATE["bundle"], STATE["metrics"], STATE["audit"], tx_id)
    except LookupError:
        raise HTTPException(404, f"Transaction {tx_id} was not found (valid IDs are 1 to {int(df.tx_id.max())}).")
    except Exception:
        raise HTTPException(503, "The model could not score this transaction. Check that the model files match the installed scikit-learn (run: python -m backend.train).")
    if len(cache) > 512:
        cache.clear()
    cache[tx_id] = ctx
    return ctx


@app.get("/api/explain/{tx_id}")
def explain_transaction(tx_id: int = Path(ge=1, le=10_000_000)):
    """Prediction, decision threshold and exact per-transaction feature contributions for one dataset transaction."""
    return get_context(tx_id)


class ChatTurn(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(max_length=1500)


class ChatIn(BaseModel):
    tx_id: int = Field(ge=1, le=10_000_000)
    message: str = Field(min_length=1, max_length=500)
    history: list[ChatTurn] = Field(default_factory=list, max_length=8)

    @model_validator(mode="after")
    def clean(self):
        self.message = self.message.strip()
        if not self.message:
            raise ValueError("message cannot be empty")
        return self


@app.get("/api/chat/status")
def chat_status():
    return chatbot.llm_status()


@app.post("/api/chat")
def chat(body: ChatIn):
    ctx = get_context(body.tx_id)                       # context is rebuilt from the requested transaction, never from the client
    try:
        res = chatbot.respond(body.message, ctx, [h.model_dump() for h in body.history])
    except Exception:
        raise HTTPException(500, "The assistant could not produce an answer. Please try again.")
    grounded = ["transaction", "model prediction", "decision threshold"] + (["feature contributions"] if ctx["explanation"] else []) + ["evaluation metrics"]
    return {"tx_id": body.tx_id, **res, "grounded_in": grounded, "explanation_available": ctx["explanation"] is not None}


app.mount("/static", StaticFiles(directory=config.FRONTEND_DIR), name="static")


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(config.FRONTEND_DIR / "index.html")
