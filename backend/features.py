"""Schema, validation and feature engineering shared by training and the API.

The harmonized dataset has only five columns: Time, Amount, TransactionType, FraudLabel, Source.
`Time` has a different unit per source (PaySim: hours/steps, ULB and IEEE-CIS: seconds), so the raw
value is NOT comparable across sources. We convert it to hour-of-day, which means the same thing everywhere.
"""
import numpy as np
import pandas as pd

COLS = ["Time", "Amount", "TransactionType", "FraudLabel", "Source"]
SOURCES = ["PaySim", "IEEE-CIS", "ULB"]
TYPES_BY_SOURCE = {"PaySim": ["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"],
                   "IEEE-CIS": ["C", "H", "R", "S", "W"],           # ProductCD product codes
                   "ULB": ["CardTransaction"]}
ALL_TYPES = [t for s in SOURCES for t in TYPES_BY_SOURCE[s]]
RAW_FEATURES = ["Source", "TransactionType", "Amount", "hour_of_day"]
CATEGORICAL = ["Source", "TransactionType"]
NUMERIC = ["log_amount", "amount_z", "hour_of_day"]
MODEL_FEATURES = CATEGORICAL + NUMERIC


def validate(df: pd.DataFrame, name: str) -> pd.DataFrame:
    missing = [c for c in COLS if c not in df.columns]
    if missing:
        raise ValueError(f"{name}: missing columns {missing}; expected {COLS}")
    if df[COLS].isna().any().any():
        raise ValueError(f"{name}: contains missing values")
    if (df["Amount"] < 0).any():
        raise ValueError(f"{name}: negative amounts")
    if not set(df["FraudLabel"].unique()) <= {0, 1}:
        raise ValueError(f"{name}: FraudLabel must be 0/1")
    bad = ~df["Source"].isin(SOURCES)
    if bad.any():
        raise ValueError(f"{name}: unknown Source values {sorted(df.loc[bad, 'Source'].unique())}")
    for s in SOURCES:
        m = df["Source"] == s
        wrong = ~df.loc[m, "TransactionType"].isin(TYPES_BY_SOURCE[s])
        if wrong.any():
            raise ValueError(f"{name}: invalid TransactionType for {s}: {sorted(df.loc[m, 'TransactionType'][wrong].unique())}")
    return df


def add_hour(df: pd.DataFrame) -> pd.DataFrame:
    """hour_of_day from the source-specific Time unit."""
    out = df.copy()
    out["hour_of_day"] = np.where(out["Source"] == "PaySim", out["Time"] % 24, (out["Time"] // 3600) % 24).astype(int)
    return out


def engineer(df: pd.DataFrame, stats: dict | None = None) -> pd.DataFrame:
    """Raw inputs (Source, TransactionType, Amount, hour_of_day) -> model features."""
    out = pd.DataFrame(index=df.index)
    out["Source"] = df["Source"].astype(str)
    out["TransactionType"] = df["TransactionType"].astype(str)
    amt = pd.to_numeric(df["Amount"], errors="coerce").astype(float).clip(lower=0)
    out["log_amount"] = np.log1p(amt)
    # how unusual the amount is *for that source* (amount scales differ by orders of magnitude)
    mu = df["Source"].map({s: v[0] for s, v in (stats or {}).items()}).fillna(out["log_amount"].mean())
    sd = df["Source"].map({s: v[1] for s, v in (stats or {}).items()}).fillna(1.0)
    out["amount_z"] = (out["log_amount"] - mu) / sd.replace(0, 1.0)
    out["hour_of_day"] = pd.to_numeric(df["hour_of_day"], errors="coerce").astype(float)
    return out[MODEL_FEATURES]
