"""Loads the Mendeley package files, validates them and checks split integrity."""
from __future__ import annotations
from collections import Counter
import pandas as pd
from . import config
from .features import validate, add_hour


def load_package() -> dict:
    missing = [f for f in config.FILES.values() if not (config.PKG_DIR / f).exists()]
    if missing:
        raise FileNotFoundError(f"Missing dataset files in {config.PKG_DIR}: {missing}. "
                                "Copy the Mendeley package CSV files into data/mendeley/.")
    d = {}
    for key, fname in config.FILES.items():
        d[key] = add_hour(validate(pd.read_csv(config.PKG_DIR / fname), fname)).reset_index(drop=True)
    return d


def _keys(df):
    return list(zip(df["Source"], df["Time"].round(4), df["Amount"].round(4)))


def integrity_report(d: dict) -> dict:
    """Facts about the package that matter for honest evaluation."""
    tr_r, te_r, tr_t, te_t, h = d["train_random"], d["test_random"], d["train_time"], d["test_time"], d["harmonized"]
    cols = ["Source", "TransactionType", "Amount", "Time", "FraudLabel"]
    canon = lambda df: df[cols].sort_values(cols).reset_index(drop=True)
    time_ok = canon(pd.concat([tr_t, te_t])).equals(canon(h))
    rand_ok = canon(pd.concat([tr_r, te_r])).equals(canon(h))
    overlap = lambda a, b: len(set(_keys(a)) & set(_keys(b)))
    chrono = {s: bool(tr_t[tr_t.Source == s].Time.max() < te_t[te_t.Source == s].Time.min()) for s in h.Source.unique()}
    return {"time_split_covers_all_rows": bool(time_ok), "random_split_covers_all_rows": bool(rand_ok),
            "duplicate_rows_in_harmonized": int(h.duplicated(cols).sum()),
            "random_split_train_test_key_overlap": int(overlap(tr_r, te_r)),
            "time_split_train_test_key_overlap": int(overlap(tr_t, te_t)),
            "time_split_is_chronological_per_source": chrono}


def random_test_flags(dashboard: pd.DataFrame, te_r: pd.DataFrame) -> list[bool]:
    """Mark which dashboard rows belong to the random-split test file (multiset matching handles duplicates)."""
    cols = ["Source", "TransactionType", "Amount", "Time", "FraudLabel"]
    want = Counter(map(tuple, te_r[cols].round(4).values.tolist()))
    flags = []
    for row in dashboard[cols].round(4).values.tolist():
        k = tuple(row)
        if want[k] > 0:
            want[k] -= 1; flags.append(True)
        else:
            flags.append(False)
    return flags
