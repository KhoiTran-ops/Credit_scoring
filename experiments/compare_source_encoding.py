r"""Controlled raw-vs-clean encoding comparison on the existing held-out IDs.

Run from the project directory with:
    .\.venv\Scripts\python.exe experiments\compare_source_encoding.py

The raw .xls reader is optional and is not part of the main pipeline; install
it in the project environment with `python -m pip install xlrd==2.0.1` first.
The script changes only EDUCATION/MARRIAGE recoding, fixes the existing split
membership and selected parameters, and writes a diagnostic CSV to results/.
It does not alter the main pipeline or select a seed based on the AUC result.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from credit_scoring import analysis as a


def main():
    raw = pd.read_excel(
        ROOT / "data" / "reference" / "default of credit card clients.xls",
        header=1,
    )
    raw = (
        raw.rename(columns={"default payment next month": a.TARGET})
        [a.MODEL_COLUMNS]
        .sort_values("ID", kind="stable")
        .reset_index(drop=True)
    )
    clean = a.load_clean_data()
    if not raw["ID"].equals(clean["ID"]) or not raw[a.TARGET].equals(clean[a.TARGET]):
        raise AssertionError("Raw and cleaned data IDs/labels do not match")

    _, test = a.load_split_indices(raw)
    is_test = np.zeros(len(raw), dtype=bool)
    is_test[test] = True
    is_train = ~is_test

    params = json.loads((a.OUT / "selected_parameters.json").read_text(encoding="utf-8"))
    rows = []
    cols = [c for c in a.MODEL_COLUMNS if c not in {"ID", "SEX", a.TARGET}]
    for name in a.MODELS:
        for encoding, frame in (("raw_codes", raw), ("merged_codes", clean)):
            X = frame.drop(columns=["ID", "SEX", a.TARGET])
            y = frame[a.TARGET].astype(int).to_numpy()
            model = a.make_model(name, cols, params[name])
            model.fit(X.loc[is_train].reset_index(drop=True), y[is_train])
            probabilities = model.predict_proba(X.loc[is_test].reset_index(drop=True))[:, 1]
            rows.append(
                {
                    "model": name,
                    "encoding": encoding,
                    "n_train": int(is_train.sum()),
                    "n_test": int(is_test.sum()),
                    "test_defaults": int(y[is_test].sum()),
                    "auc": roc_auc_score(y[is_test], probabilities),
                    "log_loss": log_loss(y[is_test], probabilities),
                    "brier": brier_score_loss(y[is_test], probabilities),
                }
            )

    result = pd.DataFrame(rows)
    output = a.OUT / "source_encoding_comparison.csv"
    result.to_csv(output, index=False, encoding="utf-8-sig")
    print(result.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
