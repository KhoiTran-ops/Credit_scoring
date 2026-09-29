r"""Fast smoke run for the three primary models on a stratified data sample.

Run from the project directory with:
    .\.venv\Scripts\python.exe tests\smoke_models.py

This checks data loading, model construction, fitting, and finite predictions.
The sample metrics are diagnostic only and are not comparable to the paper's
full, fixed test set. It intentionally skips tuning, bootstrap, SHAP, and
robustness analyses.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, log_loss
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from credit_scoring import analysis as a


SAMPLE_FRACTION = 0.20
SAMPLE_SEED = 20260929
TEST_FRACTION = 0.20


def main():
    df = a.load_data()
    if not 0 < SAMPLE_FRACTION <= 1:
        raise ValueError("SAMPLE_FRACTION must be in (0, 1]")

    sample, _ = train_test_split(
        df,
        train_size=SAMPLE_FRACTION,
        stratify=df[a.TARGET],
        random_state=SAMPLE_SEED,
    )
    y = sample[a.TARGET].astype(int).to_numpy()
    X = sample.drop(columns=["ID", "SEX", a.TARGET]).reset_index(drop=True)
    y = pd.Series(y).reset_index(drop=True)
    train_ix, test_ix = train_test_split(
        np.arange(len(sample)),
        test_size=TEST_FRACTION,
        stratify=y,
        random_state=SAMPLE_SEED,
    )

    y_train, y_test = y.iloc[train_ix], y.iloc[test_ix]
    print(
        f"Sample: {len(sample)} rows; train: {len(train_ix)}; "
        f"test: {len(test_ix)}; default rate: {y.mean():.4f}",
        flush=True,
    )

    for name in a.MODELS:
        model = a.make_model(name, list(X.columns))
        model.fit(X.iloc[train_ix], y_train)
        probabilities = model.predict_proba(X.iloc[test_ix])[:, 1]
        if not np.isfinite(probabilities).all():
            raise AssertionError(f"{name}: non-finite predictions")
        if ((probabilities < 0) | (probabilities > 1)).any():
            raise AssertionError(f"{name}: predictions outside [0, 1]")
        auc = roc_auc_score(y_test, probabilities)
        loss = log_loss(y_test, probabilities, labels=[0, 1])
        print(f"{name}: AUC={auc:.4f}; log_loss={loss:.4f}", flush=True)

    print("Smoke run passed. These sample metrics do not validate paper-level results.")


if __name__ == "__main__":
    main()
