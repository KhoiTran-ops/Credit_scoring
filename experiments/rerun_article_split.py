r"""Refit the primary models on the supplied article split, without extras.

Run from the project directory with:
    .\.venv\Scripts\python.exe experiments\rerun_article_split.py

Uses selected_parameters.json and fits raw UCI codes and the team's cleaned
CSV on identical article IDs. Skips nested tuning, bootstrap, SHAP, and
robustness; compare encoding effects conditional on the saved parameters.
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
    if not raw.ID.equals(clean.ID) or not raw[a.TARGET].equals(clean[a.TARGET]):
        raise AssertionError("Raw and clean IDs/labels differ")

    split = pd.read_csv(a.SPLIT_MEMBERSHIP)
    article_test_ids = set(pd.read_csv(a.ARTICLE_TEST_IDS).ID)
    split_test_ids = set(split.loc[split.split == "test", "ID"])
    if article_test_ids != split_test_ids or len(article_test_ids) != 6001:
        raise AssertionError("Article test IDs do not match the supplied split file")
    saved_predictions = pd.read_csv(ROOT / "data" / "splits" / "article_test_predictions.csv").set_index("ID")
    if set(saved_predictions.index) != article_test_ids:
        raise AssertionError("Saved article predictions do not match article test IDs")

    is_test = raw.ID.isin(article_test_ids).to_numpy()
    is_train = ~is_test
    params = json.loads((a.OUT / "selected_parameters.json").read_text(encoding="utf-8"))
    cols = [c for c in a.MODEL_COLUMNS if c not in {"ID", "SEX", a.TARGET}]
    metric_rows = []
    pred_out = pd.DataFrame({"ID": raw.loc[is_test, "ID"].to_numpy()})
    pred_out["y"] = raw.loc[is_test, a.TARGET].to_numpy()

    for model_name in a.MODELS:
        for encoding, frame in (("raw_codes", raw), ("merged_codes", clean)):
            X = frame.drop(columns=["ID", "SEX", a.TARGET])
            y = frame[a.TARGET].astype(int).to_numpy()
            model = a.make_model(model_name, cols, params[model_name])
            model.fit(X.loc[is_train].reset_index(drop=True), y[is_train])
            probability = model.predict_proba(X.loc[is_test].reset_index(drop=True))[:, 1]
            if not np.isfinite(probability).all():
                raise AssertionError(f"{model_name}/{encoding}: non-finite predictions")

            article_probability = saved_predictions.loc[
                pred_out.ID, f"{model_name}__none"
            ].to_numpy()
            metric_rows.append(
                {
                    "model": model_name,
                    "encoding": encoding,
                    "n_train": int(is_train.sum()),
                    "n_test": int(is_test.sum()),
                    "test_defaults": int(y[is_test].sum()),
                    "auc": roc_auc_score(y[is_test], probability),
                    "log_loss": log_loss(y[is_test], probability),
                    "brier": brier_score_loss(y[is_test], probability),
                    "article_auc": roc_auc_score(y[is_test], article_probability),
                    "auc_delta_from_article": roc_auc_score(y[is_test], probability)
                    - roc_auc_score(y[is_test], article_probability),
                    "prediction_mae_from_article": float(
                        np.mean(np.abs(probability - article_probability))
                    ),
                    "prediction_max_abs_diff_from_article": float(
                        np.max(np.abs(probability - article_probability))
                    ),
                }
            )
            pred_out[f"{model_name}__{encoding}"] = probability

    metrics = pd.DataFrame(metric_rows)
    metrics_path = a.OUT / "article_split_rerun_metrics.csv"
    predictions_path = a.OUT / "article_split_rerun_predictions.csv"
    metrics.to_csv(metrics_path, index=False, encoding="utf-8-sig")
    pred_out.to_csv(predictions_path, index=False, encoding="utf-8-sig")
    print(metrics.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print(f"Saved {metrics_path}")
    print(f"Saved {predictions_path}")


if __name__ == "__main__":
    main()
