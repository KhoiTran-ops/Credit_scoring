r"""One final RF parameter selection on the supplied article development split.

Run from the project directory with:
    .\.venv\Scripts\python.exe experiments\retune_article_random_forest.py

Uses the documented RF grid and the same 5-fold grouped CV recipe as the main
analysis. The article test set is used only once for final evaluation.
"""

import sys
from pathlib import Path

import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold

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
    memberships = pd.read_csv(a.SPLIT_MEMBERSHIP)
    test_ids = set(pd.read_csv(a.ARTICLE_TEST_IDS).ID)
    if test_ids != set(memberships.loc[memberships.split == "test", "ID"]):
        raise AssertionError("Article test ID files do not agree")

    is_test = raw.ID.isin(test_ids).to_numpy()
    is_train = ~is_test
    X = raw.drop(columns=["ID", "SEX", a.TARGET])
    y = raw[a.TARGET].astype(int).to_numpy()
    groups = a.signatures(raw.drop(columns=["ID", a.TARGET]))
    train_groups = groups[is_train]
    cv = StratifiedGroupKFold(
        n_splits=5, shuffle=True, random_state=a.SEED + 20
    )
    search = GridSearchCV(
        a.make_model("Random_Forest", list(X.columns)),
        a.GRIDS["Random_Forest"],
        cv=cv,
        scoring="roc_auc",
        refit=True,
        n_jobs=1,
        error_score="raise",
    )
    search.fit(X.loc[is_train].reset_index(drop=True), y[is_train], groups=train_groups)
    probability = search.best_estimator_.predict_proba(
        X.loc[is_test].reset_index(drop=True)
    )[:, 1]
    article = pd.read_csv(ROOT / "data" / "splits" / "article_test_predictions.csv").set_index("ID")
    article_probability = article.loc[raw.loc[is_test, "ID"], "Random_Forest__none"].to_numpy()
    result = {
        "best_params": search.best_params_,
        "development_cv_auc": float(search.best_score_),
        "article_test_auc": float(roc_auc_score(y[is_test], probability)),
        "article_reference_auc": float(roc_auc_score(y[is_test], article_probability)),
        "auc_difference": float(
            roc_auc_score(y[is_test], probability)
            - roc_auc_score(y[is_test], article_probability)
        ),
        "log_loss": float(log_loss(y[is_test], probability)),
        "brier": float(brier_score_loss(y[is_test], probability)),
    }
    pd.DataFrame([result]).to_json(
        a.OUT / "article_rf_retune_summary.json", orient="records", indent=2
    )
    print(result)


if __name__ == "__main__":
    main()
