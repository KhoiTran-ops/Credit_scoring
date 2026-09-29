"""Input and persisted article-split contract tests."""

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from credit_scoring import analysis


class ArticleDataSourceTests(unittest.TestCase):
    def test_raw_clean_and_generated_data_have_distinct_locations(self):
        root = Path(__file__).resolve().parents[1]
        self.assertEqual(analysis.SOURCE, root / "data" / "reference" / "default of credit card clients.xls")
        self.assertEqual(analysis.CLEAN_SOURCE, root / "data" / "cleaned" / "Du_lieu_sach_Chu_de_1.csv")
        self.assertEqual(analysis.SPLIT_MEMBERSHIP, root / "data" / "splits" / "article_split_membership.csv")
        self.assertEqual(analysis.DATA, root / "data" / "generated")
        self.assertEqual(analysis.OUT, root / "results")

    def test_uci_source_preserves_original_codes_and_label(self):
        data = analysis.load_data()

        self.assertEqual(len(data), 30_000)
        self.assertEqual(data["ID"].nunique(), 30_000)
        self.assertEqual(int(data[analysis.TARGET].sum()), 6_636)
        self.assertEqual(data.iloc[0]["PAY_0"], 2)
        self.assertEqual(data.iloc[0]["BILL_AMT1"], 3_913)
        self.assertEqual(data.iloc[0]["PAY_AMT2"], 689)
        self.assertEqual(int((data["EDUCATION"] == 0).sum()), 14)
        self.assertEqual(int((data["EDUCATION"] == 5).sum()), 280)
        self.assertEqual(int((data["MARRIAGE"] == 0).sum()), 54)
        self.assertEqual(list(data.columns), analysis.MODEL_COLUMNS)

    def test_clean_csv_remains_available_as_separate_sensitivity_input(self):
        data = analysis.load_clean_data()
        self.assertEqual(len(data), 30_000)
        self.assertEqual(int(data[analysis.TARGET].sum()), 6_636)
        self.assertEqual(set(data["EDUCATION"].unique()), {1, 2, 3, 4})

    def test_missing_uci_workbook_fails_explicitly(self):
        with patch.object(analysis, "SOURCE", Path("missing_clean_input.csv")):
            with self.assertRaises(FileNotFoundError):
                analysis.load_data()

    def test_saved_article_split_has_expected_counts_and_no_group_leakage(self):
        data = analysis.load_data()
        dev, test = analysis.load_split_indices(data)
        self.assertEqual((len(dev), len(test)), (23_999, 6_001))
        self.assertEqual((int(data.iloc[dev][analysis.TARGET].sum()), int(data.iloc[test][analysis.TARGET].sum())), (5_309, 1_327))
        groups = analysis.signatures(data.drop(columns=["ID", analysis.TARGET]))
        self.assertFalse(set(groups[dev]) & set(groups[test]))

    def test_article_reference_predictions_reproduce_reported_auc(self):
        root = Path(__file__).resolve().parents[1]
        reference = pd.read_csv(root / "data" / "splits" / "article_test_predictions.csv")
        self.assertEqual(len(reference), 6_001)
        self.assertEqual(reference["ID"].nunique(), 6_001)
        expected = {
            "Logistic_WoE__none": 0.7703,
            "Random_Forest__none": 0.7827,
            "XGBoost__none": 0.7840,
        }
        for column, auc in expected.items():
            self.assertAlmostEqual(roc_auc_score(reference["y"], reference[column]), auc, places=4)


if __name__ == "__main__":
    unittest.main()
