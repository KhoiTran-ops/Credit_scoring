"""Contract tests for the team's cleaned-data input."""

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from credit_scoring import analysis


class CleanDataSourceTests(unittest.TestCase):
    def test_clean_source_and_generated_data_have_distinct_locations(self):
        root = Path(__file__).resolve().parents[1]
        self.assertEqual(analysis.SOURCE, root / "data" / "cleaned" / "Du_lieu_sach_Chu_de_1.csv")
        self.assertEqual(analysis.DATA, root / "data" / "generated")
        self.assertEqual(analysis.OUT, root / "results")

    def test_clean_csv_maps_to_existing_model_schema(self):
        data = analysis.load_data()

        self.assertEqual(len(data), 30_000)
        self.assertEqual(data["ID"].nunique(), 30_000)
        self.assertEqual(int(data[analysis.TARGET].sum()), 6_636)
        self.assertEqual(data.iloc[0]["PAY_0"], 2)
        self.assertEqual(data.iloc[0]["BILL_AMT1"], 3_913)
        self.assertEqual(data.iloc[0]["PAY_AMT2"], 689)
        self.assertEqual(list(data.columns), analysis.MODEL_COLUMNS)

    def test_missing_clean_csv_does_not_fall_back_to_raw_xls(self):
        with patch.object(analysis, "SOURCE", Path("missing_clean_input.csv")):
            with self.assertRaises(FileNotFoundError):
                analysis.load_data()


if __name__ == "__main__":
    unittest.main()
