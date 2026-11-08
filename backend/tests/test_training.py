import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np

from app.drift import compare_feature
from app.training import (
    CSV_FEATURE_COLUMNS,
    EVALUATION_COHORTS,
    DatasetError,
    dataset_drift_samples,
    load_temporal_cohorts,
    monitor_loan_batch,
    score_feature_batch,
    simulated_incoming_samples,
    simulated_loan_csv,
    train_model,
)


class TrainingPipelineTests(unittest.TestCase):
    def write_dataset(self, path: Path) -> None:
        with path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=[*CSV_FEATURE_COLUMNS, "loan_status"])
            writer.writeheader()
            cohorts = [(2013, "Jan", 120)] + [
                (year, month, 40)
                for year in (2014, 2015)
                for month in ("Jan", "Apr", "Jul", "Oct")
            ]
            for year, month, row_count in cohorts:
                for index in range(row_count):
                    writer.writerow(
                        {
                            "loan_amnt": 5000 + index * 10,
                            "term": " 36 months",
                            "int_rate": 8 + index % 15,
                            "installment": 100 + index,
                            "emp_length": "" if index % 4 == 0 else "5 years",
                            "annual_inc": 40000 + index * 100,
                            "dti": 5 + index % 25,
                            "delinq_2yrs": index % 2,
                            "earliest_cr_line": "Jan-2000",
                            "issue_d": f"{month}-{year}",
                            "fico_range_low": 660 + index % 100,
                            "fico_range_high": 664 + index % 100,
                            "inq_last_6mths": index % 4,
                            "open_acc": 3 + index % 10,
                            "pub_rec": index % 2,
                            "revol_bal": 1000 + index * 50,
                            "revol_util": 5 + index % 90,
                            "total_acc": 5 + index % 30,
                            "grade": "A" if index % 2 else "B",
                            "home_ownership": "RENT" if index % 2 else "MORTGAGE",
                            "verification_status": "Verified" if index % 2 else "Not Verified",
                            "purpose": "debt_consolidation" if index % 2 else "credit_card",
                            "loan_status": "Charged Off" if index % 3 == 0 else "Fully Paid",
                        }
                    )

    def test_loads_chronological_resolved_cohorts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "accepted_2007_to_2018Q4.csv"
            self.write_dataset(path)
            train_numeric, train_categorical, labels, evaluations, counts = (
                load_temporal_cohorts(str(path))
            )

        self.assertEqual(train_numeric.shape, (120, 16))
        self.assertEqual(train_categorical.shape, (120, 4))
        self.assertEqual(labels.sum(), 40)
        self.assertEqual(evaluations["2014-Q1"][2].size, 40)
        self.assertEqual(evaluations["2015-Q4"][2].size, 40)
        self.assertEqual(counts["train"]["issue_years"], "2007-2013")
        self.assertEqual(len(counts["evaluations"]), 8)

    def test_training_saves_model_and_time_based_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "accepted_2007_to_2018Q4.csv"
            model_path = Path(directory) / "model.joblib"
            self.write_dataset(path)

            metadata = train_model(str(path), str(model_path))
            references, evaluation = dataset_drift_samples(
                "2015-Q4", str(model_path)
            )

            self.assertTrue(model_path.is_file())
            self.assertEqual(metadata["training_period"], "2007-2013")
            self.assertEqual(metadata["metrics_by_cohort"]["2015-Q4"]["rows"], 40)
            self.assertGreaterEqual(
                metadata["metrics_by_cohort"]["2015-Q4"]["roc_auc"], 0
            )
            self.assertEqual(metadata["evaluation_cohorts"], list(EVALUATION_COHORTS))
            self.assertIn("loan_amnt", references)
            self.assertEqual(len(references["loan_amnt"]), 120)
            self.assertEqual(len(evaluation["loan_amnt"]), 40)
            self.assertEqual(len(metadata["metrics_by_cohort"]), 8)

    def test_simulated_incoming_batch_uses_reference_and_injects_drift(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "accepted_2007_to_2018Q4.csv"
            model_path = Path(directory) / "model.joblib"
            self.write_dataset(path)
            train_model(str(path), str(model_path))

            reference, incoming = simulated_incoming_samples(
                sample_size=200, seed=7, model_path=str(model_path)
            )
            risk_summary = score_feature_batch(incoming, model_path=str(model_path))

        self.assertEqual(len(incoming["annual_inc"]), 200)
        self.assertEqual(risk_summary["rows_scored"], 200)
        self.assertGreaterEqual(
            risk_summary["mean_predicted_default_probability"], 0
        )
        self.assertTrue(
            all(
                np.isfinite(values).all()
                for values in reference.values()
                if values.dtype.kind in "iuf"
            )
        )
        self.assertTrue(
            (incoming["purpose"] == "simulated_new_category").any()
        )
        self.assertTrue(
            compare_feature(reference["annual_inc"], incoming["annual_inc"])[
                "drift_detected"
            ]
        )
        self.assertEqual(
            compare_feature(reference["purpose"], incoming["purpose"], categorical=True)[
                "test_name"
            ],
            "chi_square",
        )

    def test_uploaded_loan_batch_scores_and_evaluates_resolved_outcomes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "accepted_2007_to_2018Q4.csv"
            model_path = Path(directory) / "model.joblib"
            self.write_dataset(path)
            train_model(str(path), str(model_path))

            rows = []
            with path.open(newline="", encoding="utf-8") as file:
                reader = csv.DictReader(file)
                for row in list(reader)[-40:]:
                    row["issue_d"] = "Oct-2015"
                    rows.append(row)
            contents_file = Path(directory) / "incoming.csv"
            with contents_file.open("w", newline="", encoding="utf-8") as file:
                writer = csv.DictWriter(file, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)

            reference, current, summary = monitor_loan_batch(
                contents_file.read_bytes(), str(model_path)
            )

        self.assertEqual(len(reference), len(current))
        self.assertEqual(summary["rows_scored"], 40)
        self.assertEqual(summary["labeled_rows"], 40)
        self.assertIsNotNone(summary["evaluation"])
        self.assertEqual(summary["evaluation"]["resolved_rows"], 40)
        self.assertGreaterEqual(summary["mean_predicted_default_probability"], 0)

    def test_simulated_loan_csv_uses_the_uploaded_loan_format(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "accepted_2007_to_2018Q4.csv"
            model_path = Path(directory) / "model.joblib"
            self.write_dataset(path)
            train_model(str(path), str(model_path))

            contents = simulated_loan_csv(
                sample_size=100,
                seed=7,
                model_path=str(model_path),
                batch="incoming",
            )
            reference_contents = simulated_loan_csv(
                sample_size=100,
                seed=7,
                model_path=str(model_path),
                batch="reference",
            )
            rows = list(csv.DictReader(contents.decode("utf-8").splitlines()))
            reference_rows = list(
                csv.DictReader(reference_contents.decode("utf-8").splitlines())
            )
            reference, current, summary = monitor_loan_batch(
                contents,
                str(model_path),
                reference_contents=reference_contents,
            )

        self.assertEqual(len(rows), 100)
        self.assertEqual(len(reference_rows), 100)
        self.assertEqual(rows[0]["term"], " 36 months")
        self.assertEqual(rows[0]["loan_status"], "")
        self.assertEqual(summary["rows_scored"], 100)
        self.assertEqual(summary["labeled_rows"], 0)
        self.assertEqual(len(reference["loan_amnt"]), 100)
        self.assertEqual(len(current["loan_amnt"]), 100)
        self.assertTrue(
            compare_feature(reference["annual_inc"], current["annual_inc"])[
                "drift_detected"
            ]
        )

    def test_uploaded_loan_batch_rejects_non_lendingclub_csv(self) -> None:
        with self.assertRaisesRegex(DatasetError, "missing required columns"):
            monitor_loan_batch(b"amount,purpose\n100,car\n200,home\n")

    def test_dataset_drift_rejects_unsupported_cohort(self) -> None:
        with self.assertRaisesRegex(DatasetError, "Choose a supported quarter cohort"):
            dataset_drift_samples("2016-Q1", "missing-model.joblib")

    def test_missing_required_column_fails_clearly(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "incomplete.csv"
            path.write_text("loan_status,issue_d\nFully Paid,Jan-2011\n", encoding="utf-8")
            with self.assertRaisesRegex(DatasetError, "missing required columns"):
                load_temporal_cohorts(str(path))


if __name__ == "__main__":
    unittest.main()
