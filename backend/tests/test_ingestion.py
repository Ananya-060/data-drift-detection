import unittest

from app.drift import compare_feature
from app.ingestion import (
    CSVInputError,
    parse_comparison_csvs,
    parse_feature_csv,
    serialize_feature_csv,
)


class CSVIngestionTests(unittest.TestCase):
    def test_infers_numeric_and_categorical_unlabeled_features(self) -> None:
        contents = (
            b"balance,region\n100.5,north\n,west\n220.0,north\n"
        )
        features, categorical = parse_feature_csv(contents)

        self.assertEqual(categorical, {"region"})
        self.assertEqual(features["balance"].shape, (3,))
        self.assertEqual(features["region"].tolist(), ["north", "west", "north"])

    def test_comparison_requires_matching_feature_names_and_types(self) -> None:
        reference = b"balance,region\n1,north\n2,south\n"
        current = b"region,balance\nsouth,3\nnorth,4\n"
        parsed_reference, parsed_current, categorical = parse_comparison_csvs(
            reference, current
        )
        self.assertEqual(parsed_reference["balance"].tolist(), [1.0, 2.0])
        self.assertEqual(parsed_current["balance"].tolist(), [3.0, 4.0])
        self.assertEqual(categorical, {"region"})

        with self.assertRaisesRegex(CSVInputError, "same feature names"):
            parse_comparison_csvs(reference, b"balance,area\n3,x\n4,y\n")

        with self.assertRaisesRegex(CSVInputError, "same inferred type"):
            parse_comparison_csvs(reference, b"balance,region\n3,1\n4,2\n")

    def test_rejects_csv_without_enough_rows_or_with_duplicate_columns(self) -> None:
        with self.assertRaisesRegex(CSVInputError, "at least two data rows"):
            parse_feature_csv(b"balance\n5\n")
        with self.assertRaisesRegex(CSVInputError, "must be unique"):
            parse_feature_csv(b"balance,balance\n5,6\n7,8\n")

    def test_detects_distribution_change_without_outcome_labels(self) -> None:
        reference, incoming, categorical = parse_comparison_csvs(
            b"income,region\n10,north\n12,south\n14,north\n16,south\n",
            b"income,region\n80,east\n90,east\n100,east\n110,east\n",
        )
        numeric_result = compare_feature(reference["income"], incoming["income"])
        categorical_result = compare_feature(
            reference["region"], incoming["region"], categorical="region" in categorical
        )

        self.assertTrue(numeric_result["drift_detected"])
        self.assertTrue(categorical_result["drift_detected"])

    def test_serializes_feature_arrays_as_uploadable_csv(self) -> None:
        contents = serialize_feature_csv(
            {
                "balance": [10.5, 20.25],
                "region": ["north", "south"],
            }
        )
        features, categorical = parse_feature_csv(contents)

        self.assertEqual(features["balance"].tolist(), [10.5, 20.25])
        self.assertEqual(features["region"].tolist(), ["north", "south"])
        self.assertEqual(categorical, {"region"})


if __name__ == "__main__":
    unittest.main()
