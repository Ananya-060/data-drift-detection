import json
import unittest

from app.drift import (
    compare_feature,
    kolmogorov_smirnov_statistic,
    population_stability_index,
    severity_for_psi,
)


class DriftMetricsTests(unittest.TestCase):
    def test_identical_distributions_have_no_drift(self) -> None:
        values = list(range(100))
        result = compare_feature(values, values)
        self.assertEqual(result["severity"], "no_drift")
        self.assertAlmostEqual(result["psi"], 0)
        self.assertAlmostEqual(result["ks_statistic"], 0)

    def test_shifted_distribution_is_detected(self) -> None:
        reference = list(range(100))
        current = list(range(500, 600))
        result = compare_feature(reference, current)
        self.assertTrue(result["drift_detected"])
        self.assertEqual(result["severity"], "severe")
        self.assertGreater(result["ks_statistic"], 0.9)

    def test_psi_severity_boundaries(self) -> None:
        self.assertEqual(severity_for_psi(0.099), "no_drift")
        self.assertEqual(severity_for_psi(0.1), "moderate")
        self.assertEqual(severity_for_psi(0.2), "moderate")
        self.assertEqual(severity_for_psi(0.201), "severe")

    def test_constant_reference_detects_a_shift(self) -> None:
        result = compare_feature([5] * 100, [10] * 100)
        self.assertTrue(result["drift_detected"])

    def test_categorical_feature_uses_chi_square(self) -> None:
        reference = ["A"] * 90 + ["B"] * 10
        current = ["A"] * 20 + ["B"] * 80
        result = compare_feature(reference, current, categorical=True)
        self.assertEqual(result["test_name"], "chi_square")
        self.assertGreater(result["chi_square_statistic"], 0)
        self.assertTrue(result["drift_detected"])
        self.assertIs(type(result["drift_detected"]), bool)
        json.dumps(result, allow_nan=False)

    def test_unchanged_categorical_feature_has_no_drift(self) -> None:
        values = ["A", "B"] * 50
        result = compare_feature(values, values, categorical=True)
        self.assertEqual(result["severity"], "no_drift")
        self.assertFalse(result["drift_detected"])

    def test_metrics_require_enough_finite_values(self) -> None:
        with self.assertRaises(ValueError):
            population_stability_index([1], [1])
        with self.assertRaises(ValueError):
            kolmogorov_smirnov_statistic([1, float("nan")], [2, 3])


if __name__ == "__main__":
    unittest.main()
