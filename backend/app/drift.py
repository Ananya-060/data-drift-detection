import numpy as np
from numpy.typing import ArrayLike
from scipy.stats import chi2_contingency, ks_2samp


def _finite_values(values: ArrayLike, name: str) -> np.ndarray:
    data = np.asarray(values, dtype=float)
    data = data[np.isfinite(data)]
    if data.size < 2:
        raise ValueError(f"{name} must contain at least two finite values")
    return data


def population_stability_index(
    reference: ArrayLike, current: ArrayLike, bins: int = 10
) -> float:
    ref = _finite_values(reference, "reference")
    cur = _finite_values(current, "current")
    if bins < 2:
        raise ValueError("bins must be at least 2")

    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if edges.size < 2:
        edges = np.array(
            [
                -np.inf,
                np.nextafter(ref[0], -np.inf),
                np.nextafter(ref[0], np.inf),
                np.inf,
            ]
        )
    else:
        edges[0] = -np.inf
        edges[-1] = np.inf

    ref_counts, _ = np.histogram(ref, bins=edges)
    cur_counts, _ = np.histogram(cur, bins=edges)
    ref_pct = np.maximum(ref_counts / ref.size, 1e-6)
    cur_pct = np.maximum(cur_counts / cur.size, 1e-6)
    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def kolmogorov_smirnov_statistic(
    reference: ArrayLike, current: ArrayLike
) -> float:
    ref = np.sort(_finite_values(reference, "reference"))
    cur = np.sort(_finite_values(current, "current"))
    values = np.sort(np.concatenate((ref, cur)))
    ref_cdf = np.searchsorted(ref, values, side="right") / ref.size
    cur_cdf = np.searchsorted(cur, values, side="right") / cur.size
    return float(np.max(np.abs(ref_cdf - cur_cdf)))


def severity_for_psi(psi: float) -> str:
    if psi < 0.1:
        return "no_drift"
    if psi <= 0.2:
        return "moderate"
    return "severe"


def compare_feature(
    reference: ArrayLike, current: ArrayLike, categorical: bool = False
) -> dict[str, float | str | bool]:
    if categorical:
        ref = np.asarray(reference, dtype=str)
        cur = np.asarray(current, dtype=str)
        ref = ref[np.char.str_len(ref) > 0]
        cur = cur[np.char.str_len(cur) > 0]
        if ref.size < 2 or cur.size < 2:
            raise ValueError("Both categorical samples must contain two values")
        categories = np.unique(np.concatenate((ref, cur)))
        ref_counts = np.asarray([(ref == value).sum() for value in categories])
        cur_counts = np.asarray([(cur == value).sum() for value in categories])
        ref_pct = np.maximum(ref_counts / ref.size, 1e-6)
        cur_pct = np.maximum(cur_counts / cur.size, 1e-6)
        psi = float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))
        if categories.size < 2:
            statistic, p_value = 0.0, 1.0
        else:
            statistic, p_value, _, _ = chi2_contingency(
                np.vstack((ref_counts, cur_counts)), correction=False
            )
        severity = severity_for_psi(psi)
        detected = bool(severity != "no_drift" and p_value < 0.05)
        return {
            "psi": round(psi, 6),
            "chi_square_statistic": round(float(statistic), 6),
            "test_statistic": round(float(statistic), 6),
            "test_name": "chi_square",
            "p_value": round(float(p_value), 8),
            "severity": severity if detected else "no_drift",
            "drift_detected": detected,
        }

    psi = population_stability_index(reference, current)
    ks = kolmogorov_smirnov_statistic(reference, current)
    test = ks_2samp(
        _finite_values(reference, "reference"),
        _finite_values(current, "current"),
    )
    severity = severity_for_psi(psi)
    return {
        "psi": round(psi, 6),
        "ks_statistic": round(ks, 6),
        "test_statistic": round(ks, 6),
        "test_name": "kolmogorov_smirnov",
        "p_value": round(float(test.pvalue), 8),
        "severity": severity,
        "drift_detected": bool(severity != "no_drift"),
    }
