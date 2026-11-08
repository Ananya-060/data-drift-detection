import csv
import io
import os
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import joblib
import numpy as np
from numpy.typing import NDArray
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

from app.config import DATASET_PATH, MODEL_PATH


TRAIN_END_YEAR = 2013
EVALUATION_COHORTS = tuple(
    f"{year}-Q{quarter}"
    for year in (2014, 2015)
    for quarter in range(1, 5)
)
POSITIVE_STATUSES = {"Charged Off", "Default"}
NEGATIVE_STATUSES = {"Fully Paid"}
NUMERIC_FEATURES = (
    "loan_amnt",
    "int_rate",
    "installment",
    "emp_length_years",
    "annual_inc",
    "dti",
    "delinq_2yrs",
    "credit_history_months",
    "fico_range_low",
    "fico_range_high",
    "inq_last_6mths",
    "open_acc",
    "pub_rec",
    "revol_bal",
    "revol_util",
    "total_acc",
)
CATEGORICAL_FEATURES = (
    "grade",
    "home_ownership",
    "verification_status",
    "purpose",
)
FEATURE_NAMES = (*NUMERIC_FEATURES, *CATEGORICAL_FEATURES)
CSV_FEATURE_COLUMNS = (
    "loan_amnt",
    "term",
    "int_rate",
    "installment",
    "emp_length",
    "annual_inc",
    "dti",
    "delinq_2yrs",
    "earliest_cr_line",
    "issue_d",
    "fico_range_low",
    "fico_range_high",
    "inq_last_6mths",
    "open_acc",
    "pub_rec",
    "revol_bal",
    "revol_util",
    "total_acc",
    "grade",
    "home_ownership",
    "verification_status",
    "purpose",
)
REQUIRED_COLUMNS = (*CSV_FEATURE_COLUMNS, "loan_status")
CSV_PATH = Path(DATASET_PATH)
MODEL_FILE = Path(MODEL_PATH)


class DatasetError(ValueError):
    pass


def _issue_date(value: str) -> datetime | None:
    value = value.strip()
    for date_format in ("%b-%Y", "%b-%y"):
        try:
            return datetime.strptime(value, date_format)
        except ValueError:
            continue
    return None


def _number(value: str) -> float:
    try:
        result = float(value.strip().replace("%", ""))
    except (ValueError, AttributeError):
        return float("nan")
    return result if np.isfinite(result) else float("nan")


def _term_months(value: str) -> float:
    match = re.search(r"\d+", value)
    return float(match.group()) if match else float("nan")


def _employment_years(value: str) -> float:
    value = value.strip().lower()
    if value in {"", "n/a"}:
        return float("nan")
    if value == "< 1 year":
        return 0.5
    match = re.search(r"\d+", value)
    return float(match.group()) if match else float("nan")


def _loan_features(row: dict[str, str], issue_date: datetime) -> list[object]:
    numeric = [
        _number(row["loan_amnt"]),
        _number(row["int_rate"]),
        _number(row["installment"]),
        _employment_years(row["emp_length"]),
        _number(row["annual_inc"]),
        _number(row["dti"]),
        _number(row["delinq_2yrs"]),
    ]
    credit_line = _issue_date(row["earliest_cr_line"])
    if credit_line is not None and credit_line > issue_date:
        credit_line = credit_line.replace(year=credit_line.year - 100)
    credit_history_months = (
        (issue_date.year - credit_line.year) * 12
        + issue_date.month
        - credit_line.month
        if credit_line is not None
        else float("nan")
    )
    numeric.append(float(credit_history_months))
    numeric.extend(
        _number(row[name])
        for name in (
            "fico_range_low",
            "fico_range_high",
            "inq_last_6mths",
            "open_acc",
            "pub_rec",
            "revol_bal",
            "revol_util",
            "total_acc",
        )
    )
    categorical = [
        row[name].strip() or "Unknown"
        for name in CATEGORICAL_FEATURES
    ]
    return [*numeric, *categorical]


def dataset_status() -> dict[str, Any]:
    if not CSV_PATH.is_file():
        return {
            "available": False,
            "file_name": CSV_PATH.name,
            "size_bytes": 0,
            "missing_columns": list(REQUIRED_COLUMNS),
            "train_end_year": TRAIN_END_YEAR,
            "evaluation_cohorts": list(EVALUATION_COHORTS),
        }
    try:
        with CSV_PATH.open(newline="", encoding="utf-8-sig") as file:
            header = next(csv.reader(file), [])
    except (OSError, UnicodeError, csv.Error) as exc:
        raise DatasetError(f"Unable to read the dataset header: {exc}") from exc
    missing_columns = sorted(set(REQUIRED_COLUMNS).difference(header))
    return {
        "available": not missing_columns,
        "file_name": CSV_PATH.name,
        "size_bytes": CSV_PATH.stat().st_size,
        "missing_columns": missing_columns,
        "train_end_year": TRAIN_END_YEAR,
        "evaluation_cohorts": list(EVALUATION_COHORTS),
    }


def load_temporal_cohorts(
    path: str = DATASET_PATH,
) -> tuple[
    NDArray[np.float64],
    NDArray[np.str_],
    NDArray[np.int8],
    dict[str, tuple[NDArray[np.float64], NDArray[np.str_], NDArray[np.int8]]],
    dict[str, Any],
]:
    numeric_columns: list[list[float]] = [[] for _ in NUMERIC_FEATURES]
    categorical_columns: list[list[str]] = [[] for _ in CATEGORICAL_FEATURES]
    training_labels: list[int] = []
    evaluation_numeric = {
        cohort: [[] for _ in NUMERIC_FEATURES] for cohort in EVALUATION_COHORTS
    }
    evaluation_categorical = {
        cohort: [[] for _ in CATEGORICAL_FEATURES] for cohort in EVALUATION_COHORTS
    }
    evaluation_labels = {cohort: [] for cohort in EVALUATION_COHORTS}
    row_counts: Counter[int | str] = Counter()
    status_counts: Counter[str] = Counter()
    outcome_counts: dict[int | str, Counter[str]] = {
        key: Counter()
        for key in (*range(2007, TRAIN_END_YEAR + 1), *EVALUATION_COHORTS)
    }

    try:
        with open(path, newline="", encoding="utf-8-sig") as file:
            reader = csv.DictReader(file)
            header = reader.fieldnames or []
            missing = sorted(set(REQUIRED_COLUMNS).difference(header))
            if missing:
                raise DatasetError(
                    "Dataset is missing required columns: " + ", ".join(missing)
                )
            for row_number, row in enumerate(reader, start=2):
                issue_date = _issue_date(row["issue_d"])
                if issue_date is None:
                    continue
                year = issue_date.year
                if year > 2015:
                    continue
                if _term_months(row["term"]) != 36:
                    continue
                cohort = f"{year}-Q{(issue_date.month - 1) // 3 + 1}"
                cohort_key: int | str = year if year <= TRAIN_END_YEAR else cohort
                if year > TRAIN_END_YEAR and cohort not in EVALUATION_COHORTS:
                    continue
                row_counts[cohort_key] += 1
                status = row["loan_status"].strip()
                status_counts[status] += 1
                if cohort_key in outcome_counts:
                    outcome_counts[cohort_key][status] += 1

                if status in NEGATIVE_STATUSES:
                    label = 0
                elif status in POSITIVE_STATUSES:
                    label = 1
                else:
                    continue

                if year <= TRAIN_END_YEAR:
                    numeric_target = numeric_columns
                    categorical_target = categorical_columns
                    labels_target = training_labels
                elif cohort in EVALUATION_COHORTS:
                    numeric_target = evaluation_numeric[cohort]
                    categorical_target = evaluation_categorical[cohort]
                    labels_target = evaluation_labels[cohort]
                else:
                    continue

                features = _loan_features(row, issue_date)
                for index, value in enumerate(features[: len(NUMERIC_FEATURES)]):
                    numeric_target[index].append(float(value))
                for index, value in enumerate(features[len(NUMERIC_FEATURES) :]):
                    categorical_target[index].append(str(value))
                labels_target.append(label)
    except DatasetError:
        raise
    except (OSError, UnicodeError, csv.Error, KeyError) as exc:
        raise DatasetError(f"Unable to read LendingClub dataset: {exc}") from exc

    if len(training_labels) < 100:
        raise DatasetError(
            "Not enough resolved 36-month loans from 2007–2013 to train. "
            "Check the selected accepted-loans CSV."
        )
    if len(set(training_labels)) != 2:
        raise DatasetError("Training years must contain paid and charged-off loans.")

    def as_arrays(
        numeric: list[list[float]],
        categorical: list[list[str]],
        labels: list[int],
    ) -> tuple[NDArray[np.float64], NDArray[np.str_], NDArray[np.int8]]:
        return (
            np.asarray(numeric, dtype=np.float64).T,
            np.asarray(categorical, dtype=str).T,
            np.asarray(labels, dtype=np.int8),
        )

    training = as_arrays(numeric_columns, categorical_columns, training_labels)
    evaluations = {
        cohort: as_arrays(
            evaluation_numeric[cohort],
            evaluation_categorical[cohort],
            evaluation_labels[cohort],
        )
        for cohort in EVALUATION_COHORTS
    }
    counts = {
        "train": {
            "rows": int(training[2].size),
            "defaults": int(training[2].sum()),
            "non_defaults": int(training[2].size - training[2].sum()),
            "issue_years": f"2007-{TRAIN_END_YEAR}",
        },
        "evaluations": {
            cohort: {
                "rows": int(evaluations[cohort][2].size),
                "defaults": int(evaluations[cohort][2].sum()),
                "non_defaults": int(
                    evaluations[cohort][2].size - evaluations[cohort][2].sum()
                ),
                "unresolved_excluded": int(
                    row_counts[cohort] - evaluations[cohort][2].size
                ),
                "issue_year": int(cohort[:4]),
                "issue_quarter": int(cohort[-1]),
            }
            for cohort in EVALUATION_COHORTS
        },
        "resolved_training_year_counts": {
            str(year): int(outcome_counts[year]["Fully Paid"])
            + int(outcome_counts[year]["Charged Off"])
            + int(outcome_counts[year]["Default"])
            for year in range(2007, TRAIN_END_YEAR + 1)
        },
        "outcome_status_counts": dict(status_counts),
    }
    return (*training, evaluations, counts)


def _build_model() -> Pipeline:
    numeric_transform = SimpleImputer(strategy="median", keep_empty_features=True)
    categorical_transform = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OrdinalEncoder(
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                    encoded_missing_value=-1,
                ),
            ),
        ]
    )
    preprocessing = ColumnTransformer(
        transformers=[
            ("numeric", numeric_transform, list(range(len(NUMERIC_FEATURES)))),
            (
                "categorical",
                categorical_transform,
                list(range(len(NUMERIC_FEATURES), len(FEATURE_NAMES))),
            ),
        ],
        verbose_feature_names_out=False,
    )
    categorical_mask = [
        False
    ] * len(NUMERIC_FEATURES) + [True] * len(CATEGORICAL_FEATURES)
    classifier = HistGradientBoostingClassifier(
        learning_rate=0.07,
        max_iter=120,
        max_leaf_nodes=31,
        l2_regularization=1.0,
        categorical_features=categorical_mask,
        random_state=42,
    )
    return Pipeline(
        steps=[("preprocessing", preprocessing), ("classifier", classifier)]
    )


def _evaluation_metrics(
    model: Pipeline,
    numeric: NDArray[np.float64],
    categorical: NDArray[np.str_],
    labels: NDArray[np.int8],
) -> dict[str, float | int]:
    rows = int(labels.size)
    if rows < 2 or len(np.unique(labels)) != 2:
        return {
            "rows": rows,
            "defaults": int(labels.sum()),
            "non_defaults": int(rows - labels.sum()),
            "roc_auc": 0.0,
            "average_precision": 0.0,
            "brier_score": 0.0,
            "metrics_available": False,
        }
    features = _model_features(numeric, categorical)
    probabilities = model.predict_proba(features)[:, 1]
    return {
        "rows": rows,
        "defaults": int(labels.sum()),
        "non_defaults": int(rows - labels.sum()),
        "roc_auc": float(roc_auc_score(labels, probabilities)),
        "average_precision": float(average_precision_score(labels, probabilities)),
        "brier_score": float(brier_score_loss(labels, probabilities)),
        "metrics_available": True,
    }


def _model_features(
    numeric: NDArray[np.float64], categorical: NDArray[np.str_]
) -> NDArray[Any]:
    features = np.empty(
        (numeric.shape[0], numeric.shape[1] + categorical.shape[1]), dtype=object
    )
    features[:, : numeric.shape[1]] = numeric
    features[:, numeric.shape[1] :] = categorical
    return features


def _default_risk_summary(
    predictions: NDArray[np.float64],
    labels: NDArray[np.int8] | None = None,
    labeled_predictions: NDArray[np.float64] | None = None,
) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "rows_scored": int(predictions.size),
        "predicted_defaults_at_50_percent": int(np.count_nonzero(predictions >= 0.5)),
        "mean_predicted_default_probability": float(np.mean(predictions)),
        "labeled_rows": 0 if labels is None else int(labels.size),
        "evaluation": None,
    }
    if labels is not None and labels.size:
        evaluation_predictions = (
            predictions if labeled_predictions is None else labeled_predictions
        )
        evaluation: dict[str, Any] = {
            "resolved_rows": int(labels.size),
            "actual_defaults": int(labels.sum()),
            "actual_default_rate": float(np.mean(labels)),
            "roc_auc": None,
            "brier_score": float(brier_score_loss(labels, evaluation_predictions)),
        }
        if len(np.unique(labels)) == 2:
            evaluation["roc_auc"] = float(
                roc_auc_score(labels, evaluation_predictions)
            )
        summary["evaluation"] = evaluation
    return summary


def score_feature_batch(
    features: dict[str, NDArray[Any]], model_path: str = MODEL_PATH
) -> dict[str, Any]:
    path = Path(model_path)
    if not path.is_file():
        raise DatasetError("Train the LendingClub model before scoring a batch.")
    try:
        bundle = joblib.load(path)
    except (OSError, EOFError, ValueError, KeyError) as exc:
        raise DatasetError(f"Unable to read saved model bundle: {exc}") from exc
    if bundle["metadata"].get("dataset_name") != CSV_PATH.name:
        raise DatasetError("The saved model was trained on a different dataset.")
    numeric = np.column_stack([features[name] for name in NUMERIC_FEATURES])
    categorical = np.column_stack([features[name] for name in CATEGORICAL_FEATURES])
    predictions = bundle["model"].predict_proba(
        _model_features(numeric, categorical)
    )[:, 1]
    return _default_risk_summary(predictions)


def train_model(dataset_path: str = DATASET_PATH, model_path: str = MODEL_PATH) -> dict[str, Any]:
    (
        train_numeric,
        train_categorical,
        train_labels,
        evaluations,
        counts,
    ) = load_temporal_cohorts(dataset_path)
    train_features = _model_features(train_numeric, train_categorical)
    model = _build_model()
    model.fit(train_features, train_labels)
    evaluation_metrics = {
        cohort: _evaluation_metrics(model, *arrays)
        for cohort, arrays in evaluations.items()
    }
    metadata = {
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "model_type": "HistGradientBoostingClassifier",
        "dataset_name": Path(dataset_path).name,
        "split_method": "chronological_by_issue_date",
        "training_period": f"2007-{TRAIN_END_YEAR}",
        "training_loan_term_months": 36,
        "evaluation_cohorts": list(EVALUATION_COHORTS),
        "rows": counts["train"]["rows"],
        "training_rows": counts["train"]["rows"],
        "evaluation_rows": sum(
            int(entry["rows"]) for entry in evaluation_metrics.values()
        ),
        "default_count": counts["train"]["defaults"],
        "non_default_count": counts["train"]["non_defaults"],
        "features": list(FEATURE_NAMES),
        "metrics_by_cohort": evaluation_metrics,
        "cohort_counts": counts,
    }

    destination = Path(model_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = destination.with_suffix(destination.suffix + ".tmp")
    try:
        joblib.dump(
            {
                "model": model,
                "metadata": metadata,
                "reference_numeric": train_numeric,
                "reference_categorical": train_categorical,
                "evaluation_cohorts": evaluations,
            },
            temporary_path,
            compress=3,
        )
        os.replace(temporary_path, destination)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()
    return metadata


def model_status(model_path: str = MODEL_PATH) -> dict[str, Any]:
    path = Path(model_path)
    if not path.is_file():
        return {"trained": False, "metadata": None}
    try:
        bundle = joblib.load(path)
    except (OSError, EOFError, ValueError, KeyError) as exc:
        raise DatasetError(f"Unable to read saved model bundle: {exc}") from exc
    metadata = bundle["metadata"]
    matches_dataset = (
        metadata.get("dataset_name") == CSV_PATH.name
        and metadata.get("evaluation_cohorts") == list(EVALUATION_COHORTS)
    )
    return {"trained": matches_dataset, "metadata": metadata if matches_dataset else None}


def dataset_drift_samples(
    cohort: str, model_path: str = MODEL_PATH
) -> tuple[dict[str, NDArray[Any]], dict[str, NDArray[Any]]]:
    if cohort not in EVALUATION_COHORTS:
        raise DatasetError(
            "Choose a supported quarter cohort: " + ", ".join(EVALUATION_COHORTS)
        )
    path = Path(model_path)
    if not path.is_file():
        raise DatasetError("Train the LendingClub model before monitoring a cohort.")
    try:
        bundle = joblib.load(path)
    except (OSError, EOFError, ValueError, KeyError) as exc:
        raise DatasetError(f"Unable to read saved model bundle: {exc}") from exc
    if bundle["metadata"].get("dataset_name") != CSV_PATH.name:
        raise DatasetError("The saved model was trained on a different dataset.")
    if cohort not in bundle.get("evaluation_cohorts", {}):
        raise DatasetError(
            "The saved model does not contain quarterly cohorts. Retrain the model."
        )
    reference_numeric = bundle["reference_numeric"]
    reference_categorical = bundle["reference_categorical"]
    evaluation_numeric, evaluation_categorical, _ = bundle["evaluation_cohorts"][
        cohort
    ]
    return (
        {
            name: reference_numeric[:, index]
            for index, name in enumerate(NUMERIC_FEATURES)
        }
        | {
            name: reference_categorical[:, index]
            for index, name in enumerate(CATEGORICAL_FEATURES)
        },
        {
            name: evaluation_numeric[:, index]
            for index, name in enumerate(NUMERIC_FEATURES)
        }
        | {
            name: evaluation_categorical[:, index]
            for index, name in enumerate(CATEGORICAL_FEATURES)
        },
    )


def simulated_incoming_samples(
    sample_size: int = 1000, seed: int = 42, model_path: str = MODEL_PATH
) -> tuple[dict[str, NDArray[Any]], dict[str, NDArray[Any]]]:
    if not 100 <= sample_size <= 5000:
        raise DatasetError("Simulated batch size must be between 100 and 5,000.")
    path = Path(model_path)
    if not path.is_file():
        raise DatasetError("Train the LendingClub model before simulating a batch.")
    try:
        bundle = joblib.load(path)
    except (OSError, EOFError, ValueError, KeyError) as exc:
        raise DatasetError(f"Unable to read saved model bundle: {exc}") from exc
    if bundle["metadata"].get("dataset_name") != CSV_PATH.name:
        raise DatasetError("The saved model was trained on a different dataset.")
    reference_numeric = bundle["reference_numeric"]
    reference_categorical = bundle["reference_categorical"]
    if reference_numeric.shape[0] < 2:
        raise DatasetError("Not enough reference rows to create a simulated batch.")

    random = np.random.default_rng(seed)
    synthetic_numeric_columns = []
    for index, name in enumerate(NUMERIC_FEATURES):
        reference_feature = reference_numeric[:, index]
        finite_reference = reference_feature[np.isfinite(reference_feature)]
        if finite_reference.size < 2:
            raise DatasetError(
                f"Not enough finite reference values for feature '{name}'."
            )
        synthetic_feature = random.choice(
            finite_reference,
            size=sample_size,
            replace=sample_size > finite_reference.size,
        )
        noise_scale = float(np.std(finite_reference)) * 0.01
        if noise_scale:
            synthetic_feature += random.normal(
                0, noise_scale, size=sample_size
            )
            np.clip(
                synthetic_feature,
                np.min(finite_reference),
                np.max(finite_reference),
                out=synthetic_feature,
            )
        synthetic_numeric_columns.append(synthetic_feature)
    synthetic_reference_numeric = np.column_stack(synthetic_numeric_columns)

    synthetic_reference_categorical = np.column_stack(
        [
            random.choice(
                reference_categorical[:, index],
                size=sample_size,
                replace=sample_size > reference_categorical.shape[0],
            )
            for index in range(reference_categorical.shape[1])
        ]
    )
    current_numeric = synthetic_reference_numeric.copy()
    current_categorical = synthetic_reference_categorical.astype(object)

    annual_income_index = NUMERIC_FEATURES.index("annual_inc")
    current_numeric[:, annual_income_index] *= 1.4
    interest_rate_index = NUMERIC_FEATURES.index("int_rate")
    current_numeric[:, interest_rate_index] += 2.0
    purpose_index = CATEGORICAL_FEATURES.index("purpose")
    current_categorical[
        : sample_size // 3, purpose_index
    ] = "simulated_new_category"

    reference = {
        name: synthetic_reference_numeric[:, index]
        for index, name in enumerate(NUMERIC_FEATURES)
    } | {
        name: synthetic_reference_categorical[:, index]
        for index, name in enumerate(CATEGORICAL_FEATURES)
    }
    current = {
        name: current_numeric[:, index]
        for index, name in enumerate(NUMERIC_FEATURES)
    } | {
        name: current_categorical[:, index]
        for index, name in enumerate(CATEGORICAL_FEATURES)
    }
    return reference, current


def simulated_loan_csv(
    sample_size: int = 1000,
    seed: int = 42,
    model_path: str = MODEL_PATH,
    batch: Literal["reference", "incoming"] = "incoming",
) -> bytes:
    reference, incoming = simulated_incoming_samples(sample_size, seed, model_path)
    features = reference if batch == "reference" else incoming
    issue_date = datetime(2015, 10, 1)
    output = io.StringIO(newline="")
    writer = csv.DictWriter(
        output,
        fieldnames=[*CSV_FEATURE_COLUMNS, "loan_status"],
        lineterminator="\r\n",
    )
    writer.writeheader()
    for index in range(sample_size):
        credit_history_months = max(
            0, int(round(float(features["credit_history_months"][index])))
        )
        credit_line_month = (
            issue_date.year * 12 + issue_date.month - 1 - credit_history_months
        )
        credit_line = datetime(
            credit_line_month // 12,
            credit_line_month % 12 + 1,
            1,
        )
        employment_years = float(features["emp_length_years"][index])
        employment_length = (
            "< 1 year"
            if employment_years <= 0.5
            else f"{int(round(employment_years))} years"
        )
        row = {
            "loan_amnt": features["loan_amnt"][index],
            "term": " 36 months",
            "int_rate": f"{features['int_rate'][index]}%",
            "installment": features["installment"][index],
            "emp_length": employment_length,
            "annual_inc": features["annual_inc"][index],
            "dti": features["dti"][index],
            "delinq_2yrs": features["delinq_2yrs"][index],
            "earliest_cr_line": credit_line.strftime("%b-%Y"),
            "issue_d": issue_date.strftime("%b-%Y"),
            "fico_range_low": features["fico_range_low"][index],
            "fico_range_high": features["fico_range_high"][index],
            "inq_last_6mths": features["inq_last_6mths"][index],
            "open_acc": features["open_acc"][index],
            "pub_rec": features["pub_rec"][index],
            "revol_bal": features["revol_bal"][index],
            "revol_util": f"{features['revol_util'][index]}%",
            "total_acc": features["total_acc"][index],
            "grade": features["grade"][index],
            "home_ownership": features["home_ownership"][index],
            "verification_status": features["verification_status"][index],
            "purpose": features["purpose"][index],
            "loan_status": "",
        }
        writer.writerow(row)
    return output.getvalue().encode("utf-8")


def _parse_uploaded_loan_csv(
    contents: bytes,
) -> tuple[list[dict[str, str]], NDArray[np.float64], NDArray[np.str_]]:
    if not contents:
        raise DatasetError("The uploaded loan CSV is empty.")
    if len(contents) > 20 * 1024 * 1024:
        raise DatasetError("The uploaded loan CSV must be 20 MB or smaller.")

    try:
        reader = csv.DictReader(
            io.StringIO(contents.decode("utf-8-sig"), newline="")
        )
        headers = [header.strip() for header in (reader.fieldnames or [])]
        if len(headers) != len(set(headers)):
            raise DatasetError("Loan CSV column names must be unique.")
        missing = sorted(set(CSV_FEATURE_COLUMNS).difference(headers))
        if missing:
            raise DatasetError(
                "Loan CSV is missing required columns: " + ", ".join(missing)
            )
        rows = [
            {key.strip(): (value or "").strip() for key, value in row.items() if key}
            for row in reader
        ]
    except DatasetError:
        raise
    except (UnicodeDecodeError, csv.Error) as exc:
        raise DatasetError(f"Unable to parse uploaded loan CSV: {exc}") from exc

    if not 2 <= len(rows) <= 100_000:
        raise DatasetError("Loan CSV must contain between 2 and 100,000 rows.")
    if any(_term_months(row["term"]) != 36 for row in rows):
        raise DatasetError(
            "Loan-default scoring supports 36-month loans only, matching the trained model."
        )

    issue_dates = [_issue_date(row["issue_d"]) for row in rows]
    if any(issue_date is None for issue_date in issue_dates):
        raise DatasetError("Every loan row must have a valid issue_d date.")
    features = [
        _loan_features(row, issue_date)
        for row, issue_date in zip(rows, issue_dates, strict=True)
        if issue_date is not None
    ]
    numeric = np.asarray(
        [row[: len(NUMERIC_FEATURES)] for row in features], dtype=np.float64
    )
    categorical = np.asarray(
        [row[len(NUMERIC_FEATURES) :] for row in features], dtype=str
    )
    return rows, numeric, categorical


def monitor_loan_batch(
    contents: bytes,
    model_path: str = MODEL_PATH,
    reference_contents: bytes | None = None,
) -> tuple[
    dict[str, NDArray[Any]],
    dict[str, NDArray[Any]],
    dict[str, Any],
]:
    rows, numeric, categorical = _parse_uploaded_loan_csv(contents)

    path = Path(model_path)
    if not path.is_file():
        raise DatasetError("Train the LendingClub model before uploading loan data.")
    try:
        bundle = joblib.load(path)
    except (OSError, EOFError, ValueError, KeyError) as exc:
        raise DatasetError(f"Unable to read saved model bundle: {exc}") from exc
    if bundle["metadata"].get("dataset_name") != CSV_PATH.name:
        raise DatasetError("The saved model was trained on a different dataset.")

    predictions = bundle["model"].predict_proba(
        _model_features(numeric, categorical)
    )[:, 1]
    labeled_rows = [
        (index, row["loan_status"])
        for index, row in enumerate(rows)
        if row.get("loan_status") in POSITIVE_STATUSES | NEGATIVE_STATUSES
    ]
    labels = np.asarray(
        [int(status in POSITIVE_STATUSES) for _, status in labeled_rows],
        dtype=np.int8,
    )
    labeled_predictions = np.asarray(
        [predictions[index] for index, _ in labeled_rows], dtype=np.float64
    )
    prediction_summary = _default_risk_summary(
        predictions,
        labels if labels.size else None,
        labeled_predictions if labels.size else None,
    )

    if reference_contents is None:
        reference_numeric = bundle["reference_numeric"]
        reference_categorical = bundle["reference_categorical"]
    else:
        _, reference_numeric, reference_categorical = _parse_uploaded_loan_csv(
            reference_contents
        )
    reference = {
        name: reference_numeric[:, index]
        for index, name in enumerate(NUMERIC_FEATURES)
    } | {
        name: reference_categorical[:, index]
        for index, name in enumerate(CATEGORICAL_FEATURES)
    }
    current = {
        name: numeric[:, index]
        for index, name in enumerate(NUMERIC_FEATURES)
    } | {
        name: categorical[:, index]
        for index, name in enumerate(CATEGORICAL_FEATURES)
    }
    return reference, current, prediction_summary
