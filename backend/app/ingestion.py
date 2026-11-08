import csv
import io

import numpy as np
from numpy.typing import ArrayLike, NDArray


MAX_UPLOAD_BYTES = 20 * 1024 * 1024
MAX_UPLOAD_ROWS = 100_000
MAX_UPLOAD_COLUMNS = 100


class CSVInputError(ValueError):
    pass


def parse_feature_csv(contents: bytes) -> tuple[dict[str, NDArray], set[str]]:
    if not contents:
        raise CSVInputError("The uploaded CSV is empty.")
    if len(contents) > MAX_UPLOAD_BYTES:
        raise CSVInputError("Each CSV must be 20 MB or smaller.")
    try:
        reader = csv.reader(io.StringIO(contents.decode("utf-8-sig"), newline=""))
        headers = next(reader, [])
        if not headers or len(headers) > MAX_UPLOAD_COLUMNS:
            raise CSVInputError(
                f"CSV must contain between 1 and {MAX_UPLOAD_COLUMNS} feature columns."
            )
        headers = [name.strip() for name in headers]
        if any(not name.strip() for name in headers):
            raise CSVInputError("CSV column names cannot be empty.")
        if len(set(headers)) != len(headers):
            raise CSVInputError("CSV column names must be unique.")

        columns: dict[str, list[str]] = {name: [] for name in headers}
        row_count = 0
        for row in reader:
            if not row:
                continue
            if len(row) != len(headers):
                raise CSVInputError(
                    f"CSV row {row_count + 2} has {len(row)} values; "
                    f"expected {len(headers)}."
                )
            row_count += 1
            if row_count > MAX_UPLOAD_ROWS:
                raise CSVInputError(
                    f"Each CSV must contain at most {MAX_UPLOAD_ROWS:,} rows."
                )
            for name, value in zip(headers, row, strict=True):
                columns[name].append(value.strip())
    except UnicodeDecodeError as exc:
        raise CSVInputError("CSV files must use UTF-8 encoding.") from exc
    except csv.Error as exc:
        raise CSVInputError(f"Unable to parse CSV: {exc}") from exc

    if row_count < 2:
        raise CSVInputError("Each CSV must contain at least two data rows.")

    features: dict[str, NDArray] = {}
    categorical_features: set[str] = set()
    for name, values in columns.items():
        present = [value for value in values if value]
        if not present:
            raise CSVInputError(f"Feature '{name}' contains no non-empty values.")
        try:
            parsed = [float(value) for value in present]
            if not np.isfinite(parsed).all():
                raise ValueError
        except ValueError:
            categorical_features.add(name)
            features[name] = np.asarray(values, dtype=str)
            continue

        iterator = iter(parsed)
        features[name] = np.asarray(
            [next(iterator) if value else float("nan") for value in values],
            dtype=float,
        )
    return features, categorical_features


def parse_comparison_csvs(
    reference_contents: bytes, current_contents: bytes
) -> tuple[dict[str, NDArray], dict[str, NDArray], set[str]]:
    reference, reference_categorical = parse_feature_csv(reference_contents)
    current, current_categorical = parse_feature_csv(current_contents)
    if set(reference) != set(current):
        raise CSVInputError(
            "Reference and incoming CSVs must contain the same feature names."
        )
    if reference_categorical != current_categorical:
        raise CSVInputError(
            "Each feature must have the same inferred type in both CSV files."
        )
    return reference, current, reference_categorical


def serialize_feature_csv(features: dict[str, ArrayLike]) -> bytes:
    if not features:
        raise CSVInputError("A CSV export must contain at least one feature.")

    columns = {name: np.asarray(values) for name, values in features.items()}
    row_counts = {column.size for column in columns.values()}
    if len(row_counts) != 1:
        raise CSVInputError("All exported features must contain the same row count.")

    output = io.StringIO(newline="")
    writer = csv.writer(output, lineterminator="\r\n")
    writer.writerow(columns)
    row_count = next(iter(row_counts))
    for row_index in range(row_count):
        row = []
        for column in columns.values():
            value = column[row_index]
            if isinstance(value, np.generic):
                value = value.item()
            if isinstance(value, float) and not np.isfinite(value):
                value = ""
            row.append(value)
        writer.writerow(row)
    return output.getvalue().encode("utf-8")
