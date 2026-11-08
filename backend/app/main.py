from contextlib import asynccontextmanager
from typing import Literal

from fastapi import Depends, FastAPI, File, HTTPException, Query, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from numpy.typing import ArrayLike
from sqlalchemy import inspect, select, text
from sqlalchemy.orm import Session

from app.database import Base, engine, get_db
from app.drift import compare_feature
from app.ingestion import (
    MAX_UPLOAD_BYTES,
    CSVInputError,
    parse_comparison_csvs,
    serialize_feature_csv,
)
from app.models import DriftRun
from app.notifications import notify_drift
from app.schemas import (
    DatasetDriftRequest,
    DatasetStatusResponse,
    DriftComparisonRequest,
    DriftRunResponse,
    ModelStatusResponse,
)
from app.training import (
    CATEGORICAL_FEATURES,
    DatasetError,
    EVALUATION_COHORTS,
    dataset_drift_samples,
    dataset_status,
    monitor_loan_batch,
    model_status,
    score_feature_batch,
    simulated_incoming_samples,
    simulated_loan_csv,
    train_model,
)


@asynccontextmanager
async def lifespan(_: FastAPI):
    Base.metadata.create_all(bind=engine)
    if "source" not in {
        column["name"] for column in inspect(engine).get_columns("drift_runs")
    }:
        with engine.begin() as connection:
            connection.execute(
                text(
                    "ALTER TABLE drift_runs ADD COLUMN source VARCHAR(40) "
                    "NOT NULL DEFAULT 'synthetic_demo'"
                )
            )
    yield


app = FastAPI(
    title="Data Drift Intelligence System",
    version="0.1.0",
    description="Local MVP API for numeric feature drift monitoring.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


@app.get("/api/health")
def health(db: Session = Depends(get_db)) -> dict[str, str]:
    db.execute(text("SELECT 1"))
    return {"status": "ok", "database": "connected"}


def create_drift_run(
    reference: dict[str, ArrayLike],
    current: dict[str, ArrayLike],
    db: Session,
    source: str = "provided_data",
    categorical_features: set[str] | None = None,
    send_notifications: bool = True,
) -> DriftRun:
    if set(reference) != set(current):
        raise HTTPException(
            status_code=422,
            detail="Reference and current data must contain the same feature names.",
        )

    results = {}
    try:
        for feature in sorted(reference):
            results[feature] = compare_feature(
                reference[feature],
                current[feature],
                categorical=(
                    feature in CATEGORICAL_FEATURES
                    if categorical_features is None
                    else feature in categorical_features
                ),
            )
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    drifted_count = sum(result["drift_detected"] for result in results.values())
    run = DriftRun(
        status="drift_detected" if drifted_count else "healthy",
        source=source,
        feature_count=len(results),
        drifted_feature_count=drifted_count,
        results=results,
    )
    db.add(run)
    db.commit()
    db.refresh(run)
    run.notification_delivery = (
        notify_drift(run)
        if send_notifications
        else {"slack": "not_triggered", "gmail": "not_triggered"}
    )
    return run


@app.post("/api/v1/drift/runs", response_model=DriftRunResponse)
def create_run(
    request: DriftComparisonRequest, db: Session = Depends(get_db)
) -> DriftRun:
    return create_drift_run(request.reference, request.current, db)


@app.get("/api/v1/dataset/status", response_model=DatasetStatusResponse)
def get_dataset_status() -> dict:
    try:
        return dataset_status()
    except DatasetError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.get("/api/v1/model/status", response_model=ModelStatusResponse)
def get_model_status() -> dict:
    return model_status()


@app.post("/api/v1/model/train")
def train_baseline_model(db: Session = Depends(get_db)) -> dict:
    try:
        metadata = train_model()
        default_cohort = EVALUATION_COHORTS[-1]
        reference, current = dataset_drift_samples(default_cohort)
    except DatasetError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    run = create_drift_run(
        reference, current, db, source=f"lendingclub_{default_cohort}"
    )
    return {**metadata, "notification_delivery": run.notification_delivery}


@app.post("/api/v1/drift/dataset", response_model=DriftRunResponse)
def run_dataset_drift(
    request: DatasetDriftRequest, db: Session = Depends(get_db)
) -> DriftRun:
    try:
        reference, current = dataset_drift_samples(
            request.cohort
        )
    except DatasetError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return create_drift_run(
        reference, current, db, source=f"lendingclub_{request.cohort}"
    )


@app.post("/api/v1/drift/upload", response_model=DriftRunResponse)
async def run_uploaded_drift(
    reference_file: UploadFile = File(),
    incoming_file: UploadFile = File(),
    db: Session = Depends(get_db),
) -> DriftRun:
    if not (reference_file.filename or "").lower().endswith(".csv"):
        raise HTTPException(status_code=422, detail="Reference file must be a CSV.")
    if not (incoming_file.filename or "").lower().endswith(".csv"):
        raise HTTPException(status_code=422, detail="Incoming file must be a CSV.")
    try:
        reference_bytes = await reference_file.read(MAX_UPLOAD_BYTES + 1)
        incoming_bytes = await incoming_file.read(MAX_UPLOAD_BYTES + 1)
        reference, current, categorical_features = parse_comparison_csvs(
            reference_bytes, incoming_bytes
        )
    except CSVInputError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    finally:
        await reference_file.close()
        await incoming_file.close()
    return create_drift_run(
        reference,
        current,
        db,
        source="incoming_upload",
        categorical_features=categorical_features,
    )


@app.post("/api/v1/loans/monitor")
async def monitor_uploaded_loan_batch(
    reference_file: UploadFile = File(),
    loan_file: UploadFile = File(),
    db: Session = Depends(get_db),
) -> dict:
    if not (reference_file.filename or "").lower().endswith(".csv"):
        raise HTTPException(status_code=422, detail="Reference data must be a CSV file.")
    if not (loan_file.filename or "").lower().endswith(".csv"):
        raise HTTPException(status_code=422, detail="Incoming data must be a CSV file.")
    try:
        reference_contents = await reference_file.read(MAX_UPLOAD_BYTES + 1)
        contents = await loan_file.read(MAX_UPLOAD_BYTES + 1)
        reference, current, prediction_summary = monitor_loan_batch(
            contents, reference_contents=reference_contents
        )
    except DatasetError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    finally:
        await reference_file.close()
        await loan_file.close()

    run = create_drift_run(
        reference,
        current,
        db,
        source="incoming_upload",
        categorical_features=set(CATEGORICAL_FEATURES),
    )
    return {
        "id": run.id,
        "created_at": run.created_at,
        "status": run.status,
        "source": run.source,
        "feature_count": run.feature_count,
        "drifted_feature_count": run.drifted_feature_count,
        "results": run.results,
        "reference_row_count": len(reference["loan_amnt"]),
        "incoming_row_count": len(current["loan_amnt"]),
        "prediction_summary": prediction_summary,
    }


@app.get("/api/v1/loans/simulate/download")
def download_simulated_loan_csv(
    batch: Literal["reference", "incoming"] = Query(),
    sample_size: int = Query(default=1000, ge=100, le=5000),
    seed: int = Query(default=42, ge=0, le=2**32 - 1),
) -> Response:
    try:
        contents = simulated_loan_csv(sample_size, seed, batch=batch)
    except DatasetError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return Response(
        content=contents,
        media_type="text/csv",
        headers={
            "Content-Disposition": (
                f'attachment; filename="simulated_{batch}_loan_data.csv"'
            )
        },
    )


@app.post("/api/v1/drift/simulate")
def run_simulated_drift(
    db: Session = Depends(get_db),
    sample_size: int = Query(default=1000, ge=100, le=5000),
    seed: int = Query(default=42, ge=0, le=2**32 - 1),
) -> dict:
    try:
        reference, current = simulated_incoming_samples(sample_size, seed)
    except DatasetError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    run = create_drift_run(
        reference,
        current,
        db,
        source="simulated_incoming",
        send_notifications=False,
    )
    return {
        "id": run.id,
        "created_at": run.created_at,
        "status": run.status,
        "source": run.source,
        "feature_count": run.feature_count,
        "drifted_feature_count": run.drifted_feature_count,
        "results": run.results,
        "notification_delivery": run.notification_delivery,
        "prediction_summary": score_feature_batch(current),
    }


@app.get("/api/v1/drift/simulate/download")
def download_simulated_csv(
    batch: Literal["reference", "incoming"] = Query(default="incoming"),
    sample_size: int = Query(default=1000, ge=100, le=5000),
    seed: int = Query(default=42, ge=0, le=2**32 - 1),
) -> Response:
    try:
        reference, incoming = simulated_incoming_samples(sample_size, seed)
    except DatasetError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc

    features = reference if batch == "reference" else incoming
    return Response(
        content=serialize_feature_csv(features),
        media_type="text/csv",
        headers={
            "Content-Disposition": f'attachment; filename="simulated_{batch}.csv"'
        },
    )


@app.get("/api/v1/drift/runs", response_model=list[DriftRunResponse])
def list_runs(
    limit: int = Query(default=20, ge=1, le=100),
    db: Session = Depends(get_db),
) -> list[DriftRun]:
    statement = select(DriftRun).order_by(DriftRun.created_at.desc()).limit(limit)
    return list(db.scalars(statement))


@app.get("/api/v1/drift/latest", response_model=DriftRunResponse | None)
def latest_run(
    source: Literal[
        "all", "lendingclub", "simulated_incoming", "incoming_upload"
    ] = "all",
    db: Session = Depends(get_db),
) -> DriftRun | None:
    statement = select(DriftRun)
    if source == "lendingclub":
        statement = statement.where(DriftRun.source.like("lendingclub_%"))
    elif source != "all":
        statement = statement.where(DriftRun.source == source)
    statement = statement.order_by(DriftRun.created_at.desc()).limit(1)
    return db.scalar(statement)
