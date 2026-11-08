import os


DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql+psycopg://ddis:ddis_local_only@localhost:5432/ddis",
)
DATASET_PATH = os.getenv(
    "DATASET_PATH", "/data/accepted_2007_to_2018Q4.csv"
)
MODEL_PATH = os.getenv("MODEL_PATH", "/models/ddis_model.joblib")
