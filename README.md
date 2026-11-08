# Data Drift Intelligence System

Credit-risk model and feature-drift monitoring using LendingClub accepted-loan
data, a scikit-learn model, numeric and categorical drift tests, FastAPI,
PostgreSQL, and a React/TypeScript dashboard.

## Run locally

Requirements: Docker Desktop with Docker Compose.

Place `accepted_2007_to_2018Q4.csv` in `data/`. Docker mounts `data/` read-only
into the API container. The trained model and quarter cohorts are stored in a
named Docker volume.

```powershell
docker compose up --build
```

Open the dashboard at <http://localhost:5173> and the API docs at
<http://localhost:8000/docs>. Choose **Train temporal model**. Training scans
the 1.6 GB CSV and may take a few minutes on the first run.

## Drift notifications

When a cohort check or real uploaded batch detects drift, the API attempts
delivery by Gmail. Copy `.env.example` to `.env` and set
`GMAIL_SMTP_USERNAME`, `GMAIL_APP_PASSWORD`, and `GMAIL_TO`. Use an app
password with a Google account that has 2-Step
Verification enabled. Never commit `.env` or paste secrets into source files.
Restart the Compose services after changing the file. The API response and
dashboard report whether Gmail sent, failed, or is not configured;
notification failures are logged and do not discard the saved drift run.
Generated simulation runs do not send alerts.

## Incoming data monitoring

- The dashboard's **Incoming loan data** section has separate Reference data
  and Incoming data CSV inputs and one analysis action. Drift is measured
  between the uploaded reference and incoming distributions; default risk is
  scored for incoming loans. The accepted loan model input columns are
  required, and `loan_status` is optional on incoming data for outcome metrics.
- **Download reference data** and **Download incoming data** generate a
  matching synthetic 1,000-row pair that can be selected in those same inputs.
  The incoming file has intentional feature shifts. The files are simulated,
  not observed traffic or evidence of live operation, and contain no source
  rows or labels.
- Files are processed in memory and raw rows and individual predictions are
  not retained. Uploads are limited to 20 MB and 100,000 rows.
- LendingClub quarterly cohort results remain the dashboard's primary
  monitoring result; incoming and simulated loan-batch results stay separate.

## Cohort evaluation design

- Train on loans issued from 2007 through 2013.
- Restrict training and evaluation to 36-month loans so each cohort uses the
  same loan term.
- Use only application-time fields as model inputs. `loan_status` is the label:
  `Charged Off`/`Default` are positive and `Fully Paid` is negative.
- Evaluate on eight matured quarterly cohorts from 2014 Q1 through 2015 Q4.
  Current, late, and other unresolved outcomes are excluded from supervised
  metrics.
- Compare numeric features with PSI and KS, and categorical features with PSI
  plus chi-square. Each cohort drift run is stored in PostgreSQL.
- Report ROC AUC, average precision, and Brier score for each cohort.

This replays real US LendingClub loan cohorts to measure feature drift and
out-of-time model performance. It is not a live data feed and does not establish
that results generalize to Indian microfinance institutions. Outcome metrics
describe only the resolved subset and may be affected by outcome-selection bias.

## Tests

```powershell
docker compose exec -T api python -m unittest discover -s /app/tests
docker compose exec -T web npm run build
```

## Data and local security

`data/` is Git-ignored. Keep downloaded data there, review the source's terms,
and do not commit or share the loan-level data. PostgreSQL is reachable only
inside the Compose network. The API and dashboard bind to localhost. The
default database password is for local development only; do not expose this
prototype or use real borrower data without authentication, access controls,
secret management, and appropriate data-governance approval.
