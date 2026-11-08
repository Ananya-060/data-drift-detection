import { useCallback, useEffect, useState } from "react";
import type { ReactNode } from "react";
import {
  Activity,
  ArrowDownRight,
  ArrowUpRight,
  Bell,
  Check,
  ChevronDown,
  CircleHelp,
  Clock3,
  Database,
  LayoutDashboard,
  LoaderCircle,
  RefreshCw,
  Settings2,
  ShieldCheck,
  SlidersHorizontal,
  Sparkles,
  TriangleAlert,
  Upload,
} from "lucide-react";

type FeatureResult = {
  psi: number;
  test_statistic: number;
  test_name: "kolmogorov_smirnov" | "chi_square";
  p_value: number;
  severity: "no_drift" | "moderate" | "severe";
  drift_detected: boolean;
};

type DriftRun = {
  id: number;
  created_at: string;
  status: string;
  source: string;
  feature_count: number;
  drifted_feature_count: number;
  results: Record<string, FeatureResult>;
  notification_delivery?: Record<string, string>;
};

type LoanBatchRun = DriftRun & {
  reference_row_count: number;
  incoming_row_count: number;
  prediction_summary: {
    rows_scored: number;
    predicted_defaults_at_50_percent: number;
    mean_predicted_default_probability: number;
    labeled_rows: number;
    evaluation: {
      resolved_rows: number;
      actual_defaults: number;
      actual_default_rate: number;
      roc_auc: number | null;
      brier_score: number;
    } | null;
  };
};

type DatasetStatus = {
  available: boolean;
  file_name: string;
  size_bytes: number;
  missing_columns: string[];
  train_end_year: number;
  evaluation_cohorts: string[];
};

type CohortMetrics = {
  rows: number;
  defaults: number;
  non_defaults: number;
  roc_auc: number;
  average_precision: number;
  brier_score: number;
  metrics_available: boolean;
};

type TrendPoint = {
  cohort: string;
  value: number;
};

type ModelMetadata = {
  trained_at: string;
  model_type: string;
  dataset_name: string;
  split_method: string;
  training_period: string;
  rows: number;
  training_rows: number;
  evaluation_rows: number;
  default_count: number;
  non_default_count: number;
  features: string[];
  metrics_by_cohort: Record<string, CohortMetrics>;
  cohort_counts: {
    evaluations: Record<string, {
      rows: number;
      defaults: number;
      non_defaults: number;
      unresolved_excluded: number;
      issue_year: number;
      issue_quarter: number;
    }>;
  };
};

type ModelStatus = {
  trained: boolean;
  metadata: ModelMetadata | null;
};

const API = import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8000";

function App() {
  const [run, setRun] = useState<DriftRun | null>(null);
  const [loanBatchRun, setLoanBatchRun] = useState<LoanBatchRun | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [notificationMessage, setNotificationMessage] = useState("");
  const [databaseConnected, setDatabaseConnected] = useState(false);
  const [dataset, setDataset] = useState<DatasetStatus | null>(null);
  const [model, setModel] = useState<ModelStatus | null>(null);
  const [selectedCohort, setSelectedCohort] = useState("2015-Q4");
  const [referenceLoanFile, setReferenceLoanFile] = useState<File | null>(null);
  const [incomingLoanFile, setIncomingLoanFile] = useState<File | null>(null);

  const refresh = useCallback(async () => {
    try {
      const healthResponse = await fetch(`${API}/api/health`);
      if (!healthResponse.ok) throw new Error("The API or database is unavailable.");
      const health = await healthResponse.json();
      setDatabaseConnected(health.database === "connected");
      const responses = await Promise.all([
        fetch(`${API}/api/v1/drift/latest?source=lendingclub`),
        fetch(`${API}/api/v1/dataset/status`),
        fetch(`${API}/api/v1/model/status`),
      ]);
      if (responses.some((response) => !response.ok)) {
        throw new Error("Could not load dataset or model status.");
      }
      const [latestRun, datasetStatus, modelStatus] = await Promise.all(
        responses.map((response) => response.json()),
      );
      setRun(latestRun);
      setDataset(datasetStatus);
      setModel(modelStatus);
      setError("");
    } catch (reason) {
      setDatabaseConnected(false);
      setError(reason instanceof Error ? reason.message : "API unavailable.");
    }
  }, []);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  async function requestAction<T>(path: string, body?: object): Promise<T> {
    setBusy(true);
    setError("");
    try {
      const response = await fetch(`${API}${path}`, {
        method: "POST",
        headers: body ? { "Content-Type": "application/json" } : undefined,
        body: body ? JSON.stringify(body) : undefined,
      });
      const result = await response.json();
      if (!response.ok) {
        throw new Error(typeof result.detail === "string" ? result.detail : "The request failed.");
      }
      return result as T;
    } catch (reason) {
      const message = reason instanceof Error ? reason.message : "API unavailable.";
      setError(message);
      throw new Error(message);
    } finally {
      setBusy(false);
    }
  }

  function summarizeDelivery(delivery?: Record<string, string>) {
    if (!delivery) return "";
    return `Email alert — Gmail: ${delivery.gmail ?? "unknown"}.`;
  }

  async function trainBaseline() {
    try {
      const result = await requestAction<{ notification_delivery?: Record<string, string> }>(
        "/api/v1/model/train",
      );
      setNotificationMessage(summarizeDelivery(result.notification_delivery));
      await refresh();
    } catch {
      return;
    }
  }

  async function runDatasetDrift(cohort: string) {
    try {
      const result = await requestAction<DriftRun>("/api/v1/drift/dataset", {
        cohort,
      });
      setRun(result);
      setNotificationMessage(summarizeDelivery(result.notification_delivery));
    } catch {
      return;
    }
  }

  async function downloadSimulatedLoanCsv(batch: "reference" | "incoming") {
    setBusy(true);
    setError("");
    try {
      const response = await fetch(
        `${API}/api/v1/loans/simulate/download?batch=${batch}`,
      );
      if (!response.ok) {
        const result = await response.json();
        throw new Error(
          typeof result.detail === "string"
            ? result.detail
            : `The simulated ${batch} data download failed.`,
        );
      }

      const objectUrl = URL.createObjectURL(await response.blob());
      const link = document.createElement("a");
      link.href = objectUrl;
      link.download = `simulated_${batch}_loan_data.csv`;
      document.body.append(link);
      link.click();
      link.remove();
      window.setTimeout(() => URL.revokeObjectURL(objectUrl), 1000);
      setNotificationMessage(`Simulated ${batch} data downloaded.`);
    } catch (reason) {
      setError(
        reason instanceof Error
          ? reason.message
          : `The simulated ${batch} data download failed.`,
      );
    } finally {
      setBusy(false);
    }
  }

  async function uploadLoanBatch() {
    if (!referenceLoanFile || !incomingLoanFile) {
      setError("Choose both reference data and incoming data CSV files.");
      return;
    }
    setBusy(true);
    setError("");
    try {
      const formData = new FormData();
      formData.append("reference_file", referenceLoanFile);
      formData.append("loan_file", incomingLoanFile);
      const response = await fetch(`${API}/api/v1/loans/monitor`, {
        method: "POST",
        body: formData,
      });
      const result = await response.json();
      if (!response.ok) {
        throw new Error(
          typeof result.detail === "string" ? result.detail : "Loan scoring failed.",
        );
      }
      setLoanBatchRun(result as LoanBatchRun);
      setNotificationMessage("");
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Loan scoring failed.");
    } finally {
      setBusy(false);
    }
  }

  const results = run?.results ?? {};
  const drifted = run?.drifted_feature_count ?? 0;
  const healthy = run !== null && drifted === 0;
  const metadata = model?.metadata ?? null;
  const latestEvaluationCohort = dataset?.evaluation_cohorts.at(-1);
  const latestEvaluation = latestEvaluationCohort
    ? metadata?.metrics_by_cohort[latestEvaluationCohort]
    : undefined;
  const quarterlyTrend = dataset?.evaluation_cohorts
    .map((cohort) => ({
      cohort,
      metrics: metadata?.metrics_by_cohort[cohort],
    }))
    .filter(
      (entry): entry is { cohort: string; metrics: CohortMetrics } =>
        entry.metrics?.metrics_available === true,
    );
  const runContext = run?.source.startsWith("lendingclub_")
    ? `Features from ${run.source.slice("lendingclub_".length)} LendingClub loans were compared with the training cohort.`
    : "";
  const lastChecked = run
    ? new Date(run.created_at).toLocaleString()
    : "No runs yet";
  const names: Record<string, string> = {
    loan_amnt: "Loan amount",
    int_rate: "Interest rate",
    installment: "Monthly installment",
    emp_length_years: "Employment length",
    annual_inc: "Annual income",
    dti: "Debt-to-income ratio",
    delinq_2yrs: "Delinquencies (2y)",
    credit_history_months: "Credit history length",
    fico_range_low: "FICO (low)",
    fico_range_high: "FICO (high)",
    inq_last_6mths: "Credit inquiries",
    open_acc: "Open accounts",
    pub_rec: "Public records",
    revol_bal: "Revolving balance",
    revol_util: "Revolving utilization",
    total_acc: "Total accounts",
    grade: "Loan grade",
    home_ownership: "Home ownership",
    verification_status: "Income verification",
    purpose: "Loan purpose",
  };
  const runDescription = runContext
    ? `${runContext} Drift scores compare feature distributions and do not require outcome labels.`
    : "";

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <a className="brand" href="#overview" aria-label="DDIS home">
          <span className="brand-mark"><Activity size={21} strokeWidth={2.5} /></span>
          <span>drift<span className="brand-light">wise</span></span>
        </a>
        <div className="workspace-label">WORKSPACE</div>
        <div className="workspace-switch">
          <span className="workspace-avatar">S</span>
          <span className="workspace-copy"><strong>LendingClub Benchmark</strong><small>US loan data</small></span>
        </div>
        <div className="nav-label">MONITORING</div>
        <a className="nav-item active" href="#overview"><LayoutDashboard size={17} /> Overview</a>
        <a className="nav-item" href="#features"><SlidersHorizontal size={17} /> Feature drift <span className="nav-count">{run?.feature_count ?? 0}</span></a>
        <a className="nav-item" href="#alerts"><Bell size={17} /> Alerts <span className="nav-count">{drifted}</span></a>
        <div className="nav-label nav-label-spaced">MANAGE</div>
        <a className="nav-item" href="#models"><Database size={17} /> Models</a>
        <a className="nav-item" href="#methodology"><Settings2 size={17} /> Methodology</a>
        <div className="sidebar-bottom">
          <div className="help-card"><CircleHelp size={17} /><span><strong>Need a hand?</strong><small>Read the quick start guide</small></span></div>
          <div className="user-card"><div className="user-avatar">AS</div><span><strong>Admin user</strong><small>Local workspace</small></span></div>
        </div>
      </aside>

      <main className="main-content" id="overview">
        <header className="topbar">
          <div className="breadcrumbs">Monitoring <span>/</span> <strong>Overview</strong></div>
          <div className="topbar-actions"><span className="env-badge"><span /> LOCAL ENVIRONMENT</span><a className="icon-button" href="#alerts" aria-label="View monitoring alerts"><Bell size={18} /></a><div className="top-avatar">AS</div></div>
        </header>

        <div className="page">
          <section className="page-heading">
            <div><div className="eyebrow"><span className="eyebrow-dot" /> MODEL MONITORING</div><h1>Good morning, Admin <span className="wave">✳</span></h1><p>Here’s the latest health check for your credit risk model.</p></div>
            <div className="heading-actions">
              <button className="button button-outline" onClick={() => void refresh()}><RefreshCw size={15} /> Refresh</button>
              {!model?.trained
                ? <button className="button button-primary" disabled={busy || !dataset?.available} onClick={() => void trainBaseline()}>{busy ? <LoaderCircle className="spin" size={16} /> : <Database size={15} />} Train temporal model</button>
                : <><label className="cohort-picker"><span className="visually-hidden">Cohort</span><select aria-label="Select quarter cohort" value={selectedCohort} onChange={(event) => setSelectedCohort(event.target.value)} disabled={busy}>{dataset?.evaluation_cohorts.map((cohort) => <option key={cohort} value={cohort}>{cohort}</option>)}</select></label><button className="button button-primary" disabled={busy || !selectedCohort} onClick={() => void runDatasetDrift(selectedCohort)}>{busy ? <LoaderCircle className="spin" size={16} /> : <Activity size={15} />} Monitor cohort</button></>}
            </div>
          </section>

          {error && <div className="error-banner"><TriangleAlert size={17} /><span>{error} Start the API and PostgreSQL with <code>docker compose up --build</code>.</span></div>}
          {notificationMessage && <div className="notification-banner" role="status">{notificationMessage}{notificationMessage.includes("not_configured") && <> Set channel credentials in the local <code>.env</code> file and restart the API.</>}</div>}

          <div className="monitoring-results-row">
            <section className="status-banner primary-status" aria-label="LendingClub monitoring result">
              <div className={`status-icon ${!run ? "status-neutral" : healthy ? "status-good" : "status-warn"}`}>{!run ? <Clock3 size={20} /> : healthy ? <ShieldCheck size={21} /> : <TriangleAlert size={20} />}</div>
              <div className="status-copy"><strong>{!run ? "Ready for cohort monitoring" : healthy ? "No significant feature drift detected" : "Feature drift detected"}</strong><span>{!run ? dataset?.available ? "Train on resolved 36-month loans from 2007–2013, then compare eight matured quarterly cohorts from 2014-Q1 to 2015-Q4." : "Place accepted_2007_to_2018Q4.csv in the project data folder." : healthy ? `No monitored features crossed the drift threshold. ${runContext}` : `${drifted} of ${run.feature_count} features crossed the monitoring threshold. ${runContext}`}</span></div>
              <div className="status-updated"><span>LENDINGCLUB CHECK</span><strong>{lastChecked}</strong></div>
              <a className="status-more" href="#features" aria-label="View LendingClub drift results"><ChevronDown size={17} /></a>
            </section>
          </div>

          <section className="dataset-card" id="models">
            <div className="dataset-card-head"><span className="dataset-icon"><Database size={17} /></span><div className="dataset-title"><strong>Temporal credit-risk benchmark</strong><span>{dataset?.available ? dataset.file_name : "Dataset not found"}</span></div><span className={`dataset-badge ${dataset?.available ? "dataset-ready" : "dataset-missing"}`}><i />{dataset?.available ? `${(dataset.size_bytes / 1024 / 1024).toFixed(0)} MB · ready` : "CSV required"}</span></div>
            <div className="dataset-stats">
              <div><span>TRAINING YEARS</span><strong>{metadata?.training_period ?? (dataset ? `2007–${dataset.train_end_year}` : "—")}</strong></div>
              <div><span>RESOLVED TRAINING LOANS</span><strong>{metadata ? metadata.training_rows.toLocaleString() : "—"}</strong></div>
              <div><span>{latestEvaluationCohort ?? "OUT-OF-TIME"} AUC</span><strong>{latestEvaluation?.metrics_available ? latestEvaluation.roc_auc.toFixed(3) : "—"}</strong></div>
              <div><span>{latestEvaluationCohort ?? "LATEST"} DEFAULT RATE</span><strong>{latestEvaluation?.rows ? `${(latestEvaluation.defaults / latestEvaluation.rows * 100).toFixed(1)}%` : "—"}</strong></div>
            </div>
            <div className="dataset-footnote" id="methodology">{metadata ? `${metadata.model_type} · Trained on 36-month loans issued ${metadata.training_period}; only Fully Paid, Charged Off, and Default outcomes are used. Open/late loans are excluded from supervised evaluation.` : dataset?.available ? `Quarterly evaluation cohorts: ${dataset.evaluation_cohorts.join(", ")}. Train the model to calculate matured-cohort metrics and prepare drift reference data.` : "Train on resolved loan cohorts and evaluate on later matured cohorts."}</div>
          </section>

          <section className="metrics-grid">
            <MetricCard title={`${latestEvaluationCohort ?? "LATEST"} OUT-OF-TIME ROC AUC`} value={latestEvaluation?.metrics_available ? latestEvaluation.roc_auc.toFixed(3) : "—"} note={metadata ? "Later, matured loan cohort" : "Train the temporal model"} icon={<ShieldCheck size={17} />} trend={latestEvaluation?.metrics_available && latestEvaluation.roc_auc >= 0.7 ? "good" : "neutral"} />
            <MetricCard title="LENDINGCLUB COHORT DRIFTED FEATURES" value={run ? `${drifted} / ${run.feature_count}` : "—"} note={run ? (drifted ? "Review quarterly cohort features" : "All cohort features stable") : "No cohort run yet"} icon={<SlidersHorizontal size={17} />} trend={drifted ? "bad" : "neutral"} />
            <MetricCard title="LENDINGCLUB COHORT MAX PSI" value={run ? Math.max(0, ...Object.values(results).map((item) => item.psi)).toFixed(3) : "—"} note="Highest quarterly cohort PSI" icon={<Activity size={17} />} trend="neutral" />
            <MetricCard title={`${latestEvaluationCohort ?? "LATEST"} MATURED LOANS`} value={latestEvaluation ? latestEvaluation.rows.toLocaleString() : "—"} note={latestEvaluation?.rows ? `${(latestEvaluation.defaults / latestEvaluation.rows * 100).toFixed(1)}% charged off/default` : "Train the temporal model"} icon={<Clock3 size={17} />} trend="neutral" />
          </section>

          <section className="panel incoming-panel" aria-labelledby="incoming-title">
            <div className="panel-heading"><div><h2 id="incoming-title">Incoming loan data</h2><p>Upload a loan batch for results, or download simulated data to try the same workflow</p></div><Upload size={18} aria-hidden="true" /></div>
            <div className="incoming-content">
              <div className="incoming-data-workflow">
                <div className="upload-workflow-heading">
                  <strong>Compare reference and incoming loan data</strong>
                  <span>Upload both LendingClub-format CSVs. Drift and incoming default-risk results will appear here.</span>
                </div>
                <div className="loan-data-inputs">
                  <label className="upload-field"><span>Reference data CSV</span><input type="file" accept=".csv,text/csv" onChange={(event) => { setReferenceLoanFile(event.target.files?.[0] ?? null); setLoanBatchRun(null); }} disabled={busy} /><small>{referenceLoanFile?.name ?? "Choose reference data"}</small></label>
                  <label className="upload-field"><span>Incoming data CSV</span><input type="file" accept=".csv,text/csv" onChange={(event) => { setIncomingLoanFile(event.target.files?.[0] ?? null); setLoanBatchRun(null); }} disabled={busy} /><small>{incomingLoanFile?.name ?? "Choose incoming data"}</small></label>
                </div>
                <div className="incoming-data-actions">
                  <button className="button button-primary incoming-action" onClick={() => void uploadLoanBatch()} disabled={busy || !referenceLoanFile || !incomingLoanFile || !model?.trained}>{busy ? <LoaderCircle className="spin" size={15} /> : <Upload size={15} />} Analyze incoming data</button>
                  <button className="button button-outline incoming-action" onClick={() => void downloadSimulatedLoanCsv("reference")} disabled={busy || !model?.trained}>{busy ? <LoaderCircle className="spin" size={15} /> : <Activity size={15} />} Download reference data</button>
                  <button className="button button-outline incoming-action" onClick={() => void downloadSimulatedLoanCsv("incoming")} disabled={busy || !model?.trained}>{busy ? <LoaderCircle className="spin" size={15} /> : <Activity size={15} />} Download incoming data</button>
                </div>
                {loanBatchRun && <LoanBatchResultsCard run={loanBatchRun} names={names} />}
                <small className="incoming-note">Both CSVs must use the same LendingClub columns and 36-month terms; maximum 20 MB and 100,000 rows per file. Add loan_status to incoming data for outcome metrics. Downloaded reference and incoming files are a matching simulated pair.</small>
              </div>
            </div>
          </section>

          {quarterlyTrend && quarterlyTrend.length > 0 && <section className="panel trend-panel" aria-labelledby="trend-title">
            <div className="panel-heading"><div><h2 id="trend-title">Quarterly performance trends</h2><p>Track model discrimination and loan outcomes across the eight evaluation cohorts</p></div><span className="dataset-badge dataset-ready"><i /> {quarterlyTrend.length} COHORTS</span></div>
            <div className="trend-grid">
              <MetricTrendChart
                title="ROC AUC"
                description="Higher generally indicates stronger ranking performance."
                points={quarterlyTrend.map(({ cohort, metrics }) => ({ cohort, value: metrics.roc_auc }))}
                min={0.5}
                max={1}
                color="#59427d"
                format={(value) => value.toFixed(3)}
              />
              <MetricTrendChart
                title="Default rate"
                description="Charged-off/default share among resolved loans."
                points={quarterlyTrend.map(({ cohort, metrics }) => ({ cohort, value: metrics.defaults / metrics.rows }))}
                min={0}
                max={0.3}
                color="#8668ad"
                format={(value) => `${(value * 100).toFixed(1)}%`}
              />
            </div>
            <div className="panel-footer"><span>Each point represents resolved 36-month loans issued in that quarter.</span><span>Descriptive cohort trends; not a causal explanation.</span></div>
          </section>}

          {runDescription && <div className="data-note"><CircleHelp size={15} /><span>{runDescription}</span></div>}

          {metadata && <section className="panel temporal-panel">
            <div className="panel-heading"><div><h2>Out-of-time model performance</h2><p>Same model evaluated on later, resolved loan cohorts</p></div><span className="dataset-badge dataset-ready"><i /> QUARTERLY BACKTEST</span></div>
            <div className="temporal-table-wrap"><table className="temporal-table"><thead><tr><th>ISSUE COHORT</th><th>RESOLVED LOANS</th><th>DEFAULT RATE</th><th>ROC AUC</th><th>AVG. PRECISION</th><th>BRIER SCORE</th></tr></thead><tbody>
              {dataset?.evaluation_cohorts.map((cohort) => {
                const metrics = metadata.metrics_by_cohort[cohort];
                return <tr key={cohort}><td><strong>{cohort}</strong></td><td>{metrics?.rows.toLocaleString() ?? "—"}</td><td>{metrics?.rows ? `${(metrics.defaults / metrics.rows * 100).toFixed(1)}%` : "—"}</td><td>{metrics?.metrics_available ? metrics.roc_auc.toFixed(3) : "—"}</td><td>{metrics?.metrics_available ? metrics.average_precision.toFixed(3) : "—"}</td><td>{metrics?.metrics_available ? metrics.brier_score.toFixed(3) : "—"}</td></tr>;
              })}
            </tbody></table></div>
            <div className="panel-footer"><span>Resolved loans only; unpaid/current/late loans are excluded from outcome metrics.</span><span>US LendingClub data; not current Indian MFI evidence.</span></div>
          </section>}

          <section className="content-grid" id="features">
            <div className="panel feature-panel">
              <div className="panel-heading"><div><h2>LendingClub quarterly feature drift</h2><p>Actual benchmark cohort compared with the training cohort</p></div><span className="button button-small"><span className="live-dot" /> Quarterly result</span></div>
              <div className="table-scroll"><table><thead><tr><th>FEATURE</th><th>DRIFT SCORE <span className="info-dot">i</span></th><th>TEST / STATISTIC</th><th>STATUS</th></tr></thead><tbody>
                {Object.entries(results).length
                  ? Object.entries(results).map(([name, item]) => (
                    <tr key={name}>
                      <td>
                        <div className="feature-name">
                          <span className={`feature-icon ${item.drift_detected ? "feature-warning" : ""}`}>
                            {item.drift_detected ? <ArrowDownRight size={16} /> : <ArrowUpRight size={16} />}
                          </span>
                          <span><strong>{names[name] ?? name}</strong><small>{name}</small></span>
                        </div>
                      </td>
                      <td>
                        <div className="score-cell">
                          <strong>{item.psi.toFixed(3)}</strong>
                          <div className="score-track">
                            <span
                              className={item.drift_detected ? "score-fill warning-fill" : "score-fill"}
                              style={{ width: `${Math.min(item.psi / 0.3 * 100, 100)}%` }}
                            />
                          </div>
                        </div>
                      </td>
                      <td className="ks-value">
                        <span>{item.test_name === "chi_square" ? "χ²" : "KS"} {item.test_statistic.toFixed(3)}</span>
                        <small>p={item.p_value.toExponential(1)}</small>
                      </td>
                      <td>
                        <span className={`pill ${item.severity === "no_drift" ? "pill-good" : item.severity === "moderate" ? "pill-medium" : "pill-high"}`}>
                          <span />{item.severity.replace("_", " ")}
                        </span>
                      </td>
                    </tr>
                  ))
                  : <tr><td colSpan={4}><div className="empty-state"><div><Sparkles size={20} /></div><strong>Your feature results will show here</strong><span>Train the temporal model to compare matured annual cohorts.</span></div></td></tr>}
              </tbody></table></div>
              <div className="panel-footer"><span>Showing {Object.keys(results).length} monitored features</span><button onClick={() => void refresh()}>View latest results <span>→</span></button></div>
            </div>

            <aside className="panel insight-panel" id="alerts">
              <div className="panel-heading"><div><h2>Monitoring insights</h2><p>What changed in this run</p></div><span className="sparkle-mark"><Sparkles size={17} /></span></div>
              <div className="insight-body">{!run ? <div className="insight-empty"><Activity size={20} /><strong>Insights appear after a run</strong><span>Compare matured quarterly loan cohorts.</span></div> : drifted === 0 ? <div className="insight-message insight-success"><div className="insight-symbol"><Check size={16} /></div><div><strong>No significant drift detected</strong><p>All {run.feature_count} monitored features are below the PSI threshold. {runContext}</p></div></div> : <div className="insight-message insight-warning"><div className="insight-symbol"><TriangleAlert size={16} /></div><div><strong>{drifted} feature{drifted > 1 ? "s" : ""} need review</strong><p>{Object.entries(results).filter(([, v]) => v.drift_detected).map(([k]) => names[k] ?? k).join(", ")} differ across the compared cohorts. {runContext}</p></div></div>}
                <div className="threshold-card"><span>PSI THRESHOLDS</span><div><i className="threshold-green" /> No drift <strong>&lt; 0.10</strong></div><div><i className="threshold-yellow" /> Moderate <strong>0.10–0.20</strong></div><div><i className="threshold-red" /> Severe <strong>&gt; 0.20</strong></div><small>Categorical features also require a significant chi-square test (p&lt;0.05).</small></div>
              </div>
              <button className="insight-link" onClick={() => { setSelectedCohort("2014-Q1"); void runDatasetDrift("2014-Q1"); }} disabled={busy || !model?.trained}>Compare 2014-Q1 <span>→</span></button>
            </aside>
          </section>

          <footer className="page-footer"><span><span className={`footer-dot ${databaseConnected ? "" : "footer-dot-off"}`} />{databaseConnected ? "Connected to local PostgreSQL" : "PostgreSQL unavailable"}</span><span>US LendingClub benchmark <i /> not live MFI monitoring</span></footer>
        </div>
      </main>
    </div>
  );
}

function LoanBatchResultsCard({
  run,
  names,
}: {
  run: LoanBatchRun;
  names: Record<string, string>;
}) {
  const { prediction_summary: summary } = run;
  const driftedFeatures = Object.entries(run.results)
    .filter(([, result]) => result.drift_detected)
    .map(([feature]) => names[feature] ?? feature);
  const evaluation = summary.evaluation;

  return (
    <div className="loan-batch-results" role="status" aria-label="Uploaded loan batch results">
      <strong>Incoming data results</strong>
      <span>Compared {run.incoming_row_count.toLocaleString()} incoming loans with {run.reference_row_count.toLocaleString()} reference loans</span>
      <span>{summary.rows_scored.toLocaleString()} loans scored</span>
      <span>{summary.predicted_defaults_at_50_percent.toLocaleString()} predicted high-risk (≥50%)</span>
      <span>Mean predicted default risk: {(summary.mean_predicted_default_probability * 100).toFixed(1)}%</span>
      <span>Feature drift vs reference: {run.drifted_feature_count} of {run.feature_count} features{driftedFeatures.length ? ` — ${driftedFeatures.join(", ")}` : ""}</span>
      {evaluation && <span>
        Resolved labels: {evaluation.resolved_rows.toLocaleString()}; actual default rate {(evaluation.actual_default_rate * 100).toFixed(1)}%
        {evaluation.roc_auc === null ? "" : `; ROC-AUC ${evaluation.roc_auc.toFixed(3)}`}
      </span>}
    </div>
  );
}

function MetricCard({ title, value, note, icon, trend }: { title: string; value: string; note: string; icon: ReactNode; trend: "good" | "bad" | "neutral" }) {
  return <article className="metric-card"><div className="metric-top"><span>{title}</span><span className="metric-icon">{icon}</span></div><div className="metric-value">{value}</div><div className={`metric-note ${trend === "good" ? "note-good" : trend === "bad" ? "note-bad" : ""}`}>{trend === "good" ? <ArrowUpRight size={14} /> : trend === "bad" ? <ArrowDownRight size={14} /> : <span className="note-line" />}{note}</div></article>;
}

function MetricTrendChart({
  title,
  description,
  points,
  min,
  max,
  color,
  format,
}: {
  title: string;
  description: string;
  points: TrendPoint[];
  min: number;
  max: number;
  color: string;
  format: (value: number) => string;
}) {
  const width = 640;
  const height = 220;
  const left = 44;
  const right = 12;
  const top = 14;
  const plotHeight = 142;
  const plotWidth = width - left - right;
  const coordinates = points.map((point, index) => ({
    ...point,
    x: left + (points.length === 1 ? plotWidth / 2 : (index * plotWidth) / (points.length - 1)),
    y: top + ((max - point.value) / (max - min)) * plotHeight,
  }));
  const path = coordinates
    .map((point, index) => `${index === 0 ? "M" : "L"} ${point.x} ${point.y}`)
    .join(" ");

  return (
    <article className="trend-chart">
      <div className="trend-chart-heading"><div><h3>{title}</h3><p>{description}</p></div><strong>{format(points.at(-1)?.value ?? 0)}</strong></div>
      <svg viewBox={`0 0 ${width} ${height}`} role="img" aria-label={`${title} by quarterly cohort`}>
        {[0, 1, 2, 3].map((step) => {
          const value = max - ((max - min) * step) / 3;
          const y = top + (plotHeight * step) / 3;
          return (
            <g key={step}>
              <line x1={left} x2={width - right} y1={y} y2={y} className="trend-gridline" />
              <text x={left - 8} y={y + 3} textAnchor="end" className="trend-axis-label">{format(value)}</text>
            </g>
          );
        })}
        <path d={path} fill="none" stroke={color} strokeWidth="3" strokeLinecap="round" strokeLinejoin="round" />
        {coordinates.map((point) => (
          <g key={point.cohort}>
            <circle cx={point.x} cy={point.y} r="5" fill={color} stroke="white" strokeWidth="2">
              <title>{point.cohort}: {format(point.value)}</title>
            </circle>
            <text x={point.x} y={top + plotHeight + 22} textAnchor="middle" className="trend-cohort-label">{point.cohort.replace("-", "\n")}</text>
          </g>
        ))}
      </svg>
    </article>
  );
}

export default App;
