import { Activity, ArrowRight, Cpu, FileSpreadsheet, RefreshCw, Shield, Wifi } from "lucide-react";
import type { ReactNode } from "react";
import { apiBaseUrl } from "../lib/api";
import { MODEL_SCHEMA } from "../lib/model-schema";
import type { SystemStatus } from "../lib/useSystemStatus";

interface OverviewProps {
  status: SystemStatus;
  navigate: (view: "detect" | "batch") => void;
}

export function Overview({ status, navigate }: OverviewProps) {
  const apiOnline = status.health.state === "ok";
  const modelReady = status.readiness.state === "ready";
  const model = status.model.state === "ok" ? status.model.value : null;

  return (
    <div className="page-stack">
      <section className="hero-panel">
        <div className="hero-orb hero-orb-one" />
        <div className="hero-orb hero-orb-two" />
        <div className="relative z-10 max-w-2xl">
          <span className="hero-kicker"><span className="hero-kicker-dot" /> NETWORK FLOW CLASSIFICATION</span>
          <h1 className="hero-title">Know what’s moving<br className="hidden sm:block" /> across your network.</h1>
          <p className="hero-copy">Classify CIC-IDS2017 flow features with a saved XGBoost model through your local inference API. Predictions support analysis; they do not confirm an incident.</p>
          <div className="mt-7 flex flex-wrap gap-3">
            <button className="button-primary" onClick={() => navigate("detect")}>
              Analyze a flow <ArrowRight size={16} />
            </button>
            <button className="button-secondary" onClick={() => navigate("batch")}>
              <FileSpreadsheet size={16} /> Analyze a CSV
            </button>
          </div>
        </div>
        <div className="hero-visual" aria-hidden="true">
          <div className="hero-ring hero-ring-outer" />
          <div className="hero-ring hero-ring-inner" />
          <div className="hero-core"><Shield size={30} strokeWidth={1.4} /></div>
          <span className="orbit-node orbit-node-top"><Activity size={15} /></span>
          <span className="orbit-node orbit-node-side"><Cpu size={15} /></span>
          <span className="orbit-node orbit-node-bottom"><Wifi size={15} /></span>
        </div>
        <div className="hero-footer">
          <span className="inline-flex items-center gap-2"><span className="status-dot status-dot-cyan" /> LOCAL INFERENCE</span>
          <span>77 model features</span>
          <span>15 supported classes</span>
        </div>
      </section>

      <div className="section-heading">
        <div>
          <p className="eyebrow">System status</p>
          <h2 className="section-title">Your detection service</h2>
        </div>
        <button className="button-quiet" onClick={() => void status.refresh()} aria-label="Refresh system status">
          <RefreshCw size={15} /> Refresh
        </button>
      </div>

      <div className="grid gap-4 md:grid-cols-3">
        <StatusCard
          icon={<Wifi size={17} />}
          label="API connectivity"
          value={status.health.state === "loading" ? "Checking" : apiOnline ? "Connected" : "Unavailable"}
          tone={status.health.state === "loading" ? "neutral" : apiOnline ? "good" : "bad"}
          detail={apiOnline ? apiBaseUrl : status.health.state === "error" ? status.health.message : "Waiting for API"}
        />
        <StatusCard
          icon={<Shield size={17} />}
          label="Model readiness"
          value={status.readiness.state === "loading" ? "Checking" : modelReady ? "Ready" : status.readiness.state === "not-ready" ? "Not ready" : "Unknown"}
          tone={status.readiness.state === "ready" ? "good" : status.readiness.state === "not-ready" ? "warn" : status.readiness.state === "error" ? "bad" : "neutral"}
          detail={status.readiness.state === "error" ? status.readiness.message : "Read from GET /ready"}
        />
        <StatusCard
          icon={<Cpu size={17} />}
          label="Loaded model"
          value={model ? model.model_name : status.model.state === "loading" ? "Checking" : "Not available"}
          tone={model ? "good" : status.model.state === "error" ? "warn" : "neutral"}
          detail={model ? `Version ${model.model_version ?? "not reported"}` : status.model.state === "error" ? status.model.message : "Read from GET /models"}
        />
      </div>

      {model && !status.schemaMatches && (
        <div className="notice notice-warn" role="alert">
          <span className="notice-symbol">!</span>
          <span>The API model schema does not match the frontend’s generated {MODEL_SCHEMA.modelVersion} feature schema. Refresh the schema from the configured artifact before sending predictions.</span>
        </div>
      )}

      <section className="panel about-panel">
        <div className="about-icon"><Activity size={19} /></div>
        <div>
          <p className="eyebrow">About this system</p>
          <h2 className="text-base font-semibold text-slate-100">Machine-learning network-flow classification</h2>
          <p className="mt-2 max-w-3xl text-sm leading-6 text-muted">Adaptive IDS sends numeric CIC-IDS2017 flow features to the existing FastAPI service. The backend returns a class prediction and per-class model probabilities. The selected model was trained on a class-capped subset, so its published evaluation is not a full-dataset or natural-prevalence estimate.</p>
        </div>
      </section>

      <div className="grid gap-4 lg:grid-cols-2">
        <ActionCard
          icon={<Activity size={18} />}
          title="Single-flow detection"
          copy="Enter the model’s exact 77 numeric features and inspect the returned class probabilities."
          action="Open flow analyzer"
          onClick={() => navigate("detect")}
        />
        <ActionCard
          icon={<FileSpreadsheet size={18} />}
          title="Batch CSV analysis"
          copy={`Validate and classify up to ${MODEL_SCHEMA.batchLimitDefault} records per request using the existing batch endpoint.`}
          action="Upload a CSV"
          onClick={() => navigate("batch")}
        />
      </div>
    </div>
  );
}

function StatusCard({ icon, label, value, tone, detail }: { icon: ReactNode; label: string; value: string; tone: string; detail: string }) {
  return (
    <article className="panel status-card">
      <div className="flex items-center justify-between">
        <span className="status-icon">{icon}</span>
        <span className={`status-pill status-pill-${tone}`}><span className="status-dot" />{value}</span>
      </div>
      <p className="mt-4 text-xs font-medium uppercase tracking-[0.12em] text-muted">{label}</p>
      <p className="mt-2 truncate text-sm text-slate-200" title={detail}>{detail}</p>
    </article>
  );
}

function ActionCard({ icon, title, copy, action, onClick }: { icon: ReactNode; title: string; copy: string; action: string; onClick: () => void }) {
  return (
    <button className="panel action-card group text-left" onClick={onClick}>
      <span className="action-icon">{icon}</span>
      <div className="mt-4 flex items-start justify-between gap-3">
        <div>
          <h3 className="text-sm font-semibold text-slate-100">{title}</h3>
          <p className="mt-2 max-w-md text-sm leading-6 text-muted">{copy}</p>
        </div>
        <ArrowRight className="mt-0.5 shrink-0 text-cyan transition-transform group-hover:translate-x-1" size={17} />
      </div>
      <span className="mt-5 inline-flex items-center gap-1 text-xs font-semibold text-cyan">{action}<ArrowRight size={13} /></span>
    </button>
  );
}
