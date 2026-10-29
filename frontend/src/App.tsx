import { useState } from "react";
import {
  Activity,
  ArrowUpRight,
  Clock3,
  FileSpreadsheet,
  Hexagon,
  LayoutDashboard,
  ShieldCheck,
  Siren,
} from "lucide-react";
import { BatchAnalysis } from "./components/BatchAnalysis";
import { DetectionPage } from "./components/DetectionPage";
import { Overview } from "./components/Overview";
import { apiBaseUrl } from "./lib/api";
import { useSystemStatus } from "./lib/useSystemStatus";

type View = "overview" | "detect" | "batch";

const NAV_ITEMS = [
  { id: "overview", label: "Overview", icon: LayoutDashboard },
  { id: "detect", label: "Threat detection", icon: Siren },
  { id: "batch", label: "Batch analysis", icon: FileSpreadsheet },
] as const;

export default function App() {
  const [view, setView] = useState<View>("overview");
  const system = useSystemStatus();
  const activeLabel = NAV_ITEMS.find((item) => item.id === view)?.label ?? "Overview";

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <a className="brand" href="#overview" onClick={(event) => { event.preventDefault(); setView("overview"); }}>
          <span className="brand-mark"><Hexagon size={22} strokeWidth={1.7} /><ShieldCheck className="brand-mark-inner" size={11} /></span>
          <span><strong>Adaptive<span className="brand-accent">IDS</span></strong><small>NETWORK DEFENSE LAB</small></span>
        </a>

        <div className="sidebar-system">
          <div className="flex items-center justify-between">
            <span className="eyebrow">LOCAL SYSTEM</span>
            <span className={`status-dot ${system.health.state === "ok" ? "status-dot-green" : system.health.state === "error" ? "status-dot-red" : "status-dot-muted"}`} />
          </div>
          <p className="mt-2 text-xs font-medium text-slate-200">{system.health.state === "ok" ? "API connection" : system.health.state === "loading" ? "Checking service" : "API offline"}</p>
          <p className="mt-1 truncate font-mono text-[10px] text-muted" title={apiBaseUrl}>{apiBaseUrl}</p>
        </div>

        <nav className="sidebar-nav" aria-label="Main navigation">
          <p className="nav-label">WORKSPACE</p>
          {NAV_ITEMS.map(({ id, label, icon: Icon }) => (
            <button key={id} className={`nav-item ${view === id ? "nav-item-active" : ""}`} onClick={() => setView(id)} aria-current={view === id ? "page" : undefined}>
              <Icon size={17} strokeWidth={1.8} /><span>{label}</span>
              {view === id && <span className="nav-active-marker" />}
            </button>
          ))}
          <p className="nav-label nav-label-spaced">PLANNED</p>
          <button className="nav-item nav-item-disabled" disabled title="Prediction history is not available in the current API">
            <Clock3 size={17} strokeWidth={1.8} /><span>Prediction history</span><span className="nav-soon">LATER</span>
          </button>
        </nav>

        <div className="sidebar-bottom">
          <div className="sidebar-model">
            <span className="model-pulse"><Activity size={14} /></span>
            <div className="min-w-0"><p className="text-xs font-medium text-slate-200">Saved XGBoost</p><p className="mt-0.5 truncate font-mono text-[10px] text-muted">{system.model.state === "ok" ? system.model.value.model_version ?? "Version n/a" : "Model metadata unavailable"}</p></div>
          </div>
          <div className="sidebar-footnote"><span className="status-dot status-dot-cyan" /> LOCAL INFERENCE ONLY</div>
        </div>
      </aside>

      <div className="app-main">
        <header className="topbar">
          <div className="flex min-w-0 items-center gap-3">
            <span className="topbar-section">Workspace</span><span className="topbar-divider">/</span>
            <span className="truncate text-sm font-medium text-slate-200">{activeLabel}</span>
          </div>
          <div className="topbar-right">
            <span className="environment-chip"><span className="status-dot status-dot-cyan" /> LOCAL ENVIRONMENT</span>
            <a className="docs-link" href="http://127.0.0.1:8000/docs" target="_blank" rel="noreferrer">API docs <ArrowUpRight size={13} /></a>
          </div>
        </header>

        <main className="content-area">
          <div className="mobile-nav" aria-label="Mobile navigation">
            {NAV_ITEMS.map(({ id, label, icon: Icon }) => (
              <button key={id} onClick={() => setView(id)} className={view === id ? "mobile-nav-active" : ""} aria-current={view === id ? "page" : undefined}>
                <Icon size={16} /><span>{label}</span>
              </button>
            ))}
          </div>
          {view === "overview" && <Overview status={system} navigate={setView} />}
          {view === "detect" && <DetectionPage canPredict={system.canPredict} schemaMatches={system.schemaMatches} />}
          {view === "batch" && <BatchAnalysis canPredict={system.canPredict} schemaMatches={system.schemaMatches} />}
        </main>
        <footer className="app-footer">
          <span>Adaptive IDS <span className="footer-separator">·</span> CIC-IDS2017 flow classification</span>
          <span>Model results are decision support, not confirmed incidents.</span>
        </footer>
      </div>
    </div>
  );
}
