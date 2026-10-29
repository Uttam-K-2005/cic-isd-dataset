import { ArrowUpRight, ShieldAlert, ShieldCheck } from "lucide-react";
import type { PredictionResponse } from "../types";

export function ProbabilityList({ probabilities }: { probabilities: Record<string, number> }) {
  const sorted = Object.entries(probabilities).sort((left, right) => right[1] - left[1]);
  return (
    <div className="space-y-3" aria-label="Class probabilities">
      {sorted.map(([label, value]) => (
        <div key={label}>
          <div className="mb-1.5 flex items-center justify-between gap-4 text-xs">
            <span className="truncate text-slate-300" title={label}>{label}</span>
            <span className="shrink-0 font-mono tabular-nums text-slate-400">{(value * 100).toFixed(2)}%</span>
          </div>
          <div className="h-1.5 overflow-hidden rounded-full bg-slate-800">
            <div className="h-full rounded-full bg-cyan" style={{ width: `${Math.max(0, Math.min(100, value * 100))}%` }} />
          </div>
        </div>
      ))}
    </div>
  );
}

export function PredictionResult({ prediction }: { prediction: PredictionResponse }) {
  const Icon = prediction.is_attack ? ShieldAlert : ShieldCheck;
  return (
    <section className="panel result-panel" aria-labelledby="prediction-title" aria-live="polite">
      <div className="panel-heading">
        <div>
          <p className="eyebrow">Inference result</p>
          <h2 className="panel-title" id="prediction-title">Flow classification</h2>
        </div>
        <span className="result-live"><span className="status-dot status-dot-green" /> From live API</span>
      </div>

      <div className={`prediction-hero ${prediction.is_attack ? "prediction-attack" : "prediction-benign"}`}>
        <div className="flex items-start gap-4">
          <span className="prediction-icon"><Icon size={22} /></span>
          <div className="min-w-0 flex-1">
            <span className="eyebrow">Predicted class</span>
            <h3 className="mt-1 break-words text-2xl font-semibold tracking-tight text-white">{prediction.predicted_class}</h3>
            <p className="mt-2 text-sm text-slate-300">{prediction.is_attack ? "Attack class predicted" : "Benign class predicted"}</p>
          </div>
          <span className={`category-pill ${prediction.is_attack ? "category-attack" : "category-benign"}`}>
            {prediction.is_attack ? "Attack" : "Benign"}
          </span>
        </div>
        <div className="mt-5 flex flex-wrap items-center gap-x-5 gap-y-2 border-t border-white/10 pt-4 text-xs text-slate-400">
          <span>Model <strong className="ml-1 font-medium text-slate-200">{prediction.model_name}</strong></span>
          <span>Version <strong className="ml-1 font-mono font-medium text-slate-200">{prediction.model_version ?? "Not available"}</strong></span>
          <span className="inline-flex items-center gap-1">ID <span className="font-mono">{prediction.request_id.slice(0, 8)}</span><ArrowUpRight size={12} /></span>
        </div>
      </div>

      <div className="mt-6">
        <div className="mb-4 flex items-end justify-between gap-3">
          <div>
            <h3 className="text-sm font-semibold text-slate-100">Class probabilities</h3>
            <p className="mt-1 text-xs text-muted">Raw model outputs · not calibrated confidence</p>
          </div>
          <span className="font-mono text-xs text-muted">{Object.keys(prediction.probabilities).length} classes</span>
        </div>
        <ProbabilityList probabilities={prediction.probabilities} />
      </div>

      <p className="notice notice-neutral mt-5">A prediction is a machine-learning classification, not proof of a confirmed security incident.</p>
    </section>
  );
}
