import { useState } from "react";
import { AlertCircle, CircleCheck, LoaderCircle } from "lucide-react";
import { FeatureForm } from "./FeatureForm";
import { PredictionResult } from "./PredictionResult";
import { ApiError, predictFlow } from "../lib/api";
import { MODEL_SCHEMA } from "../lib/model-schema";
import type { FlowRecord, PredictionResponse } from "../types";

export function DetectionPage({ canPredict, schemaMatches }: { canPredict: boolean; schemaMatches: boolean }) {
  const [prediction, setPrediction] = useState<PredictionResponse | null>(null);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);

  async function submit(record: FlowRecord) {
    setError("");
    setPrediction(null);
    setBusy(true);
    try {
      setPrediction(await predictFlow(record));
    } catch (reason) {
      setError(reason instanceof ApiError ? reason.message : "The prediction could not be completed.");
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="page-stack">
      <PageIntro
        eyebrow="Threat detection / Single flow"
        title="Classify a network flow"
        copy="Enter a complete feature record. The request is sent to your local FastAPI service and classified by the saved XGBoost model."
      />

      {!schemaMatches && <div className="notice notice-warn" role="alert"><AlertCircle size={17} /> The active API model and generated form schema differ. Prediction requests are disabled until you sync the schema.</div>}
      {!canPredict && schemaMatches && <div className="notice notice-warn" role="status"><AlertCircle size={17} /> The model API is not ready. Check system status and retry when it is available.</div>}
      {error && <div className="notice notice-error" role="alert"><AlertCircle size={17} /><span>{error}</span></div>}
      {prediction && <PredictionResult prediction={prediction} />}

      <section className="panel form-panel">
        <div className="panel-heading">
          <div>
            <p className="eyebrow">Input schema · model {MODEL_SCHEMA.modelVersion}</p>
            <h2 className="panel-title">Flow feature values</h2>
          </div>
          <span className="schema-chip"><CircleCheck size={14} /> Schema verified</span>
        </div>
        <FeatureForm disabled={busy || !canPredict} onSubmit={submit} />
      </section>

      {busy && <div className="fixed bottom-5 right-5 z-20 flex items-center gap-3 rounded-xl border border-cyan/20 bg-panel px-4 py-3 text-sm text-slate-200 shadow-panel" role="status"><LoaderCircle className="animate-spin text-cyan" size={17} /> Sending to the inference API…</div>}
    </div>
  );
}

export function PageIntro({ eyebrow, title, copy }: { eyebrow: string; title: string; copy: string }) {
  return (
    <div className="page-intro">
      <p className="eyebrow">{eyebrow}</p>
      <h1 className="page-title">{title}</h1>
      <p className="page-copy">{copy}</p>
    </div>
  );
}
