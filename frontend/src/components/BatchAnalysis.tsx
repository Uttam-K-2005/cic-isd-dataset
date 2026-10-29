import { useRef, useState, type ChangeEvent } from "react";
import { AlertCircle, ArrowDownToLine, Check, FileSpreadsheet, LoaderCircle, Upload } from "lucide-react";
import { ApiError, predictBatch } from "../lib/api";
import { CsvValidationError, MAX_CSV_BYTES, parseFlowCsv, predictionResultsCsv, type ParsedFlowCsv } from "../lib/csv";
import { MAX_BATCH_SIZE, MODEL_SCHEMA } from "../lib/model-schema";
import type { BatchPredictionResponse } from "../types";
import { PageIntro } from "./DetectionPage";

interface BatchAnalysisProps {
  canPredict: boolean;
  schemaMatches: boolean;
}

export function BatchAnalysis({ canPredict, schemaMatches }: BatchAnalysisProps) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [fileName, setFileName] = useState("");
  const [parsed, setParsed] = useState<ParsedFlowCsv | null>(null);
  const [issues, setIssues] = useState<string[]>([]);
  const [result, setResult] = useState<BatchPredictionResponse | null>(null);
  const [busy, setBusy] = useState(false);

  async function chooseFile(event: ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0];
    setFileName(file?.name ?? "");
    setParsed(null);
    setResult(null);
    setIssues([]);
    if (!file) return;
    if (!file.name.toLowerCase().endsWith(".csv")) {
      setIssues(["Choose a CSV file with a .csv extension."]);
      return;
    }
    if (file.size > MAX_CSV_BYTES) {
      setIssues(["This file is larger than the 5 MB browser limit. Split it into smaller files."]);
      return;
    }
    try {
      const text = await file.text();
      setParsed(parseFlowCsv(text, MODEL_SCHEMA.features, MAX_BATCH_SIZE));
    } catch (reason) {
      setIssues(reason instanceof CsvValidationError ? reason.issues : ["The selected CSV could not be read."]);
    }
  }

  async function submit() {
    if (!parsed || !canPredict) return;
    setBusy(true);
    setIssues([]);
    setResult(null);
    try {
      const response = await predictBatch(parsed.records);
      if (response.predictions.length !== parsed.sourceRows.length) {
        throw new Error("The API returned a different number of predictions than submitted rows.");
      }
      setResult(response);
    } catch (reason) {
      setIssues([reason instanceof ApiError || reason instanceof Error ? reason.message : "Batch prediction failed."]);
    } finally {
      setBusy(false);
    }
  }

  function downloadResults() {
    if (!result || !parsed) return;
    const blob = new Blob([predictionResultsCsv(parsed.sourceRows, result)], { type: "text/csv;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = `adaptive-ids-predictions-${new Date().toISOString().slice(0, 10)}.csv`;
    link.click();
    URL.revokeObjectURL(url);
  }

  return (
    <div className="page-stack">
      <PageIntro
        eyebrow="Threat detection / Batch analysis"
        title="Analyze a flow batch"
        copy="Choose a local CSV. It is parsed in your browser, validated against the model schema, then sent to your configured API as a bounded JSON batch."
      />

      {!schemaMatches && <div className="notice notice-warn" role="alert"><AlertCircle size={17} /> The active API model and generated form schema differ. Batch requests are disabled until you sync the schema.</div>}
      {!canPredict && schemaMatches && <div className="notice notice-warn" role="status"><AlertCircle size={17} /> The model API is not ready. You can select a file, but cannot submit it yet.</div>}
      {issues.length > 0 && (
        <div className="notice notice-error items-start" role="alert">
          <AlertCircle size={17} className="mt-0.5 shrink-0" />
          <div><p className="font-semibold">Batch analysis needs attention</p><ul className="mt-1 list-inside list-disc space-y-1">{issues.map((issue, index) => <li key={`${issue}-${index}`}>{issue}</li>)}</ul></div>
        </div>
      )}

      <section className="panel upload-panel">
        <div className="panel-heading">
          <div>
            <p className="eyebrow">Local file processing</p>
            <h2 className="panel-title">Select a feature CSV</h2>
          </div>
          <span className="schema-chip"><FileSpreadsheet size={14} /> Max {MAX_BATCH_SIZE} rows</span>
        </div>

        <input ref={inputRef} className="sr-only" type="file" accept=".csv,text/csv" onChange={chooseFile} aria-label="Choose flow CSV file" />
        <button type="button" className="upload-dropzone" onClick={() => inputRef.current?.click()} disabled={busy}>
          <span className="upload-icon"><Upload size={20} /></span>
          <span className="block text-sm font-semibold text-slate-100">{fileName || "Choose a CSV to validate"}</span>
          <span className="mt-1 block text-xs text-muted">{fileName ? `${parsed?.records.length ?? 0} valid data rows` : `Exact ${MODEL_SCHEMA.featureCount}-feature header · no Label column · up to ${MAX_BATCH_SIZE} rows`}</span>
          <span className="mt-4 inline-flex items-center gap-2 rounded-lg border border-line bg-panel-2 px-3 py-2 text-xs font-medium text-slate-200">Browse files <Upload size={13} /></span>
        </button>

        {parsed && (
          <div className="mt-4 flex flex-col gap-3 rounded-xl border border-emerald-400/15 bg-emerald-400/[0.045] p-4 sm:flex-row sm:items-center sm:justify-between">
            <div className="flex items-start gap-3">
              <span className="success-icon"><Check size={15} /></span>
              <div>
                <p className="text-sm font-semibold text-slate-100">File is ready to submit</p>
                <p className="mt-1 text-xs text-muted">{parsed.records.length} rows · {MODEL_SCHEMA.featureCount} features · no labels sent</p>
              </div>
            </div>
            <button className="button-primary" type="button" disabled={!canPredict || busy} onClick={() => void submit()}>
              {busy ? <><LoaderCircle size={15} className="animate-spin" /> Predicting…</> : <>Run batch prediction <ArrowDownToLine size={15} /></>}
            </button>
          </div>
        )}

        <p className="mt-4 text-xs leading-5 text-muted">CSV contents stay in this browser until you submit. A valid file’s numeric features are sent only to the API URL shown in the sidebar. Oversized or invalid files are rejected in full—no rows are truncated.</p>
      </section>

      {result && parsed && (
        <BatchResults response={result} sourceRows={parsed.sourceRows} onDownload={downloadResults} />
      )}
    </div>
  );
}

export function BatchResults({
  response,
  sourceRows,
  onDownload,
}: {
  response: BatchPredictionResponse;
  sourceRows: number[];
  onDownload: () => void;
}) {
  return (
    <section className="panel results-table-panel" aria-labelledby="batch-results-title">
      <div className="panel-heading">
        <div>
          <p className="eyebrow">Completed · {response.model_name}{response.model_version ? ` · ${response.model_version}` : ""}</p>
          <h2 className="panel-title" id="batch-results-title">{response.predictions.length} predictions returned</h2>
        </div>
        <button className="button-secondary" onClick={onDownload}><ArrowDownToLine size={15} /> Download CSV</button>
      </div>
      <div className="table-scroll">
        <table className="results-table">
          <thead><tr><th>Data row</th><th>Predicted class</th><th>Category</th><th>Predicted class probability</th></tr></thead>
          <tbody>
            {response.predictions.map((prediction, index) => {
              const probability = prediction.probabilities[prediction.predicted_class];
              return (
                <tr key={`${prediction.request_id}-${index}`}>
                  <td className="font-mono text-muted">{sourceRows[index]}</td>
                  <td className="font-medium text-slate-100">{prediction.predicted_class}</td>
                  <td><span className={`category-pill ${prediction.is_attack ? "category-attack" : "category-benign"}`}>{prediction.is_attack ? "Attack class" : "Benign"}</span></td>
                  <td className="font-mono text-slate-300">{typeof probability === "number" ? `${(probability * 100).toFixed(2)}%` : "Not available"}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      <p className="mt-4 text-xs text-muted">Each row shows the actual API prediction. A predicted attack class is not proof of a confirmed incident.</p>
    </section>
  );
}
