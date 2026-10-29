import { useMemo, useState, type FormEvent } from "react";
import { AlertCircle, ArrowRight, Search, SlidersHorizontal } from "lucide-react";
import { MODEL_SCHEMA } from "../lib/model-schema";
import { validateFeatureValues } from "../lib/validation";
import type { FeatureDefinition, FlowRecord } from "../types";

interface FeatureFormProps {
  disabled?: boolean;
  onSubmit: (record: FlowRecord) => void;
}

export function FeatureForm({ disabled = false, onSubmit }: FeatureFormProps) {
  const [values, setValues] = useState<Record<string, string>>({});
  const [errors, setErrors] = useState<Record<string, string>>({});
  const [query, setQuery] = useState("");
  const [submitted, setSubmitted] = useState(false);
  const groups = useMemo(() => {
    const grouped = new Map<string, FeatureDefinition[]>();
    for (const feature of MODEL_SCHEMA.features) {
      const list = grouped.get(feature.group) ?? [];
      list.push(feature);
      grouped.set(feature.group, list);
    }
    return [...grouped.entries()];
  }, []);
  const filteredGroups = groups
    .map(([group, features]) => [group, features.filter((feature) => feature.name.toLowerCase().includes(query.toLowerCase()))] as const)
    .filter(([, features]) => features.length > 0);
  const filledCount = MODEL_SCHEMA.features.filter(({ name }) => Boolean(values[name]?.trim())).length;
  const errorGroups = new Set(MODEL_SCHEMA.features.filter(({ name }) => errors[name]).map(({ group }) => group));

  function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setSubmitted(true);
    const result = validateFeatureValues(values, MODEL_SCHEMA.features);
    setErrors(result.errors);
    if (result.record) onSubmit(result.record);
  }

  function update(name: string, value: string) {
    setValues((current) => ({ ...current, [name]: value }));
    setErrors((current) => {
      if (!current[name]) return current;
      const next = { ...current };
      delete next[name];
      return next;
    });
  }

  return (
    <form className="space-y-5" noValidate onSubmit={submit}>
      <div className="flex flex-col gap-3 rounded-xl border border-line bg-panel-2 p-4 sm:flex-row sm:items-center sm:justify-between">
        <div>
          <div className="flex items-center gap-2 text-sm font-semibold text-slate-100">
            <SlidersHorizontal size={16} className="text-cyan" /> Flow feature vector
          </div>
          <p className="mt-1 text-xs text-muted">All {MODEL_SCHEMA.featureCount} fields are required. Values follow the saved model’s float32 input contract.</p>
        </div>
        <div className="flex items-center gap-3">
          <div className="h-1.5 w-24 overflow-hidden rounded-full bg-slate-800" aria-label={`${filledCount} of ${MODEL_SCHEMA.featureCount} fields filled`}>
            <div className="h-full rounded-full bg-cyan transition-all" style={{ width: `${(filledCount / MODEL_SCHEMA.featureCount) * 100}%` }} />
          </div>
          <span className="font-mono text-xs text-muted">{filledCount}/{MODEL_SCHEMA.featureCount}</span>
        </div>
      </div>

      <label className="relative block max-w-md">
        <Search size={16} className="pointer-events-none absolute left-3 top-1/2 -translate-y-1/2 text-muted" />
        <span className="sr-only">Filter feature fields</span>
        <input
          className="control pl-9"
          placeholder="Find a feature by name…"
          value={query}
          onChange={(event) => setQuery(event.target.value)}
          type="search"
        />
      </label>

      {submitted && Object.keys(errors).length > 0 && (
        <div className="notice notice-error" role="alert">
          <AlertCircle size={17} />
          <span>{Object.keys(errors).length} field{Object.keys(errors).length === 1 ? "" : "s"} need attention before submitting.</span>
        </div>
      )}

      <div className="space-y-3">
        {filteredGroups.map(([group, features], groupIndex) => (
          <details
            className="feature-section group"
            key={group}
            open={query.length > 0 || errorGroups.has(group) || (groupIndex === 0 && query.length === 0)}
          >
            <summary className="feature-summary">
              <span>
                <span className="block text-sm font-semibold text-slate-100">{group}</span>
                <span className="mt-0.5 block text-xs text-muted">{features.length} model input{features.length === 1 ? "" : "s"}</span>
              </span>
              <span className="summary-chevron" aria-hidden="true">⌄</span>
            </summary>
            <div className="feature-grid">
              {features.map((feature) => (
                <FeatureField
                  key={feature.name}
                  feature={feature}
                  value={values[feature.name] ?? ""}
                  error={errors[feature.name]}
                  onChange={update}
                />
              ))}
            </div>
          </details>
        ))}
      </div>

      <div className="flex flex-col gap-3 border-t border-line pt-5 sm:flex-row sm:items-center sm:justify-between">
        <p className="max-w-xl text-xs leading-5 text-muted">A classification is a model output for this flow. It is not confirmation of an incident or an automated response.</p>
        <button className="button-primary shrink-0" type="submit" disabled={disabled}>
          {disabled ? "Classifying…" : "Classify flow"}
          {!disabled && <ArrowRight size={16} />}
        </button>
      </div>
    </form>
  );
}

function FeatureField({
  feature,
  value,
  error,
  onChange,
}: {
  feature: FeatureDefinition;
  value: string;
  error?: string;
  onChange: (name: string, value: string) => void;
}) {
  const id = `feature-${feature.name.replace(/[^a-zA-Z0-9]+/g, "-").toLowerCase()}`;
  return (
    <div className="min-w-0">
      <label htmlFor={id} className="mb-1.5 flex min-h-5 items-center justify-between gap-2 text-xs font-medium text-slate-300">
        <span className="truncate" title={feature.name}>{feature.name}</span>
        {feature.allowsNegative && <span className="shrink-0 text-[10px] text-cyan" title="Negative values occur in the source data">±</span>}
      </label>
      <input
        id={id}
        name={feature.name}
        type="number"
        step="any"
        min={feature.allowsNegative ? undefined : 0}
        inputMode="decimal"
        className={`control font-mono text-sm ${error ? "control-invalid" : ""}`}
        value={value}
        onChange={(event) => onChange(feature.name, event.target.value)}
        aria-invalid={Boolean(error)}
        aria-describedby={error ? `${id}-error` : undefined}
      />
      {error && <p className="mt-1 text-[11px] leading-4 text-rose-300" id={`${id}-error`}>{error}</p>}
    </div>
  );
}
