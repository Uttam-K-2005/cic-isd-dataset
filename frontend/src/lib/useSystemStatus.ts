import { useCallback, useEffect, useState } from "react";
import { ApiError, getHealth, getModelMetadata, getReadiness } from "./api";
import { MODEL_SCHEMA } from "./model-schema";
import type { HealthResponse, ModelMetadataResponse } from "../types";

type Check<T> = { state: "loading" } | { state: "ok"; value: T } | { state: "error"; message: string };
type ReadyCheck = { state: "loading" } | { state: "ready" } | { state: "not-ready" } | { state: "error"; message: string };

export function useSystemStatus() {
  const [health, setHealth] = useState<Check<HealthResponse>>({ state: "loading" });
  const [readiness, setReadiness] = useState<ReadyCheck>({ state: "loading" });
  const [model, setModel] = useState<Check<ModelMetadataResponse>>({ state: "loading" });

  const refresh = useCallback(async () => {
    const [healthResult, readyResult, modelResult] = await Promise.allSettled([
      getHealth(), getReadiness(), getModelMetadata(),
    ]);
    setHealth(healthResult.status === "fulfilled" && healthResult.value.status === "ok"
      ? { state: "ok", value: healthResult.value }
      : { state: "error", message: healthResult.status === "rejected" ? healthResult.reason.message : "Unexpected health response" });
    if (readyResult.status === "fulfilled" && readyResult.value.model_loaded) setReadiness({ state: "ready" });
    else if (readyResult.status === "rejected" && readyResult.reason instanceof ApiError && readyResult.reason.status === 503) {
      setReadiness({ state: "not-ready" });
    } else {
      setReadiness({ state: "error", message: readyResult.status === "rejected" ? readyResult.reason.message : "Unexpected readiness response" });
    }
    setModel(modelResult.status === "fulfilled"
      ? { state: "ok", value: modelResult.value }
      : { state: "error", message: modelResult.reason.message });
  }, []);

  useEffect(() => {
    void refresh();
    const interval = window.setInterval(() => void refresh(), 30_000);
    return () => window.clearInterval(interval);
  }, [refresh]);

  const schemaMatches = model.state === "ok"
    && model.value.model_name === MODEL_SCHEMA.modelName
    && model.value.model_version === MODEL_SCHEMA.modelVersion
    && JSON.stringify(model.value.supported_classes) === JSON.stringify(MODEL_SCHEMA.labelClasses);
  const canPredict = health.state === "ok" && readiness.state === "ready" && schemaMatches;
  return { health, readiness, model, schemaMatches, canPredict, refresh };
}

export type SystemStatus = ReturnType<typeof useSystemStatus>;
