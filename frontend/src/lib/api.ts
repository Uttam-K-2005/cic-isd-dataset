import type {
  BatchPredictionResponse,
  FlowRecord,
  HealthResponse,
  ModelMetadataResponse,
  PredictionResponse,
  ReadinessResponse,
} from "../types";

const API_BASE_URL = (import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8000").replace(/\/+$/, "");
const REQUEST_TIMEOUT_MS = 15_000;

export class ApiError extends Error {
  constructor(
    message: string,
    readonly status?: number,
  ) {
    super(message);
    this.name = "ApiError";
  }
}

function isObject(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

async function requestJson(path: string, init?: RequestInit): Promise<unknown> {
  let response: Response;
  try {
    response = await fetch(`${API_BASE_URL}${path}`, {
      ...init,
      signal: AbortSignal.timeout(REQUEST_TIMEOUT_MS),
      headers: { Accept: "application/json", ...init?.headers },
    });
  } catch (error) {
    if (error instanceof DOMException && error.name === "TimeoutError") {
      throw new ApiError("The API request timed out. Check the backend and try again.");
    }
    throw new ApiError("Cannot reach the API. Make sure the backend is running and try again.");
  }

  let body: unknown;
  try {
    body = await response.json();
  } catch {
    throw new ApiError("The API returned an unreadable response.", response.status);
  }
  if (!response.ok) {
    const detail = isObject(body) && typeof body.detail === "string" ? body.detail : "";
    if (response.status === 503) throw new ApiError("The model is not ready yet. Check backend readiness.", 503);
    if (response.status === 429) throw new ApiError("The API is busy. Wait a moment and try again.", 429);
    if (response.status === 422) throw new ApiError(detail || "The submitted flow does not match the model input schema.", 422);
    if (response.status >= 500) throw new ApiError("The API could not complete this request. Check backend logs.", response.status);
    throw new ApiError(detail || `The API rejected the request (${response.status}).`, response.status);
  }
  return body;
}

function parsePrediction(value: unknown): PredictionResponse {
  if (!isObject(value) || typeof value.predicted_class !== "string" || typeof value.is_attack !== "boolean"
    || typeof value.request_id !== "string" || typeof value.model_name !== "string"
    || !(typeof value.model_version === "string" || value.model_version === null)
    || typeof value.predicted_at !== "string" || !isObject(value.probabilities)) {
    throw new ApiError("The API prediction response did not match the expected format.");
  }
  const probabilities: Record<string, number> = {};
  for (const [label, probability] of Object.entries(value.probabilities)) {
    if (typeof probability !== "number" || !Number.isFinite(probability)) {
      throw new ApiError("The API returned invalid class probabilities.");
    }
    probabilities[label] = probability;
  }
  if (Object.keys(probabilities).length === 0) throw new ApiError("The API returned no class probabilities.");
  return { ...value, probabilities } as PredictionResponse;
}

export async function getHealth(): Promise<HealthResponse> {
  const value = await requestJson("/health");
  if (!isObject(value) || typeof value.status !== "string") throw new ApiError("The API health response is invalid.");
  return { status: value.status };
}

export async function getReadiness(): Promise<ReadinessResponse> {
  const value = await requestJson("/ready");
  if (!isObject(value) || typeof value.status !== "string" || typeof value.model_loaded !== "boolean") {
    throw new ApiError("The API readiness response is invalid.");
  }
  return { status: value.status, model_loaded: value.model_loaded };
}

export async function getModelMetadata(): Promise<ModelMetadataResponse> {
  const value = await requestJson("/models");
  if (!isObject(value) || typeof value.model_name !== "string"
    || !(typeof value.model_version === "string" || value.model_version === null)
    || !Array.isArray(value.supported_classes) || !value.supported_classes.every((label) => typeof label === "string")) {
    throw new ApiError("The API model metadata response is invalid.");
  }
  return {
    model_name: value.model_name,
    model_version: value.model_version,
    supported_classes: value.supported_classes as string[],
  };
}

export async function predictFlow(record: FlowRecord): Promise<PredictionResponse> {
  const value = await requestJson("/predict", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(record),
  });
  return parsePrediction(value);
}

export async function predictBatch(records: FlowRecord[]): Promise<BatchPredictionResponse> {
  const value = await requestJson("/predict/batch", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ records }),
  });
  if (!isObject(value) || typeof value.request_id !== "string" || !Array.isArray(value.predictions)
    || value.predictions.length !== records.length || typeof value.model_name !== "string"
    || !(typeof value.model_version === "string" || value.model_version === null)
    || typeof value.predicted_at !== "string") {
    throw new ApiError("The API batch response did not match the expected format.");
  }
  return {
    request_id: value.request_id,
    predictions: value.predictions.map(parsePrediction),
    model_name: value.model_name,
    model_version: value.model_version,
    predicted_at: value.predicted_at,
  };
}

export const apiBaseUrl = API_BASE_URL;
