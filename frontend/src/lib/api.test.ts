import { afterEach, describe, expect, it, vi } from "vitest";
import { ApiError, predictBatch, predictFlow } from "./api";
import { MODEL_SCHEMA } from "./model-schema";
import type { PredictionResponse } from "../types";

const prediction: PredictionResponse = {
  request_id: "request-1", predicted_class: "BENIGN", is_attack: false,
  probabilities: { BENIGN: 1 }, model_name: "xgboost", model_version: MODEL_SCHEMA.modelVersion,
  predicted_at: "2026-10-29T12:00:00+00:00",
};

afterEach(() => vi.unstubAllGlobals());

describe("FastAPI request contracts", () => {
  it("posts a top-level ordered record to /predict", async () => {
    const fetchMock = vi.fn().mockResolvedValue({ ok: true, json: async () => prediction });
    vi.stubGlobal("fetch", fetchMock);
    const record = Object.fromEntries(MODEL_SCHEMA.features.map(({ name }) => [name, 1]));
    await expect(predictFlow(record)).resolves.toEqual(prediction);
    const [url, init] = fetchMock.mock.calls[0] as [string, RequestInit];
    expect(url).toMatch(/\/predict$/);
    expect(JSON.parse(String(init.body))).toEqual(record);
    expect(JSON.parse(String(init.body))).not.toHaveProperty("Label");
  });

  it("uses the /predict/batch records wrapper and validates result count", async () => {
    const records = [Object.fromEntries(MODEL_SCHEMA.features.map(({ name }) => [name, 0]))];
    const batch = { request_id: "request-1", predictions: [prediction], model_name: "xgboost", model_version: MODEL_SCHEMA.modelVersion, predicted_at: prediction.predicted_at };
    const fetchMock = vi.fn().mockResolvedValue({ ok: true, json: async () => batch });
    vi.stubGlobal("fetch", fetchMock);
    await expect(predictBatch(records)).resolves.toEqual(batch);
    const [, init] = fetchMock.mock.calls[0] as [string, RequestInit];
    expect(JSON.parse(String(init.body))).toEqual({ records });
  });

  it("turns readiness, validation, server, malformed response, and network errors into safe messages", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue({ ok: false, status: 503, json: async () => ({ detail: "internal path" }) }));
    await expect(predictFlow({})).rejects.toMatchObject({ status: 503, message: expect.stringMatching(/not ready/) });

    vi.stubGlobal("fetch", vi.fn().mockResolvedValue({ ok: false, status: 422, json: async () => ({ detail: "missing Flow Duration" }) }));
    await expect(predictFlow({})).rejects.toMatchObject({ status: 422, message: "missing Flow Duration" });

    vi.stubGlobal("fetch", vi.fn().mockResolvedValue({ ok: false, status: 500, json: async () => ({ detail: "Traceback secret path" }) }));
    await expect(predictFlow({})).rejects.toMatchObject({ status: 500, message: "The API could not complete this request. Check backend logs." });

    vi.stubGlobal("fetch", vi.fn().mockResolvedValue({ ok: true, json: async () => ({ unexpected: true }) }));
    await expect(predictFlow({})).rejects.toThrow(/response did not match/);

    vi.stubGlobal("fetch", vi.fn().mockRejectedValue(new TypeError("Failed to fetch")));
    await expect(predictFlow({})).rejects.toThrow(/Cannot reach the API/);
    expect(ApiError).toBeDefined();
  });
});
