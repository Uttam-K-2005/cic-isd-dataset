import { describe, expect, it } from "vitest";
import { CsvValidationError, parseFlowCsv, predictionResultsCsv } from "./csv";
import { MODEL_SCHEMA } from "./model-schema";
import type { BatchPredictionResponse } from "../types";

const names = MODEL_SCHEMA.features.map(({ name }) => name);
const header = names.join(",");
const validRow = names.map(() => "0").join(",");

describe("CSV validation for the existing batch API", () => {
  it("parses a complete feature row in metadata order and honors observed negative domains", () => {
    const negativeFeature = MODEL_SCHEMA.features.find((feature) => feature.allowsNegative)!;
    const values = names.map((name) => name === negativeFeature.name ? "-3" : "0");
    const parsed = parseFlowCsv(`${header}\n${values.join(",")}\n`, MODEL_SCHEMA.features, 64);
    expect(parsed.records).toHaveLength(1);
    expect(Object.keys(parsed.records[0])).toEqual(names);
    expect(parsed.records[0][negativeFeature.name]).toBe(-3);
    expect(parsed.sourceRows).toEqual([2]);
  });

  it("rejects Label, missing features, and unexpected columns", () => {
    expect(() => parseFlowCsv(`${header},Label\n${validRow},BENIGN`, MODEL_SCHEMA.features, 64))
      .toThrow(/Label column/);
    expect(() => parseFlowCsv(`${names.slice(1).join(",")}\n${validRow.split(",").slice(1).join(",")}`, MODEL_SCHEMA.features, 64))
      .toThrow(/Missing required columns/);
    expect(() => parseFlowCsv(`${header},Unexpected\n${validRow},x`, MODEL_SCHEMA.features, 64))
      .toThrow(/Unexpected columns/);
  });

  it("reports row and feature for missing, nonnumeric, and negative values", () => {
    const negativeOnly = MODEL_SCHEMA.features.find((feature) => !feature.allowsNegative)!;
    const missing = names.map((name) => name === names[3] ? "" : "0").join(",");
    expect(() => parseFlowCsv(`${header}\n${missing}`, MODEL_SCHEMA.features, 64))
      .toThrow(new RegExp(`Row 2, column “${names[3]}”`));
    const nonnumeric = names.map((name) => name === names[4] ? "not-a-number" : "0").join(",");
    expect(() => parseFlowCsv(`${header}\n${nonnumeric}`, MODEL_SCHEMA.features, 64))
      .toThrow(new RegExp(`Row 2, column “${names[4]}”`));
    const negative = names.map((name) => name === negativeOnly.name ? "-1" : "0").join(",");
    expect(() => parseFlowCsv(`${header}\n${negative}`, MODEL_SCHEMA.features, 64))
      .toThrow(new RegExp(`Row 2, column “${negativeOnly.name}”`));
  });

  it("rejects empty, header-only, malformed, and oversized files without truncating", () => {
    expect(() => parseFlowCsv("  ", MODEL_SCHEMA.features, 64)).toThrow(CsvValidationError);
    expect(() => parseFlowCsv(header, MODEL_SCHEMA.features, 64)).toThrow(/no flow records/);
    expect(() => parseFlowCsv(`\"unterminated\n${validRow}`, MODEL_SCHEMA.features, 64)).toThrow(/CSV parsing error/);
    expect(() => parseFlowCsv(`${header}\n${validRow}\n${validRow}`, MODEL_SCHEMA.features, 1)).toThrow(/No rows were sent/);
  });

  it("exports actual returned results as CSV with probabilities in response order", () => {
    const response: BatchPredictionResponse = {
      request_id: "request-1", model_name: "xgboost", model_version: MODEL_SCHEMA.modelVersion,
      predicted_at: "2026-10-29T12:00:00+00:00",
      predictions: [{
        request_id: "request-1", predicted_class: "BENIGN", is_attack: false,
        probabilities: { BENIGN: 1 }, model_name: "xgboost", model_version: MODEL_SCHEMA.modelVersion,
        predicted_at: "2026-10-29T12:00:00+00:00",
      }],
    };
    const csv = predictionResultsCsv([2], response);
    expect(csv).toContain("csv_data_row,predicted_class,category");
    expect(csv).toContain("2,BENIGN,benign,xgboost");
    expect(csv).toContain("probability:BENIGN");
  });
});
