import { describe, expect, it, vi } from "vitest";
import { fireEvent, render, screen } from "@testing-library/react";
import { FeatureForm } from "./FeatureForm";
import { PredictionResult } from "./PredictionResult";
import { BatchResults } from "./BatchAnalysis";
import { MODEL_SCHEMA } from "../lib/model-schema";
import type { BatchPredictionResponse, PredictionResponse } from "../types";

const prediction: PredictionResponse = {
  request_id: "id-12345678", predicted_class: "DoS Hulk", is_attack: true,
  probabilities: { BENIGN: 0.12, "DoS Hulk": 0.88 }, model_name: "xgboost",
  model_version: MODEL_SCHEMA.modelVersion, predicted_at: "2026-10-29T12:00:00+00:00",
};

describe("React feature and result components", () => {
  it("generates the complete 77-field form from metadata and excludes Label", () => {
    render(<FeatureForm onSubmit={vi.fn()} />);
    const inputs = screen.getAllByRole("spinbutton");
    expect(inputs).toHaveLength(77);
    expect(new Set(inputs.map((input) => input.getAttribute("name")))).toEqual(new Set(MODEL_SCHEMA.features.map(({ name }) => name)));
    expect(screen.queryByLabelText("Label")).not.toBeInTheDocument();
  });

  it("shows validation errors instead of submitting incomplete values", () => {
    const onSubmit = vi.fn();
    render(<FeatureForm onSubmit={onSubmit} />);
    fireEvent.click(screen.getByRole("button", { name: /classify flow/i }));
    expect(onSubmit).not.toHaveBeenCalled();
    expect(screen.getByRole("alert")).toHaveTextContent("77 fields need attention");
  });

  it("renders actual prediction class, category, and probability values", () => {
    render(<PredictionResult prediction={prediction} />);
    expect(screen.getByRole("heading", { name: "DoS Hulk" })).toBeInTheDocument();
    expect(screen.getByText("Attack class predicted")).toBeInTheDocument();
    expect(screen.getByText("88.00%")).toBeInTheDocument();
    expect(screen.getByText(/not proof of a confirmed security incident/i)).toBeInTheDocument();
  });

  it("renders batch response rows and their real API categories", () => {
    const response: BatchPredictionResponse = {
      request_id: "batch-1", predictions: [prediction], model_name: "xgboost",
      model_version: MODEL_SCHEMA.modelVersion, predicted_at: prediction.predicted_at,
    };
    render(<BatchResults response={response} sourceRows={[2]} onDownload={vi.fn()} />);
    expect(screen.getByRole("heading", { name: "1 predictions returned" })).toBeInTheDocument();
    expect(screen.getByRole("cell", { name: "2" })).toBeInTheDocument();
    expect(screen.getByRole("cell", { name: "DoS Hulk" })).toBeInTheDocument();
    expect(screen.getByText("Attack class")).toBeInTheDocument();
  });
});
