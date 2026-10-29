import Papa from "papaparse";
import type { BatchPredictionResponse, FeatureDefinition, FlowRecord, PredictionResponse } from "../types";

export interface ParsedFlowCsv {
  records: FlowRecord[];
  sourceRows: number[];
}

export class CsvValidationError extends Error {
  constructor(readonly issues: string[]) {
    super(issues[0] ?? "The CSV could not be validated.");
    this.name = "CsvValidationError";
  }
}

export const MAX_CSV_BYTES = 5_000_000;

export function parseFlowCsv(text: string, features: FeatureDefinition[], maxBatchSize: number): ParsedFlowCsv {
  if (!text.trim()) throw new CsvValidationError(["The CSV file is empty."]);
  const parsed = Papa.parse<Record<string, string>>(text, {
    header: true,
    skipEmptyLines: "greedy",
    dynamicTyping: false,
    transformHeader: (header) => header.replace(/^\uFEFF/, "").trim(),
  });
  const fatalParseErrors = parsed.errors.filter((issue) => issue.code !== "TooFewFields" && issue.code !== "TooManyFields");
  if (fatalParseErrors.length) {
    const first = fatalParseErrors[0];
    throw new CsvValidationError([`CSV parsing error near data row ${Math.max(1, first.row ?? 0)}: ${first.message}`]);
  }
  const headers = parsed.meta.fields ?? [];
  if (headers.length === 0) throw new CsvValidationError(["The CSV needs a header row with the model feature names."]);
  if (headers.includes("Label")) {
    throw new CsvValidationError(["This CSV contains a Label column. Remove target labels before requesting predictions."]);
  }
  const duplicates = headers.filter((name, index) => headers.indexOf(name) !== index);
  if (duplicates.length) throw new CsvValidationError([`Duplicate columns: ${[...new Set(duplicates)].join(", ")}`]);
  const expected = features.map(({ name }) => name);
  const missing = expected.filter((name) => !headers.includes(name));
  const unexpected = headers.filter((name) => !expected.includes(name));
  if (missing.length || unexpected.length) {
    const issues = [];
    if (missing.length) issues.push(`Missing required columns (${missing.length}): ${missing.slice(0, 8).join(", ")}${missing.length > 8 ? ", …" : ""}`);
    if (unexpected.length) issues.push(`Unexpected columns (${unexpected.length}): ${unexpected.slice(0, 8).join(", ")}${unexpected.length > 8 ? ", …" : ""}`);
    throw new CsvValidationError(issues);
  }
  if (parsed.data.length === 0) throw new CsvValidationError(["The CSV has a header but no flow records."]);
  if (parsed.data.length > maxBatchSize) {
    throw new CsvValidationError([`The file contains ${parsed.data.length} rows; the current frontend batch limit is ${maxBatchSize}. No rows were sent.`]);
  }

  const featureByName = new Map(features.map((feature) => [feature.name, feature]));
  const records: FlowRecord[] = [];
  const sourceRows: number[] = [];
  const issues: string[] = [];
  parsed.data.forEach((row, index) => {
    const dataRow = index + 2;
    const record: FlowRecord = {};
    if (Object.hasOwn(row, "__parsed_extra")) {
      issues.push(`Row ${dataRow}, column “extra value”: too many fields for the CSV header.`);
    }
    for (const name of expected) {
      const raw = row[name]?.trim() ?? "";
      if (!raw) {
        issues.push(`Row ${dataRow}, column “${name}”: value is required.`);
      } else {
        const value = Number(raw);
        if (!Number.isFinite(value) || !Number.isFinite(Math.fround(value))) {
          issues.push(`Row ${dataRow}, column “${name}”: enter a finite number within float32 range.`);
        } else if (!featureByName.get(name)!.allowsNegative && value < 0) {
          issues.push(`Row ${dataRow}, column “${name}”: negative values are not valid for this feature.`);
        } else {
          record[name] = value;
        }
      }
    }
    records.push(record);
    sourceRows.push(dataRow);
  });
  if (issues.length) throw new CsvValidationError(issues.slice(0, 100).concat(issues.length > 100 ? [`And ${issues.length - 100} more validation issues.`] : []));
  return { records, sourceRows };
}

function csvCell(value: string | number | boolean): string {
  const text = String(value);
  return /[",\r\n]/.test(text) ? `"${text.replace(/"/g, '""')}"` : text;
}

export function predictionResultsCsv(
  sourceRows: number[],
  response: BatchPredictionResponse,
): string {
  const labels = Object.keys(response.predictions[0]?.probabilities ?? {});
  const headers = ["csv_data_row", "predicted_class", "category", "model_name", "model_version", "predicted_at", ...labels.map((label) => `probability:${label}`)];
  const lines = [headers.map(csvCell).join(",")];
  response.predictions.forEach((prediction: PredictionResponse, index) => {
    const cells: Array<string | number | boolean> = [
      sourceRows[index], prediction.predicted_class, prediction.is_attack ? "attack" : "benign",
      prediction.model_name, prediction.model_version ?? "", prediction.predicted_at,
      ...labels.map((label) => prediction.probabilities[label]),
    ];
    lines.push(cells.map(csvCell).join(","));
  });
  return `${lines.join("\r\n")}\r\n`;
}
