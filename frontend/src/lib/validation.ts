import type { FeatureDefinition, FlowRecord } from "../types";

export type FieldErrors = Record<string, string>;

export function validateFeatureValues(
  values: Record<string, string>,
  features: FeatureDefinition[],
): { record?: FlowRecord; errors: FieldErrors } {
  const errors: FieldErrors = {};
  const record: FlowRecord = {};
  for (const feature of features) {
    const raw = values[feature.name]?.trim() ?? "";
    if (raw === "") {
      errors[feature.name] = "Required";
      continue;
    }
    const number = Number(raw);
    if (!Number.isFinite(number) || !Number.isFinite(Math.fround(number))) {
      errors[feature.name] = "Enter a finite number within the model's float32 range";
      continue;
    }
    if (!feature.allowsNegative && number < 0) {
      errors[feature.name] = "This feature is nonnegative in the source data";
      continue;
    }
    record[feature.name] = number;
  }
  if (Object.hasOwn(values, "Label")) errors.Label = "Label is the target and cannot be sent to the model";
  return Object.keys(errors).length ? { errors } : { record, errors };
}

export function buildOrderedRecord(
  values: Record<string, string>,
  features: FeatureDefinition[],
): FlowRecord {
  const { record, errors } = validateFeatureValues(values, features);
  if (!record || Object.keys(errors).length) throw new Error("Cannot build a prediction request from invalid fields");
  return Object.fromEntries(features.map(({ name }) => [name, record[name]]));
}
