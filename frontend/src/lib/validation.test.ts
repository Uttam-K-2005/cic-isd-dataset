import { describe, expect, it } from "vitest";
import { MODEL_SCHEMA } from "./model-schema";
import { buildOrderedRecord, validateFeatureValues } from "./validation";

describe("model feature schema and form validation", () => {
  it("uses the artifact schema with 77 ordered numeric features and no target", () => {
    expect(MODEL_SCHEMA.features).toHaveLength(77);
    expect(MODEL_SCHEMA.features[0].name).toBe("Destination Port");
    expect(MODEL_SCHEMA.features.at(-1)?.name).toBe("Idle Min");
    expect(MODEL_SCHEMA.features.some((feature) => feature.name === "Label")).toBe(false);
    expect(MODEL_SCHEMA.features.every((feature) => feature.dtype === "number")).toBe(true);
  });

  it("requires each feature, rejects invalid values, and orders payload from model metadata", () => {
    const validValues = Object.fromEntries(MODEL_SCHEMA.features.map(({ name }) => [name, "0"]));
    const result = validateFeatureValues(validValues, MODEL_SCHEMA.features);
    expect(result.errors).toEqual({});
    expect(buildOrderedRecord(validValues, MODEL_SCHEMA.features)).toEqual(result.record);
    expect(Object.keys(result.record ?? {})).toEqual(MODEL_SCHEMA.features.map(({ name }) => name));
    expect(Object.hasOwn(result.record ?? {}, "Label")).toBe(false);

    const incomplete = { ...validValues };
    delete incomplete[MODEL_SCHEMA.features[0].name];
    expect(validateFeatureValues(incomplete, MODEL_SCHEMA.features).errors[MODEL_SCHEMA.features[0].name]).toBe("Required");
  });

  it("permits negative values only for fields where they occur in the source", () => {
    const negativeFeature = MODEL_SCHEMA.features.find((feature) => feature.allowsNegative)!;
    const nonnegativeFeature = MODEL_SCHEMA.features.find((feature) => !feature.allowsNegative)!;
    const values = Object.fromEntries(MODEL_SCHEMA.features.map(({ name }) => [name, "0"]));
    values[negativeFeature.name] = "-1";
    expect(validateFeatureValues(values, MODEL_SCHEMA.features).errors[negativeFeature.name]).toBeUndefined();
    values[nonnegativeFeature.name] = "-1";
    expect(validateFeatureValues(values, MODEL_SCHEMA.features).errors[nonnegativeFeature.name]).toMatch(/nonnegative/);
  });

  it("rejects float32 overflow and never substitutes zero", () => {
    const feature = MODEL_SCHEMA.features[0];
    const incomplete = Object.fromEntries(MODEL_SCHEMA.features.map(({ name }) => [name, "0"]));
    incomplete[feature.name] = "";
    expect(validateFeatureValues(incomplete, MODEL_SCHEMA.features).errors[feature.name]).toBe("Required");
    incomplete[feature.name] = "1e300";
    expect(validateFeatureValues(incomplete, MODEL_SCHEMA.features).errors[feature.name]).toMatch(/float32/);
  });
});
