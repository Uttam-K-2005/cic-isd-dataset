import schemaData from "../data/model-schema.json";
import type { ModelSchema } from "../types";

export const MODEL_SCHEMA = schemaData as ModelSchema;

const configuredBatchLimit = Number(import.meta.env.VITE_MAX_BATCH_SIZE);
export const MAX_BATCH_SIZE = Number.isInteger(configuredBatchLimit)
  && configuredBatchLimit >= 1 && configuredBatchLimit <= 256
  ? configuredBatchLimit
  : MODEL_SCHEMA.batchLimitDefault;
