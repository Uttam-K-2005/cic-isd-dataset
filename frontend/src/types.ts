export interface FeatureDefinition {
  name: string;
  group: string;
  dtype: "number";
  allowsNegative: boolean;
  minimumObserved: number;
  negativeRowsObserved: number;
}

export interface ModelSchema {
  modelName: string;
  modelVersion: string;
  labelClasses: string[];
  inputDtype: "float32";
  featureCount: number;
  batchLimitDefault: number;
  featureDomainSource: string;
  profiledRows: number;
  features: FeatureDefinition[];
}

export interface HealthResponse {
  status: string;
}

export interface ReadinessResponse {
  status: string;
  model_loaded: boolean;
}

export interface ModelMetadataResponse {
  model_name: string;
  model_version: string | null;
  supported_classes: string[];
}

export type FlowRecord = Record<string, number>;

export interface PredictionResponse {
  request_id: string;
  predicted_class: string;
  is_attack: boolean;
  probabilities: Record<string, number>;
  model_name: string;
  model_version: string | null;
  predicted_at: string;
}

export interface BatchPredictionResponse {
  request_id: string;
  predictions: PredictionResponse[];
  model_name: string;
  model_version: string | null;
  predicted_at: string;
}
