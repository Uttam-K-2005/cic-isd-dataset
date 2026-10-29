import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from ml.data import LabelCodec, separate_features_target
from ml.metrics import classification_metrics
from ml.model_io import align_features, load_bundle
from ml.model_io import save_bundle
from ml.models import build_models, predict_labels
from ml.splitting import exact_row_groups, grouped_stratified_split


class DataContractTests(unittest.TestCase):
    def test_target_is_separate_from_features(self):
        frame = pd.DataFrame({"feature": [1, 2], "Label": ["BENIGN", "Bot"]})
        features, labels = separate_features_target(frame)
        self.assertEqual(list(features.columns), ["feature"])
        self.assertNotIn("Label", features)
        self.assertEqual(labels.tolist(), ["BENIGN", "Bot"])

    def test_label_codec_round_trip_and_unknown_label(self):
        codec = LabelCodec(["BENIGN", "Bot"])
        encoded = codec.encode(["Bot", "BENIGN"])
        self.assertEqual(codec.decode(encoded), ["Bot", "BENIGN"])
        with self.assertRaisesRegex(ValueError, "Unknown labels"):
            codec.encode(["new label"])

    def test_grouped_split_is_reproducible_disjoint_and_keeps_duplicates_together(self):
        rows = []
        for label, offset in (("BENIGN", 0), ("Bot", 100), ("DDoS", 200)):
            for i in range(8):
                rows.append({"f1": offset + i, "f2": i % 3, "Label": label})
        rows.append(rows[0].copy())
        frame = pd.DataFrame(rows)
        x, y = separate_features_target(frame)
        split1 = grouped_stratified_split(x, y, seed=17)
        split2 = grouped_stratified_split(x, y, seed=17)
        for first, second in zip(split1[:3], split2[:3]):
            np.testing.assert_array_equal(first, second)
        group_ids = exact_row_groups(x, y)
        sets = [set(group_ids[idx]) for idx in split1[:3]]
        self.assertFalse(sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2])
        duplicate_indices = np.flatnonzero((x["f1"] == rows[0]["f1"]) & (y == "BENIGN"))
        membership = {next(i for i, part in enumerate(split1[:3]) if int(idx) in set(part)) for idx in duplicate_indices}
        self.assertEqual(len(membership), 1)


class InferenceContractTests(unittest.TestCase):
    def test_align_features_rejects_missing_and_unexpected_keys(self):
        with self.assertRaisesRegex(ValueError, "missing=.*f2"):
            align_features({"f1": 1}, ["f1", "f2"])
        with self.assertRaisesRegex(ValueError, "unexpected=.*f3"):
            align_features({"f1": 1, "f2": 2, "f3": 3}, ["f1", "f2"])
        aligned = align_features({"f2": 2, "f1": 1}, ["f1", "f2"])
        self.assertEqual(aligned.iloc[0].tolist(), [1, 2])
        with self.assertRaisesRegex(ValueError, "finite"):
            align_features({"f1": float("inf"), "f2": 2}, ["f1", "f2"])

    def test_missing_model_artifact_fails_clearly(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(FileNotFoundError, "Model artifact is missing"):
                load_bundle(directory)

    def test_prediction_shape_with_model_stub(self):
        class Stub:
            def predict(self, values):
                return np.asarray(["BENIGN"] * len(values))

        result = predict_labels(Stub(), "random_forest", np.zeros((4, 2)), ["BENIGN", "Bot"])
        self.assertEqual(result.shape, (4,))


class MetricTests(unittest.TestCase):
    def test_multiclass_metrics_and_false_positive_rate(self):
        result = classification_metrics(
            ["BENIGN", "BENIGN", "Bot", "Bot"],
            ["BENIGN", "Bot", "Bot", "Bot"], ["BENIGN", "Bot"],
        )
        self.assertEqual(result["accuracy"], 0.75)
        self.assertEqual(result["per_class"]["BENIGN"]["false_positive_rate_ovr"], 0.0)
        self.assertEqual(result["per_class"]["Bot"]["false_positive_rate_ovr"], 0.5)
        self.assertEqual(result["confusion_matrix"], [[1, 1], [0, 2]])


@unittest.skipUnless(importlib.util.find_spec("sklearn"), "scikit-learn is not installed")
class OptionalModelTests(unittest.TestCase):
    def test_preprocessing_is_inside_logistic_pipeline(self):
        models = build_models(requested=["logistic_regression"])
        logistic = models["logistic_regression"]
        self.assertEqual(logistic.steps[0][0], "standardscaler")
        x_train = np.array([[0.0], [1.0], [10.0], [11.0], [20.0], [21.0]])
        y_train = np.array(["A", "A", "B", "B", "C", "C"])
        logistic.fit(x_train, y_train)
        scaler = logistic.named_steps["standardscaler"]
        self.assertAlmostEqual(float(scaler.mean_[0]), float(x_train.mean()))
        self.assertEqual(logistic.predict(np.array([[5.0]])).shape, (1,))

    def test_model_artifact_save_and_load(self):
        from sklearn.dummy import DummyClassifier

        model = DummyClassifier(strategy="most_frequent").fit(np.array([[0.0], [1.0]]), ["BENIGN", "Bot"])
        with tempfile.TemporaryDirectory() as directory:
            artifact = save_bundle({"model": model, "model_name": "dummy", "feature_names": ["f"],
                                    "label_classes": ["BENIGN", "Bot"]}, directory, "dummy")
            loaded = load_bundle(artifact)
            self.assertEqual(loaded["feature_names"], ["f"])
            self.assertEqual(loaded["model"].predict(np.array([[5.0]])).shape, (1,))


if __name__ == "__main__":
    unittest.main()
