import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from week3_preprocessing import (
    KNOWN_LABELS,
    PreprocessingConfig,
    _publish_no_replace,
    clean_chunk,
    normalize_headers,
    run,
    validate_schema,
)


class HeaderValidationTests(unittest.TestCase):
    def test_whitespace_is_normalized_and_duplicates_are_named(self):
        names, duplicates = normalize_headers([" A ", "A", " Label "])
        self.assertEqual(names, ["A", "A__duplicate_2", "Label"])
        self.assertEqual(duplicates, ["A"])

    def test_missing_label_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "missing.csv"
            path.write_text("feature\n1\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Label"):
                validate_schema([path], expected_features=None)

    def test_inconsistent_file_schemas_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            first, second = Path(directory) / "a.csv", Path(directory) / "b.csv"
            first.write_text("feature,Label\n1,BENIGN\n", encoding="utf-8")
            second.write_text("other,Label\n1,BENIGN\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Inconsistent schema"):
                validate_schema([first, second], expected_features=None)

    def test_missing_expected_feature_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "missing-feature.csv"
            path.write_text("feature,Label\n1,BENIGN\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Unexpected feature schema"):
                validate_schema([path], expected_features=("feature", "required"))

    def test_empty_input_list_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "No CSV"):
            validate_schema([])

    def test_empty_csv_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "empty.csv"
            path.touch()
            with self.assertRaisesRegex(ValueError, "empty"):
                validate_schema([path])


class ChunkCleaningTests(unittest.TestCase):
    def test_missing_values_and_infinity_rows_are_counted_and_removed(self):
        frame = pd.DataFrame({"feature": [1, np.nan, np.inf, 4], "Label": ["BENIGN"] * 4})
        cleaned, report = clean_chunk(frame, ["feature"], KNOWN_LABELS)
        self.assertEqual(cleaned["feature"].tolist(), [1.0, 4.0])
        self.assertEqual(report["missing_values_by_column"], {"feature": 1})
        self.assertEqual(report["infinite_values_by_column"], {"feature": 1})
        self.assertEqual(report["rows_removed_for_invalid_or_missing"], 2)

    def test_unexpected_label_is_rejected(self):
        frame = pd.DataFrame({"feature": [1], "Label": ["UNKNOWN ATTACK"]})
        with self.assertRaisesRegex(ValueError, "Unexpected target labels"):
            clean_chunk(frame, ["feature"], KNOWN_LABELS)

    def test_non_numeric_feature_is_counted(self):
        frame = pd.DataFrame({"feature": ["bad", "2"], "Label": ["BENIGN", "BENIGN"]})
        cleaned, report = clean_chunk(frame, ["feature"], KNOWN_LABELS)
        self.assertEqual(cleaned["feature"].tolist(), [2])
        self.assertEqual(report["non_numeric_values_by_column"], {"feature": 1})

    def test_duplicate_columns_only_collapse_when_equal(self):
        same = pd.DataFrame({"feature": [1], "feature__duplicate_2": [1], "Label": ["BENIGN"]})
        cleaned, report = clean_chunk(same, ["feature"], KNOWN_LABELS)
        self.assertNotIn("feature__duplicate_2", cleaned)
        self.assertEqual(report["duplicate_columns_collapsed"], ["feature"])
        different = pd.DataFrame({"feature": [1], "feature__duplicate_2": [2], "Label": ["BENIGN"]})
        with self.assertRaisesRegex(ValueError, "different values"):
            clean_chunk(different, ["feature"], KNOWN_LABELS)


class OutputAndConfigTests(unittest.TestCase):
    def test_output_publication_never_replaces_existing_file(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, destination = root / "temp", root / "result.csv"
            source.write_text("new", encoding="utf-8")
            destination.write_text("keep", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                _publish_no_replace(source, destination)
            self.assertEqual(destination.read_text(encoding="utf-8"), "keep")

    def test_preprocessing_configuration_is_reproducible(self):
        self.assertEqual(PreprocessingConfig(), PreprocessingConfig())
        self.assertEqual(PreprocessingConfig().random_seed, 42)
        self.assertEqual(PreprocessingConfig().deduplication_scope, "within_chunk_only")

    def test_small_run_writes_a_new_csv_and_quality_report(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_dir, output_dir = root / "input", root / "output"
            source_dir.mkdir()
            source = source_dir / "sample.csv"
            source.write_text("feature,Label\n1,BENIGN\n1,BENIGN\n2,BENIGN\n", encoding="utf-8")
            config = PreprocessingConfig(input_dir=str(source_dir), output_dir=str(output_dir),
                                         chunk_size=10, expected_labels=("BENIGN",),
                                         expected_feature_columns=("feature",))
            output, report_path, report = run(config)
            self.assertTrue(output.exists())
            self.assertTrue(report_path.exists())
            self.assertEqual(report["final_row_count"], 2)
            self.assertEqual(report["duplicates_removed_within_chunks"], 1)
            self.assertEqual(json.loads(report_path.read_text(encoding="utf-8"))["final_row_count"], 2)


if __name__ == "__main__":
    unittest.main()
