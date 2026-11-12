import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from src.data_utils import (
    align_source_translation,
    inspect_audio_directory,
    load_audio_segment,
    validate_rows,
)
from src.evaluation import error_rates, minimally_normalize, normalized_for_scoring


class ProjectUtilityTests(unittest.TestCase):
    def test_scoring_normalizers_preserve_dialect_words(self):
        text = "  إنتَ، تونس!  "
        self.assertEqual(minimally_normalize(text), "إنتَ، تونس!")
        self.assertEqual(normalized_for_scoring(text), "إنتَ تونس")

    def test_error_rates_are_corpus_level_and_zero_for_exact_match(self):
        self.assertEqual(error_rates(["أنا هنا"], ["أنا هنا"]), {"wer": 0.0, "cer": 0.0})

    def test_error_rates_measure_word_and_character_edits(self):
        self.assertEqual(error_rates(["one two"], ["one"]), {"wer": 0.5, "cer": 0.5})

    def test_empty_reference_does_not_make_silent_zero_error_rate(self):
        with self.assertRaises(ValueError):
            validate_rows(
                [
                    {
                        "audio_filename": "talk",
                        "start_time": "0",
                        "end_time": "1",
                        "translation": "",
                    }
                ],
                text_column="translation",
            )

    def test_scoring_rejects_all_empty_references(self):
        with self.assertRaises(ValueError):
            error_rates(["  "], [""])

    def test_translation_pair_requires_same_talk_count_and_interval_overlap(self):
        source = [
            {
                "audio_filename": "talk",
                "start_time": "0.0",
                "end_time": "2.0",
                "transcription": "مرحبا",
            }
        ]
        target = [
            {
                "audio_filename": "talk",
                "start_time": "0.5",
                "end_time": "2.5",
                "translation": "hello",
            }
        ]
        pairs, report = align_source_translation(source, target)
        self.assertEqual(len(pairs), 1)
        self.assertEqual(report["candidate_pairs"], 1)
        self.assertEqual(pairs[0]["translation"], "hello")

        target[0]["start_time"] = "3.0"
        target[0]["end_time"] = "4.0"
        pairs, report = align_source_translation(source, target)
        self.assertEqual(pairs, [])
        self.assertIn("talk", report["skipped_talks"])

    def test_translation_pair_rejects_tiny_interval_overlap(self):
        source = [
            {
                "audio_filename": "talk",
                "start_time": "0",
                "end_time": "2",
                "transcription": "مرحبا",
            }
        ]
        target = [
            {
                "audio_filename": "talk",
                "start_time": "1.9",
                "end_time": "3.9",
                "translation": "hello",
            }
        ]
        pairs, report = align_source_translation(source, target)
        self.assertEqual(pairs, [])
        self.assertIn("insufficient_interval_overlap", report["skipped_talks"]["talk"])

    def test_audio_inventory_names_missing_wavs_without_downloading(self):
        rows = {
            "train": [
                {
                    "audio_filename": "talk_one",
                    "start_time": "0",
                    "end_time": "1",
                    "transcription": "مرحبا",
                }
            ],
            "dev": [],
            "test": [],
        }
        with TemporaryDirectory() as temporary_directory:
            report = inspect_audio_directory(temporary_directory, rows)

        self.assertFalse(report["ready"])
        self.assertEqual(report["missing_files"], ["talk_one.wav"])
        self.assertEqual(report["splits"]["train"]["ready"], False)
        self.assertEqual(report["splits"]["train"]["missing_files"], ["talk_one.wav"])
        self.assertEqual(report["splits"]["dev"]["ready"], True)

    def test_audio_loader_rejects_path_like_talk_ids(self):
        with TemporaryDirectory() as temporary_directory:
            with self.assertRaises(ValueError):
                load_audio_segment(temporary_directory, "..\\outside", 0.0, 1.0)


if __name__ == "__main__":
    unittest.main()
