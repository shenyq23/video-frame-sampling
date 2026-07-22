from __future__ import annotations

import unittest

from pydantic import ValidationError

from app.schemas import AKSParameters, CreateJobConfig


class SchemaTests(unittest.TestCase):
    def test_default_aks_configuration_is_ready_for_clip(self) -> None:
        config = CreateJobConfig(query="a red suitcase is opened")
        self.assertEqual(config.algorithm, "aks")
        self.assertEqual(config.parameters.aks_mode, "robust")
        self.assertEqual(config.parameters.feature_backend, "clip")
        self.assertEqual(config.parameters.max_num_frames, 32)

    def test_query_is_trimmed(self) -> None:
        config = CreateJobConfig(query="  target event  ")
        self.assertEqual(config.query, "target event")

    def test_remote_backend_requires_profile(self) -> None:
        with self.assertRaisesRegex(ValidationError, "feature profile"):
            AKSParameters(feature_backend="pangu")

    def test_invalid_frame_budget_is_rejected(self) -> None:
        with self.assertRaises(ValidationError):
            AKSParameters(max_num_frames=0)

    def test_candidate_interval_accepts_arbitrary_positive_finite_numbers(self) -> None:
        self.assertEqual(AKSParameters(sample_interval=0.000123).sample_interval, 0.000123)
        self.assertEqual(AKSParameters(sample_interval=7200.25).sample_interval, 7200.25)
        for invalid in (0, -0.1, float("inf"), float("nan")):
            with self.subTest(invalid=invalid), self.assertRaises(ValidationError):
                AKSParameters(sample_interval=invalid)
