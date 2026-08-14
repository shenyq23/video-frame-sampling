from __future__ import annotations

import unittest

from pydantic import ValidationError

from app.schemas import AKSParameters, CreateJobConfig, CreateSessionConfig, SAGEParameters


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

    def test_sage_uses_its_own_parameter_model(self) -> None:
        job = CreateJobConfig.model_validate(
            {"algorithm": "sage", "query": "举起奖杯", "parameters": {"asr_mode": "none"}}
        )
        session = CreateSessionConfig.model_validate(
            {"algorithm": "sage", "parameters": {"asr_mode": "upload", "device": "cpu"}}
        )
        self.assertIsInstance(job.parameters, SAGEParameters)
        self.assertEqual(job.parameters.budget, 8)
        self.assertIsInstance(session.parameters, SAGEParameters)
        self.assertEqual(session.parameters.asr_mode, "upload")

    def test_sage_rejects_invalid_mode_and_budget(self) -> None:
        with self.assertRaises(ValidationError):
            SAGEParameters(asr_mode="ocr")  # type: ignore[arg-type]
        with self.assertRaises(ValidationError):
            SAGEParameters(budget=0)
