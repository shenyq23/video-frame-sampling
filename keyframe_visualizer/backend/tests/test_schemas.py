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

