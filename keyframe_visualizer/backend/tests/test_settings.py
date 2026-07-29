from __future__ import annotations

import json
import os
import unittest
from unittest.mock import patch

from app.settings import feature_profile_status, vlm_profile_status


class FeatureProfileStatusTests(unittest.TestCase):
    def test_reports_only_environment_variable_names(self) -> None:
        environment = {
            "PANGU_EMBED_API_KEY": "secret-value",
            "MEP_EMBED_APPID": "app-id",
        }
        with patch.dict(os.environ, environment, clear=True):
            status = feature_profile_status()
        self.assertTrue(status["pangu-default"]["credentials_ready"])
        self.assertFalse(status["mep-default"]["credentials_ready"])
        self.assertEqual(
            status["mep-default"]["missing_environment_variables"],
            ["MEP_EMBED_SECRET_KEY"],
        )
        self.assertNotIn("secret-value", json.dumps(status, ensure_ascii=False))

    def test_vlm_status_never_exposes_credentials(self) -> None:
        environment = {
            "MEP_VLM_APPID": "vlm-app-id",
            "MEP_VLM_SECRET_KEY": "vlm-secret-value",
        }
        with patch.dict(os.environ, environment, clear=True):
            status = vlm_profile_status()
        self.assertTrue(status["mep-vlm-default"]["credentials_ready"])
        serialized = json.dumps(status, ensure_ascii=False)
        self.assertNotIn("vlm-app-id", serialized)
        self.assertNotIn("vlm-secret-value", serialized)
