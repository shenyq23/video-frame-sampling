from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from app.algorithms.vsi_adapter import VSIAdapter


class VSIAdapterAssetTests(unittest.TestCase):
    def test_usable_file_rejects_git_lfs_pointer(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.pt"
            path.write_text(
                "version https://git-lfs.github.com/spec/v1\n"
                "oid sha256:0000000000000000000000000000000000000000000000000000000000000000\n"
                "size 123456\n",
                encoding="utf-8",
            )
            self.assertFalse(VSIAdapter._usable_file(path, minimum_size=1))

    def test_default_yolo_resolves_to_bundled_absolute_path(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / "yolov8s-worldv2.pt"
            model.write_bytes(b"model data")
            with (
                patch("app.algorithms.vsi_adapter.VSI_ROOT", root),
                patch("app.algorithms.vsi_adapter.VSI_BUNDLED_YOLO_MODEL", model),
                patch.object(VSIAdapter, "_usable_file", return_value=True),
            ):
                self.assertEqual(VSIAdapter._resolve_yolo_model(model.name), str(model))

    def test_default_text_model_resolves_to_bundled_snapshot(self) -> None:
        default_model = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
        bundled = Path("/tmp/vsi-text-model")
        with patch.object(VSIAdapter, "_bundled_text_model", return_value=bundled):
            self.assertEqual(VSIAdapter._resolve_text_model(default_model), str(bundled))


if __name__ == "__main__":
    unittest.main()
