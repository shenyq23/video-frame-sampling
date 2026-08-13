from __future__ import annotations

import json
import tempfile
import unittest
import zipfile
from pathlib import Path

from app.models import ClipModelStore, ModelArchiveError


class ClipModelStoreTests(unittest.TestCase):
    @staticmethod
    def _write_valid_archive(path: Path) -> None:
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr("clip/config.json", json.dumps({"model_type": "clip"}))
            archive.writestr("clip/preprocessor_config.json", "{}")
            archive.writestr("clip/model.safetensors", b"test weights")
            archive.writestr("clip/tokenizer.json", "{}")

    def test_installs_lists_and_resolves_clip_archive(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / "model.zip"
            self._write_valid_archive(archive)
            store = ClipModelStore(root / "models")

            metadata = store.install_archive(
                archive, display_name="Offline CLIP", original_filename="model.zip"
            )
            self.assertEqual(metadata["name"], "Offline CLIP")
            self.assertFalse(archive.exists())
            resolved = store.resolve(metadata["id"])
            self.assertTrue((resolved / "config.json").is_file())
            self.assertEqual(store.list()[0]["id"], metadata["id"])

    def test_rejects_archive_path_traversal(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / "unsafe.zip"
            with zipfile.ZipFile(archive, "w") as output:
                output.writestr("../outside.txt", "unsafe")
            store = ClipModelStore(root / "models")
            with self.assertRaisesRegex(ModelArchiveError, "不安全路径"):
                store.install_archive(
                    archive, display_name="Unsafe", original_filename="unsafe.zip"
                )
            self.assertFalse((root / "outside.txt").exists())

    def test_rejects_incomplete_model(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / "incomplete.zip"
            with zipfile.ZipFile(archive, "w") as output:
                output.writestr("config.json", json.dumps({"model_type": "clip"}))
            store = ClipModelStore(root / "models")
            with self.assertRaisesRegex(ModelArchiveError, "未找到完整 CLIP"):
                store.install_archive(
                    archive, display_name="Incomplete", original_filename="incomplete.zip"
                )
