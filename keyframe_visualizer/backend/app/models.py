from __future__ import annotations

import json
import shutil
import tarfile
import uuid
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO


MAX_ARCHIVE_FILES = 50_000
MAX_EXTRACTED_BYTES = 30 * 1024 * 1024 * 1024


class ModelArchiveError(ValueError):
    pass


class ClipModelStore:
    def __init__(self, root: Path):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def list(self) -> list[dict[str, Any]]:
        models: list[dict[str, Any]] = []
        for metadata_path in self.root.glob("*/metadata.json"):
            try:
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            model_dir = metadata_path.parent
            if self._valid_model_root(model_dir, raise_error=False):
                models.append(metadata)
        return sorted(models, key=lambda item: item.get("created_at", ""), reverse=True)

    def resolve(self, model_id: str) -> Path:
        if not model_id or not model_id.replace("-", "").isalnum():
            raise ValueError("Invalid CLIP model id")
        model_dir = (self.root / model_id).resolve()
        if self.root.resolve() not in model_dir.parents:
            raise ValueError("Invalid CLIP model id")
        self._valid_model_root(model_dir)
        return model_dir

    def install_archive(
        self, archive_path: Path, *, display_name: str, original_filename: str
    ) -> dict[str, Any]:
        model_id = uuid.uuid4().hex
        extraction_root = self.root / f".extract-{model_id}"
        final_root = self.root / model_id
        extraction_root.mkdir(parents=True)
        try:
            if zipfile.is_zipfile(archive_path):
                self._extract_zip(archive_path, extraction_root)
            elif tarfile.is_tarfile(archive_path):
                self._extract_tar(archive_path, extraction_root)
            else:
                raise ModelArchiveError("只支持 ZIP、TAR、TAR.GZ 或 TGZ 模型压缩包")

            model_root = self._find_model_root(extraction_root)
            shutil.move(str(model_root), str(final_root))
            total_bytes = sum(path.stat().st_size for path in final_root.rglob("*") if path.is_file())
            from datetime import datetime, timezone

            metadata = {
                "id": model_id,
                "name": display_name.strip() or Path(original_filename).stem,
                "source_filename": original_filename,
                "size_bytes": total_bytes,
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
            (final_root / "metadata.json").write_text(
                json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            return metadata
        except Exception:
            if final_root.exists():
                shutil.rmtree(final_root)
            raise
        finally:
            if extraction_root.exists():
                shutil.rmtree(extraction_root)
            archive_path.unlink(missing_ok=True)

    @staticmethod
    def _safe_destination(root: Path, archive_name: str) -> Path:
        normalized = archive_name.replace("\\", "/")
        relative = PurePosixPath(normalized)
        if relative.is_absolute() or ".." in relative.parts:
            raise ModelArchiveError(f"压缩包包含不安全路径：{archive_name}")
        destination = (root / Path(*relative.parts)).resolve()
        if root.resolve() not in destination.parents and destination != root.resolve():
            raise ModelArchiveError(f"压缩包包含不安全路径：{archive_name}")
        return destination

    @staticmethod
    def _copy_limited(source: BinaryIO, destination: Path, expected_size: int) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        written = 0
        with destination.open("wb") as output:
            while chunk := source.read(1024 * 1024):
                written += len(chunk)
                if written > expected_size + 1024:
                    raise ModelArchiveError("压缩包条目的实际大小异常")
                output.write(chunk)

    def _extract_zip(self, archive_path: Path, destination: Path) -> None:
        with zipfile.ZipFile(archive_path) as archive:
            entries = archive.infolist()
            if len(entries) > MAX_ARCHIVE_FILES:
                raise ModelArchiveError("模型压缩包文件数量过多")
            if sum(entry.file_size for entry in entries) > MAX_EXTRACTED_BYTES:
                raise ModelArchiveError("模型解压后超过 30 GB 限制")
            for entry in entries:
                target = self._safe_destination(destination, entry.filename)
                unix_type = (entry.external_attr >> 16) & 0o170000
                if unix_type == 0o120000:
                    raise ModelArchiveError("模型压缩包不能包含符号链接")
                if entry.is_dir():
                    target.mkdir(parents=True, exist_ok=True)
                    continue
                with archive.open(entry) as source:
                    self._copy_limited(source, target, entry.file_size)

    def _extract_tar(self, archive_path: Path, destination: Path) -> None:
        with tarfile.open(archive_path) as archive:
            entries = archive.getmembers()
            if len(entries) > MAX_ARCHIVE_FILES:
                raise ModelArchiveError("模型压缩包文件数量过多")
            if sum(entry.size for entry in entries if entry.isfile()) > MAX_EXTRACTED_BYTES:
                raise ModelArchiveError("模型解压后超过 30 GB 限制")
            for entry in entries:
                target = self._safe_destination(destination, entry.name)
                if entry.issym() or entry.islnk():
                    raise ModelArchiveError("模型压缩包不能包含链接")
                if entry.isdir():
                    target.mkdir(parents=True, exist_ok=True)
                    continue
                if not entry.isfile():
                    raise ModelArchiveError("模型压缩包包含不支持的特殊文件")
                source = archive.extractfile(entry)
                if source is None:
                    raise ModelArchiveError(f"无法读取模型文件：{entry.name}")
                with source:
                    self._copy_limited(source, target, entry.size)

    def _find_model_root(self, extraction_root: Path) -> Path:
        candidates = []
        for config_path in extraction_root.rglob("config.json"):
            model_root = config_path.parent
            if self._valid_model_root(model_root, raise_error=False):
                candidates.append(model_root)
        if not candidates:
            raise ModelArchiveError(
                "压缩包中未找到完整 CLIP 模型；需要 config.json、preprocessor_config.json 和模型权重"
            )
        shallowest = min(len(path.relative_to(extraction_root).parts) for path in candidates)
        candidates = [
            path for path in candidates if len(path.relative_to(extraction_root).parts) == shallowest
        ]
        if len(candidates) != 1:
            raise ModelArchiveError("压缩包中检测到多个 CLIP 模型，请每次只上传一个模型")
        return candidates[0]

    @staticmethod
    def _valid_model_root(model_root: Path, raise_error: bool = True) -> bool:
        config_path = model_root / "config.json"
        processor_path = model_root / "preprocessor_config.json"
        weights = list(model_root.glob("*.safetensors")) + list(model_root.glob("*.bin"))
        has_tokenizer = (model_root / "tokenizer.json").is_file() or (
            (model_root / "vocab.json").is_file() and (model_root / "merges.txt").is_file()
        )
        valid = (
            config_path.is_file()
            and processor_path.is_file()
            and bool(weights)
            and has_tokenizer
        )
        if valid:
            try:
                config = json.loads(config_path.read_text(encoding="utf-8"))
                valid = config.get("model_type") == "clip" or any(
                    "CLIP" in str(name) for name in config.get("architectures", [])
                )
            except (OSError, json.JSONDecodeError):
                valid = False
        if not valid and raise_error:
            raise ModelArchiveError(
                "CLIP 模型目录缺少配置、权重或 tokenizer 文件，或 config.json 不是 CLIP 配置"
            )
        return valid
