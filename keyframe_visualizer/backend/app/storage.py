from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .schemas import JobRecord


TERMINAL_STATUSES = {"succeeded", "failed"}


class JobStore:
    def __init__(self, database_path: Path):
        self.database_path = database_path
        self._lock = threading.RLock()
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path, timeout=30)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS jobs (
                    id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    status TEXT NOT NULL,
                    stage TEXT NOT NULL,
                    progress REAL NOT NULL,
                    algorithm TEXT NOT NULL,
                    query TEXT NOT NULL,
                    config_json TEXT NOT NULL,
                    video_path TEXT NOT NULL,
                    original_filename TEXT NOT NULL,
                    output_dir TEXT NOT NULL,
                    manifest_path TEXT,
                    error TEXT
                )
                """
            )

    def create(
        self,
        *,
        job_id: str,
        algorithm: str,
        query: str,
        config: dict[str, Any],
        video_path: Path,
        original_filename: str,
        output_dir: Path,
    ) -> None:
        now = datetime.now(timezone.utc).isoformat()
        with self._lock, self._connect() as connection:
            connection.execute(
                """INSERT INTO jobs
                (id, created_at, updated_at, status, stage, progress, algorithm,
                 query, config_json, video_path, original_filename, output_dir)
                VALUES (?, ?, ?, 'queued', '等待执行', 0, ?, ?, ?, ?, ?, ?)""",
                (
                    job_id,
                    now,
                    now,
                    algorithm,
                    query,
                    json.dumps(config, ensure_ascii=False),
                    str(video_path),
                    original_filename,
                    str(output_dir),
                ),
            )

    def update(self, job_id: str, **values: Any) -> None:
        allowed = {"status", "stage", "progress", "manifest_path", "error"}
        fields = {key: value for key, value in values.items() if key in allowed}
        if not fields:
            return
        fields["updated_at"] = datetime.now(timezone.utc).isoformat()
        assignments = ", ".join(f"{key} = ?" for key in fields)
        with self._lock, self._connect() as connection:
            connection.execute(
                f"UPDATE jobs SET {assignments} WHERE id = ?",
                (*fields.values(), job_id),
            )

    def get_raw(self, job_id: str) -> dict[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
        return dict(row) if row else None

    def list_raw(self, limit: int = 100) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM jobs ORDER BY created_at DESC LIMIT ?", (limit,)
            ).fetchall()
        return [dict(row) for row in rows]

    def pending_ids(self) -> list[str]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT id FROM jobs WHERE status IN ('queued', 'running') ORDER BY created_at"
            ).fetchall()
        return [str(row["id"]) for row in rows]

    def delete(self, job_id: str) -> bool:
        with self._lock, self._connect() as connection:
            cursor = connection.execute("DELETE FROM jobs WHERE id = ?", (job_id,))
        return cursor.rowcount > 0

    @staticmethod
    def to_record(row: dict[str, Any]) -> JobRecord:
        try:
            config = json.loads(row.get("config_json") or "{}")
            parameters = config.get("parameters", {})
            if not isinstance(parameters, dict):
                parameters = {}
        except (TypeError, json.JSONDecodeError):
            parameters = {}
        return JobRecord(
            id=row["id"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            status=row["status"],
            stage=row["stage"],
            progress=row["progress"],
            algorithm=row["algorithm"],
            query=row["query"],
            original_filename=row["original_filename"],
            parameters=parameters,
            error=row.get("error"),
            manifest_available=bool(row.get("manifest_path")),
        )
