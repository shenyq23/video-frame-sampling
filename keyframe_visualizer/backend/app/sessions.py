from __future__ import annotations

import asyncio
import json
import traceback
from pathlib import Path
from typing import Optional

from .algorithms.registry import AlgorithmRegistry
from .storage import SessionStore


class VideoSessionManager:
    def __init__(self, store: SessionStore, registry: AlgorithmRegistry):
        self.store = store
        self.registry = registry
        self.queue: Optional[asyncio.Queue] = None
        self.worker: Optional[asyncio.Task] = None

    async def start(self) -> None:
        self.queue = asyncio.Queue()
        for session_id in self.store.pending_ids():
            self.store.update(
                session_id, status="queued", stage="服务重启后重新排队", progress=0
            )
            await self.queue.put(session_id)
        self.worker = asyncio.create_task(self._worker_loop())

    async def stop(self) -> None:
        if self.worker:
            self.worker.cancel()
            try:
                await self.worker
            except asyncio.CancelledError:
                pass
        self.worker = None
        self.queue = None

    async def enqueue(self, session_id: str) -> None:
        if self.queue is None:
            raise RuntimeError("Session manager is not running")
        await self.queue.put(session_id)

    async def _worker_loop(self) -> None:
        if self.queue is None:
            raise RuntimeError("Session manager is not running")
        while True:
            session_id = await self.queue.get()
            try:
                await asyncio.to_thread(self._execute, session_id)
            finally:
                self.queue.task_done()

    def _execute(self, session_id: str) -> None:
        row = self.store.get_raw(session_id)
        if not row:
            return
        self.store.update(
            session_id, status="running", stage="准备视频预处理", progress=0.01, error=None
        )
        try:
            config = json.loads(row["config_json"])
            adapter = self.registry.get(row["algorithm"])
            prepare_session = getattr(adapter, "prepare_session", None)
            if prepare_session is None:
                raise ValueError(f"{row['algorithm']} does not support video sessions")

            def progress(stage: str, value: float) -> None:
                self.store.update(session_id, stage=stage, progress=max(0, min(1, value)))

            metadata_path = prepare_session(
                session_id=session_id,
                video_path=Path(row["video_path"]),
                original_filename=row["original_filename"],
                parameters=config["parameters"],
                session_dir=Path(row["session_dir"]),
                progress=progress,
            )
            metadata = json.loads(Path(metadata_path).read_text(encoding="utf-8"))
            candidate_count = int(
                metadata.get("candidate_sampling", {}).get("candidate_count") or 0
            )
            self.store.update(
                session_id,
                status="succeeded",
                stage="视频已准备好",
                progress=1.0,
                metadata_path=str(metadata_path),
                candidate_count=candidate_count,
            )
        except Exception as error:
            traceback.print_exc()
            self.store.update(
                session_id,
                status="failed",
                stage="视频预处理失败",
                error=str(error),
            )
