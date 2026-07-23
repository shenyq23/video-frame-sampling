from __future__ import annotations

import asyncio
import json
import traceback
from pathlib import Path
from typing import Optional

from .algorithms.registry import AlgorithmRegistry
from .storage import JobStore
from .video_reader import release_video


class JobManager:
    def __init__(self, store: JobStore, registry: AlgorithmRegistry):
        self.store = store
        self.registry = registry
        self.queue: Optional[asyncio.Queue] = None
        self.worker: Optional[asyncio.Task] = None

    async def start(self) -> None:
        self.queue = asyncio.Queue()
        for job_id in self.store.pending_ids():
            self.store.update(job_id, status="queued", stage="服务重启后重新排队", progress=0)
            await self.queue.put(job_id)
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

    async def enqueue(self, job_id: str) -> None:
        if self.queue is None:
            raise RuntimeError("Job manager is not running")
        await self.queue.put(job_id)

    async def _worker_loop(self) -> None:
        if self.queue is None:
            raise RuntimeError("Job manager is not running")
        while True:
            job_id = await self.queue.get()
            try:
                await asyncio.to_thread(self._execute, job_id)
            finally:
                self.queue.task_done()

    def _execute(self, job_id: str) -> None:
        row = self.store.get_raw(job_id)
        if not row:
            return
        self.store.update(job_id, status="running", stage="准备任务", progress=0.01, error=None)
        video_path = Path(row["video_path"])
        try:
            config = json.loads(row["config_json"])
            adapter = self.registry.get(row["algorithm"])

            def progress(stage: str, value: float) -> None:
                self.store.update(job_id, stage=stage, progress=max(0, min(1, value)))

            manifest_path = adapter.run(
                job_id=job_id,
                video_path=video_path,
                original_filename=row["original_filename"],
                query=row["query"],
                parameters=config["parameters"],
                output_dir=Path(row["output_dir"]),
                progress=progress,
            )
            release_video(video_path)
            self.store.update(
                job_id,
                status="succeeded",
                stage="完成",
                progress=1.0,
                manifest_path=str(manifest_path),
            )
        except Exception as error:
            traceback.print_exc()
            release_video(video_path)
            self.store.update(
                job_id,
                status="failed",
                stage="运行失败",
                error=str(error),
            )
        finally:
            release_video(video_path)
