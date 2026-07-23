from __future__ import annotations

import asyncio
import json
import re
import shutil
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response, StreamingResponse
from pydantic import ValidationError

from .algorithms.registry import AlgorithmRegistry
from .jobs import JobManager
from .models import ClipModelStore, ModelArchiveError
from .schemas import CreateJobConfig, JobRecord
from .settings import (
    CLIP_MODELS_DIR,
    DATABASE_PATH,
    RUNS_DIR,
    UPLOAD_DIR,
    ensure_data_directories,
    feature_profile_status,
)
from .storage import JobStore, TERMINAL_STATUSES


ensure_data_directories()
store = JobStore(DATABASE_PATH)
registry = AlgorithmRegistry()
manager = JobManager(store, registry)
clip_models = ClipModelStore(CLIP_MODELS_DIR)


@asynccontextmanager
async def lifespan(_: FastAPI):
    await manager.start()
    yield
    await manager.stop()


app = FastAPI(title="Keyframe Visualizer API", version="0.1.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

ALLOWED_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v"}
MAX_UPLOAD_BYTES = 8 * 1024 * 1024 * 1024
MAX_MODEL_UPLOAD_BYTES = 12 * 1024 * 1024 * 1024


def _row_or_404(job_id: str) -> dict:
    row = store.get_raw(job_id)
    if not row:
        raise HTTPException(status_code=404, detail="Job not found")
    return row


def _safe_media_path(root: Path, relative_path: str) -> Path:
    candidate = (root / relative_path).resolve()
    if root.resolve() not in candidate.parents and candidate != root.resolve():
        raise HTTPException(status_code=400, detail="Invalid media path")
    if not candidate.is_file():
        raise HTTPException(status_code=404, detail="Media not found")
    return candidate


def _task_owned_path(raw_path: str, root: Path, label: str) -> Path:
    candidate = Path(raw_path).expanduser().resolve()
    resolved_root = root.resolve()
    if resolved_root not in candidate.parents:
        raise HTTPException(
            status_code=409,
            detail=f"Refusing to delete {label} outside the managed data directory",
        )
    return candidate


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/api/algorithms")
def algorithms() -> list[dict]:
    return registry.metadata()


@app.get("/api/settings/feature-profiles")
def feature_profiles_status() -> dict:
    return feature_profile_status()


@app.get("/api/models/clip")
def list_clip_models() -> list[dict]:
    return clip_models.list()


@app.post("/api/models/clip", status_code=201)
async def upload_clip_model(
    archive: UploadFile = File(...),
    name: str = Form(""),
) -> dict:
    original_filename = archive.filename or "clip-model"
    incoming = CLIP_MODELS_DIR / f".incoming-{uuid.uuid4().hex}"
    size = 0
    try:
        with incoming.open("wb") as destination:
            while chunk := await archive.read(1024 * 1024):
                size += len(chunk)
                if size > MAX_MODEL_UPLOAD_BYTES:
                    raise HTTPException(status_code=413, detail="CLIP 模型压缩包超过 12 GB 限制")
                destination.write(chunk)
        if size == 0:
            raise HTTPException(status_code=400, detail="模型压缩包为空")
        try:
            return clip_models.install_archive(
                incoming,
                display_name=name,
                original_filename=original_filename,
            )
        except ModelArchiveError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
    finally:
        incoming.unlink(missing_ok=True)


@app.get("/api/jobs", response_model=list[JobRecord])
def list_jobs() -> list[JobRecord]:
    return [store.to_record(row) for row in store.list_raw()]


@app.get("/api/jobs/{job_id}", response_model=JobRecord)
def get_job(job_id: str) -> JobRecord:
    return store.to_record(_row_or_404(job_id))


@app.delete("/api/jobs/{job_id}", status_code=204)
def delete_job(job_id: str) -> Response:
    row = _row_or_404(job_id)
    if row["status"] not in TERMINAL_STATUSES:
        raise HTTPException(
            status_code=409,
            detail="排队中或运行中的任务不能清除，请等待任务结束后重试",
        )

    video_path = _task_owned_path(row["video_path"], UPLOAD_DIR, "uploaded video")
    output_dir = _task_owned_path(row["output_dir"], RUNS_DIR, "run output")

    if video_path.exists():
        if not video_path.is_file():
            raise HTTPException(status_code=409, detail="任务上传路径不是普通文件")
        video_path.unlink()
    if output_dir.exists():
        if not output_dir.is_dir():
            raise HTTPException(status_code=409, detail="任务输出路径不是目录")
        shutil.rmtree(output_dir)

    store.delete(job_id)
    return Response(status_code=204)


@app.post("/api/jobs", response_model=JobRecord, status_code=202)
async def create_job(
    video: UploadFile = File(...),
    config: str = Form(...),
) -> JobRecord:
    try:
        parsed_config = CreateJobConfig.model_validate_json(config)
    except ValidationError as error:
        raise HTTPException(status_code=422, detail=json.loads(error.json())) from error

    suffix = Path(video.filename or "video").suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=415, detail=f"Unsupported video type: {suffix or 'unknown'}")

    job_id = uuid.uuid4().hex
    safe_name = re.sub(r"[^A-Za-z0-9._-]+", "_", Path(video.filename or "video").name)
    upload_path = UPLOAD_DIR / f"{job_id}_{safe_name}"
    size = 0
    with upload_path.open("wb") as destination:
        while chunk := await video.read(1024 * 1024):
            size += len(chunk)
            if size > MAX_UPLOAD_BYTES:
                destination.close()
                upload_path.unlink(missing_ok=True)
                raise HTTPException(status_code=413, detail="Video exceeds the 8 GB limit")
            destination.write(chunk)
    if size == 0:
        upload_path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail="Uploaded video is empty")

    output_dir = RUNS_DIR / job_id
    config_dict = parsed_config.model_dump()
    store.create(
        job_id=job_id,
        algorithm=parsed_config.algorithm,
        query=parsed_config.query,
        config=config_dict,
        video_path=upload_path,
        original_filename=video.filename or safe_name,
        output_dir=output_dir,
    )
    await manager.enqueue(job_id)
    return store.to_record(_row_or_404(job_id))


@app.get("/api/jobs/{job_id}/events")
async def job_events(job_id: str) -> StreamingResponse:
    _row_or_404(job_id)

    async def stream():
        previous = None
        while True:
            row = _row_or_404(job_id)
            record = store.to_record(row).model_dump(mode="json")
            payload = json.dumps(record, ensure_ascii=False)
            if payload != previous:
                yield f"data: {payload}\n\n"
                previous = payload
            if row["status"] in TERMINAL_STATUSES:
                break
            await asyncio.sleep(0.5)

    return StreamingResponse(stream(), media_type="text/event-stream")


@app.get("/api/jobs/{job_id}/manifest")
def get_manifest(job_id: str) -> dict:
    row = _row_or_404(job_id)
    if not row.get("manifest_path"):
        raise HTTPException(status_code=409, detail="Manifest is not available")
    path = Path(row["manifest_path"])
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Manifest file is missing")
    return json.loads(path.read_text(encoding="utf-8"))


@app.get("/api/jobs/{job_id}/manifest/download")
def download_manifest(job_id: str) -> FileResponse:
    row = _row_or_404(job_id)
    if not row.get("manifest_path"):
        raise HTTPException(status_code=409, detail="Manifest is not available")
    return FileResponse(row["manifest_path"], filename=f"{job_id}_manifest.json")


@app.get("/api/jobs/{job_id}/video")
def get_video(job_id: str) -> FileResponse:
    row = _row_or_404(job_id)
    return FileResponse(row["video_path"])


@app.get("/api/jobs/{job_id}/media/{relative_path:path}")
def get_media(job_id: str, relative_path: str) -> FileResponse:
    row = _row_or_404(job_id)
    return FileResponse(_safe_media_path(Path(row["output_dir"]), relative_path))
