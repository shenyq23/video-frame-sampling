from __future__ import annotations

import asyncio
import json
import re
import shutil
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response, StreamingResponse
from pydantic import ValidationError

from .algorithms.registry import AlgorithmRegistry
from .jobs import JobManager
from .models import ClipModelStore, ModelArchiveError
from .schemas import (
    CreateJobConfig,
    CreateSessionConfig,
    JobRecord,
    SessionRecord,
    VlmAnswerRequest,
    VlmAnswerResult,
)
from .settings import (
    CLIP_MODELS_DIR,
    DATABASE_PATH,
    RUNS_DIR,
    SESSIONS_DIR,
    UPLOAD_DIR,
    ensure_data_directories,
    feature_profile_status,
    vlm_profile_status,
)
from .storage import JobStore, SessionStore, TERMINAL_STATUSES
from .sessions import VideoSessionManager
from .vlm.client import VlmRequestError
from .vlm.service import VlmAnswerService


ensure_data_directories()
store = JobStore(DATABASE_PATH)
session_store = SessionStore(DATABASE_PATH)
registry = AlgorithmRegistry()
manager = JobManager(store, registry, session_store=session_store)
session_manager = VideoSessionManager(session_store, registry)
clip_models = ClipModelStore(CLIP_MODELS_DIR)
vlm_answers = VlmAnswerService()


@asynccontextmanager
async def lifespan(_: FastAPI):
    await session_manager.start()
    await manager.start()
    yield
    await manager.stop()
    await session_manager.stop()


app = FastAPI(title="Keyframe Visualizer API", version="0.1.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"http://(localhost|127\.0\.0\.1):\d+",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

ALLOWED_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v"}
MAX_UPLOAD_BYTES = 8 * 1024 * 1024 * 1024
MAX_MODEL_UPLOAD_BYTES = 12 * 1024 * 1024 * 1024
DELETE_RETRY_DELAYS_SECONDS = (0.15, 0.3, 0.6, 1.0, 1.5)
PREPROCESS_PARAMETER_KEYS = {
    "candidate_sampling",
    "sample_interval",
    "feature_backend",
    "feature_profile",
    "clip_model_id",
    "model_name",
    "device",
}


def _row_or_404(job_id: str) -> dict:
    row = store.get_raw(job_id)
    if not row:
        raise HTTPException(status_code=404, detail="Job not found")
    return row


def _session_or_404(session_id: str) -> dict:
    row = session_store.get_raw(session_id)
    if not row:
        raise HTTPException(status_code=404, detail="Video session not found")
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


def _delete_with_retries(path: Path, *, is_directory: bool, label: str) -> None:
    if not path.exists():
        return
    for attempt in range(len(DELETE_RETRY_DELAYS_SECONDS) + 1):
        try:
            if is_directory:
                shutil.rmtree(path)
            else:
                path.unlink()
            return
        except FileNotFoundError:
            return
        except PermissionError as error:
            if attempt >= len(DELETE_RETRY_DELAYS_SECONDS):
                windows_error = getattr(error, "winerror", None)
                reason = (
                    "文件仍被浏览器或其他程序占用"
                    if windows_error in {32, 33}
                    else "没有权限删除文件"
                )
                raise HTTPException(
                    status_code=423,
                    detail=(
                        f"无法清除{label}：{reason}。请关闭其他打开该视频的网页或程序后重试。"
                        f"路径：{path}"
                    ),
                ) from error
            time.sleep(DELETE_RETRY_DELAYS_SECONDS[attempt])
        except OSError as error:
            if attempt >= len(DELETE_RETRY_DELAYS_SECONDS):
                raise HTTPException(
                    status_code=409,
                    detail=f"无法清除{label}：{error}。路径：{path}",
                ) from error
            time.sleep(DELETE_RETRY_DELAYS_SECONDS[attempt])


def _validate_session_job_parameters(session: dict, parameters: dict) -> None:
    try:
        session_config = json.loads(session.get("config_json") or "{}")
        session_parameters = session_config.get("parameters", {})
        if not isinstance(session_parameters, dict):
            session_parameters = {}
    except (TypeError, json.JSONDecodeError):
        session_parameters = {}
    mismatched = [
        key
        for key in sorted(PREPROCESS_PARAMETER_KEYS)
        if session_parameters.get(key) != parameters.get(key)
    ]
    if mismatched:
        raise HTTPException(
            status_code=409,
            detail=(
                "这些参数会影响候选帧或图像特征缓存，不能在同一个视频会话内修改："
                + ", ".join(mismatched)
                + "。请重新准备视频会话。"
            ),
        )


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/api/algorithms")
def algorithms() -> list[dict]:
    return registry.metadata()


@app.get("/api/settings/feature-profiles")
def feature_profiles_status() -> dict:
    return feature_profile_status()


@app.get("/api/settings/vlm-profiles")
def vlm_profiles_status() -> dict:
    return vlm_profile_status()


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


@app.get("/api/sessions", response_model=list[SessionRecord])
def list_sessions() -> list[SessionRecord]:
    return [session_store.to_record(row) for row in session_store.list_raw()]


@app.get("/api/sessions/{session_id}", response_model=SessionRecord)
def get_session(session_id: str) -> SessionRecord:
    return session_store.to_record(_session_or_404(session_id))


@app.post("/api/sessions", response_model=SessionRecord, status_code=202)
async def create_session(
    video: UploadFile = File(...),
    config: str = Form(...),
) -> SessionRecord:
    try:
        parsed_config = CreateSessionConfig.model_validate_json(config)
    except ValidationError as error:
        raise HTTPException(status_code=422, detail=json.loads(error.json())) from error

    suffix = Path(video.filename or "video").suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=415, detail=f"Unsupported video type: {suffix or 'unknown'}")

    session_id = uuid.uuid4().hex
    session_dir = SESSIONS_DIR / session_id
    source_dir = session_dir / "source"
    cache_dir = session_dir / "preprocess"
    source_dir.mkdir(parents=True)
    safe_name = re.sub(r"[^A-Za-z0-9._-]+", "_", Path(video.filename or "video").name)
    upload_path = source_dir / safe_name
    size = 0
    try:
        with upload_path.open("wb") as destination:
            while chunk := await video.read(1024 * 1024):
                size += len(chunk)
                if size > MAX_UPLOAD_BYTES:
                    destination.close()
                    raise HTTPException(status_code=413, detail="Video exceeds the 8 GB limit")
                destination.write(chunk)
        if size == 0:
            raise HTTPException(status_code=400, detail="Uploaded video is empty")
    except Exception:
        if session_dir.exists():
            shutil.rmtree(session_dir, ignore_errors=True)
        raise

    config_dict = parsed_config.model_dump()
    session_store.create(
        session_id=session_id,
        algorithm=parsed_config.algorithm,
        config=config_dict,
        video_path=upload_path,
        original_filename=video.filename or safe_name,
        session_dir=session_dir,
        cache_dir=cache_dir,
    )
    await session_manager.enqueue(session_id)
    return session_store.to_record(_session_or_404(session_id))


@app.get("/api/sessions/{session_id}/events")
async def session_events(session_id: str) -> StreamingResponse:
    _session_or_404(session_id)

    async def stream():
        previous = None
        while True:
            row = _session_or_404(session_id)
            record = session_store.to_record(row).model_dump(mode="json")
            payload = json.dumps(record, ensure_ascii=False)
            if payload != previous:
                yield f"data: {payload}\n\n"
                previous = payload
            if row["status"] in TERMINAL_STATUSES:
                break
            await asyncio.sleep(0.5)

    return StreamingResponse(stream(), media_type="text/event-stream")


@app.delete("/api/sessions/{session_id}", status_code=204)
def delete_session(session_id: str) -> Response:
    row = _session_or_404(session_id)
    if row["status"] not in TERMINAL_STATUSES:
        raise HTTPException(
            status_code=409,
            detail="排队中或运行中的视频会话不能清除，请等待预处理结束后重试",
        )
    session_jobs = store.list_by_session_raw(session_id)
    non_terminal = [job for job in session_jobs if job["status"] not in TERMINAL_STATUSES]
    if non_terminal:
        raise HTTPException(
            status_code=409,
            detail="这个视频会话仍有排队中或运行中的 query，请等待任务结束后重试",
        )

    for job in session_jobs:
        output_dir = _task_owned_path(job["output_dir"], RUNS_DIR, "run output")
        if output_dir.exists():
            _delete_with_retries(output_dir, is_directory=True, label="任务输出")
    session_dir = _task_owned_path(row["session_dir"], SESSIONS_DIR, "video session")
    if session_dir.exists():
        _delete_with_retries(session_dir, is_directory=True, label="视频会话")
    store.delete_by_session(session_id)
    session_store.delete(session_id)
    return Response(status_code=204)


@app.post("/api/sessions/{session_id}/jobs", response_model=JobRecord, status_code=202)
async def create_session_job(session_id: str, parsed_config: CreateJobConfig) -> JobRecord:
    session = _session_or_404(session_id)
    if session["status"] != "succeeded" or not session.get("metadata_path"):
        raise HTTPException(status_code=409, detail="视频还没有预处理完成")
    config_dict = parsed_config.model_dump()
    _validate_session_job_parameters(session, config_dict["parameters"])

    job_id = uuid.uuid4().hex
    output_dir = RUNS_DIR / job_id
    store.create(
        job_id=job_id,
        algorithm=parsed_config.algorithm,
        query=parsed_config.query,
        config=config_dict,
        video_path=Path(session["video_path"]),
        original_filename=session["original_filename"],
        output_dir=output_dir,
        session_id=session_id,
        owns_video=False,
    )
    await manager.enqueue(job_id)
    return store.to_record(_row_or_404(job_id))


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

    output_dir = _task_owned_path(row["output_dir"], RUNS_DIR, "run output")

    if output_dir.exists():
        if not output_dir.is_dir():
            raise HTTPException(status_code=409, detail="任务输出路径不是目录")
        _delete_with_retries(output_dir, is_directory=True, label="任务输出")

    if row.get("owns_video"):
        video_path = _task_owned_path(row["video_path"], UPLOAD_DIR, "uploaded video")
        if video_path.exists():
            if not video_path.is_file():
                raise HTTPException(status_code=409, detail="任务上传路径不是普通文件")
            _delete_with_retries(video_path, is_directory=False, label="上传视频")

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


@app.get("/api/jobs/{job_id}/vlm-answer", response_model=VlmAnswerResult)
def get_vlm_answer(
    job_id: str,
    frame_set: str = Query(pattern="^(selected|uniform|candidates)$"),
) -> VlmAnswerResult:
    row = _row_or_404(job_id)
    result = vlm_answers.saved_answer(Path(row["output_dir"]), frame_set)
    if result is None:
        raise HTTPException(status_code=404, detail="这组帧还没有保存的 VLM 回答")
    return VlmAnswerResult.model_validate(result)


@app.post("/api/jobs/{job_id}/vlm-answer", response_model=VlmAnswerResult)
def create_vlm_answer(job_id: str, request: VlmAnswerRequest) -> VlmAnswerResult:
    row = _row_or_404(job_id)
    if row["status"] != "succeeded" or not row.get("manifest_path"):
        raise HTTPException(status_code=409, detail="只有抽帧成功的任务才能进行 VLM 问答")
    manifest_path = Path(row["manifest_path"])
    if not manifest_path.is_file():
        raise HTTPException(status_code=404, detail="Manifest file is missing")
    try:
        result = vlm_answers.answer(
            job_id=job_id,
            output_dir=Path(row["output_dir"]),
            manifest_path=manifest_path,
            frame_set=request.frame_set,
            query=request.query,
            profile_id=request.vlm_profile,
        )
    except VlmRequestError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    return VlmAnswerResult.model_validate(result)


@app.get("/api/jobs/{job_id}/video")
def get_video(job_id: str) -> FileResponse:
    row = _row_or_404(job_id)
    return FileResponse(row["video_path"])


@app.get("/api/jobs/{job_id}/media/{relative_path:path}")
def get_media(job_id: str, relative_path: str) -> FileResponse:
    row = _row_or_404(job_id)
    try:
        return FileResponse(_safe_media_path(Path(row["output_dir"]), relative_path))
    except HTTPException as error:
        if error.status_code != 404 or not row.get("session_id"):
            raise
        session = session_store.get_raw(str(row["session_id"]))
        if not session:
            raise
        return FileResponse(_safe_media_path(Path(session["session_dir"]), relative_path))
