from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Callable

import requests


class SageAsrError(RuntimeError):
    """Raised when the remote SAGE ASR service cannot complete a job."""


class SageAsrClient:
    def __init__(self, config: dict[str, Any]):
        self.base_url = str(config.get("base_url", "")).rstrip("/")
        self.token = str(config.get("token", ""))
        self.connect_timeout = float(config.get("connect_timeout", 15))
        self.upload_timeout = float(config.get("upload_timeout", 600))
        self.download_timeout = float(config.get("download_timeout", 120))
        self.job_timeout = float(config.get("job_timeout", 2400))
        self.poll_interval = float(config.get("poll_interval", 5))
        self.delete_remote = bool(config.get("delete_remote", True))
        if not self.base_url or not self.token:
            raise SageAsrError("远程 ASR 未配置，请设置 SAGE_ASR_BASE_URL 和 SAGE_ASR_TOKEN")

    @property
    def headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.token}"}

    @staticmethod
    def _payload(response: requests.Response) -> dict[str, Any]:
        try:
            payload = response.json()
        except ValueError as error:
            raise SageAsrError(
                f"远程 ASR 返回非 JSON 响应：HTTP {response.status_code} {response.text[:300]}"
            ) from error
        if not response.ok:
            detail = payload.get("detail", payload)
            raise SageAsrError(f"远程 ASR 请求失败：HTTP {response.status_code} {detail}")
        if not isinstance(payload, dict):
            raise SageAsrError("远程 ASR 返回的数据不是对象")
        return payload

    @staticmethod
    def _valid_json_file(path: Path) -> bool:
        try:
            json.loads(path.read_text(encoding="utf-8"))
            return True
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            return False

    @staticmethod
    def _write_state(state_path: Path, payload: dict[str, Any]) -> None:
        state_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = state_path.with_suffix(state_path.suffix + ".part")
        temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        os.replace(temporary, state_path)

    def _request_job(self, video_path: Path) -> dict[str, Any]:
        try:
            with video_path.open("rb") as source:
                response = requests.post(
                    f"{self.base_url}/asr/jobs",
                    headers=self.headers,
                    files={"video": (video_path.name, source, "application/octet-stream")},
                    timeout=(self.connect_timeout, self.upload_timeout),
                )
            return self._payload(response)
        except requests.RequestException as error:
            raise SageAsrError(f"上传视频到远程 ASR 失败：{error}") from error

    def _read_job(self, job_id: str) -> dict[str, Any]:
        try:
            response = requests.get(
                f"{self.base_url}/asr/jobs/{job_id}",
                headers=self.headers,
                timeout=(self.connect_timeout, 30),
            )
            return self._payload(response)
        except requests.RequestException as error:
            raise SageAsrError(f"查询远程 ASR 任务失败：{error}") from error

    def _download(self, job_id: str, destination: Path) -> None:
        temporary = destination.with_suffix(destination.suffix + ".part")
        temporary.unlink(missing_ok=True)
        try:
            with requests.get(
                f"{self.base_url}/asr/jobs/{job_id}/result",
                headers=self.headers,
                stream=True,
                timeout=(self.connect_timeout, self.download_timeout),
            ) as response:
                response.raise_for_status()
                with temporary.open("wb") as output:
                    for chunk in response.iter_content(1024 * 1024):
                        if chunk:
                            output.write(chunk)
            json.loads(temporary.read_text(encoding="utf-8"))
            os.replace(temporary, destination)
        except (requests.RequestException, OSError, json.JSONDecodeError) as error:
            temporary.unlink(missing_ok=True)
            raise SageAsrError(f"下载远程 ASR 结果失败：{error}") from error

    def delete(self, job_id: str) -> None:
        try:
            requests.delete(
                f"{self.base_url}/asr/jobs/{job_id}",
                headers=self.headers,
                timeout=(self.connect_timeout, 30),
            ).raise_for_status()
        except requests.RequestException:
            # Local ASR is already durable; remote cleanup is best-effort.
            pass

    def generate(
        self,
        *,
        video_path: Path,
        destination: Path,
        state_path: Path,
        progress: Callable[[str, float], None],
    ) -> str:
        destination.parent.mkdir(parents=True, exist_ok=True)
        job_id: str | None = None
        state: dict[str, Any] = {}
        if state_path.is_file():
            try:
                state = json.loads(state_path.read_text(encoding="utf-8"))
                job_id = str(state["job_id"])
            except (KeyError, TypeError, ValueError, OSError, json.JSONDecodeError):
                job_id = None
                state = {}

        if job_id and state.get("downloaded") is True and self._valid_json_file(destination):
            progress("复用已下载的远程 ASR JSON", 0.88)
            return job_id

        if job_id:
            progress("恢复远程 ASR 任务", 0.12)
            try:
                job = self._read_job(job_id)
            except SageAsrError:
                job_id = None
            else:
                if job.get("status") == "failed":
                    raise SageAsrError(f"远程 ASR 失败：{job.get('error') or '未知错误'}")
        if not job_id:
            progress("上传视频到远程 ASR", 0.10)
            job = self._request_job(video_path)
            job_id = str(job.get("job_id", ""))
            if not job_id:
                raise SageAsrError("远程 ASR 未返回 job_id")
            self._write_state(
                state_path,
                {"job_id": job_id, "created_at": time.time(), "downloaded": False},
            )

        started = time.monotonic()
        while True:
            job = self._read_job(job_id)
            remote_status = str(job.get("status", "unknown"))
            remote_stage = str(job.get("stage", remote_status))
            elapsed = time.monotonic() - started
            progress(
                f"远程 ASR：{remote_stage}",
                min(0.86, 0.20 + 0.66 * elapsed / max(self.job_timeout, 1)),
            )
            if remote_status == "succeeded":
                break
            if remote_status == "failed":
                raise SageAsrError(f"远程 ASR 失败：{job.get('error') or '未知错误'}")
            if elapsed >= self.job_timeout:
                raise SageAsrError(f"远程 ASR 超过 {self.job_timeout:.0f} 秒仍未完成")
            time.sleep(self.poll_interval)

        progress("下载远程 ASR JSON", 0.88)
        self._download(job_id, destination)
        self._write_state(
            state_path,
            {
                "job_id": job_id,
                "created_at": state.get("created_at", time.time()),
                "downloaded": True,
                "downloaded_at": time.time(),
            },
        )
        return job_id
