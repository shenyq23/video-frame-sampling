from __future__ import annotations

import base64
import hashlib
import hmac
import json
import time
import uuid
from typing import Any, Callable, Optional

import requests


class VlmRequestError(RuntimeError):
    """Raised when the configured VLM service cannot return a usable answer."""


class MepVlmClient:
    def __init__(self, config: dict[str, Any], appid: str, secret_key: str):
        self.elb = str(config["elb"])
        self.appid = appid
        self.secret_key = secret_key
        self.b_id = str(config["b_id"])
        self.flow_id = str(config["flow_id"])
        self.stream = bool(config.get("stream", True))
        self.connect_timeout = float(config.get("connect_timeout_seconds", 15))
        self.read_timeout = float(config.get("read_timeout_seconds", 240))
        self.verify_tls = bool(config.get("verify_tls", False))

    def execute(
        self,
        request_data: dict[str, Any],
        callback: Optional[Callable[[str], None]] = None,
    ) -> str:
        request_id = uuid.uuid4().hex[:8]
        method = "POST"
        headers = self._make_headers(method, request_id)
        payload = self._make_request_body(request_data, request_id)
        url = self._request_url(request_id)
        try:
            response = requests.post(
                url,
                headers=headers,
                json=payload,
                stream=self.stream,
                verify=self.verify_tls,
                timeout=(self.connect_timeout, self.read_timeout),
            )
            response.raise_for_status()
            if self.stream:
                return self._read_stream(response, callback)
            return self._read_response(response)
        except requests.RequestException as error:
            raise VlmRequestError(f"VLM 服务请求失败：{error}") from error

    def _request_url(self, request_id: str) -> str:
        if not self.stream:
            return self.elb
        base = self.elb[:-8] + "/predict" if self.elb.endswith("/service") else self.elb
        separator = "&" if "?" in base else "?"
        return (
            f"{base}{separator}bId={self.b_id}&flowId={self.flow_id}"
            f"&uuId={request_id}"
        )

    def _make_headers(self, method: str, request_id: str) -> dict[str, str]:
        url_path = "/predict" if self.stream else "/service"
        canonical_query = (
            f"bId={self.b_id}&flowId={self.flow_id}&uuId={request_id}"
            if self.stream
            else ""
        )
        timestamp = str(round(time.time() * 1000))
        string_to_sign = (
            f"{method}&{url_path}&{canonical_query}&&appid={self.appid}"
            f"&timestamp={timestamp}"
        )
        signature = base64.b64encode(
            hmac.new(
                self.secret_key.encode("utf-8"),
                string_to_sign.encode("utf-8"),
                digestmod=hashlib.sha256,
            ).digest()
        ).decode("utf-8")
        access_token = (
            f"CLOUDSOA-HMAC-SHA256 appid={self.appid}, timestamp={timestamp}, "
            f"signmode=easy, signature=\"{signature}\""
        )
        return {"Content-Type": "application/json", "Authorization": access_token}

    def _make_request_body(
        self, request_data: dict[str, Any], request_id: str
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {"version": "1.0", "data": request_data}
        if not self.stream:
            payload["meta"] = {
                "bId": self.b_id,
                "flowId": self.flow_id,
                "isPressureTest": False,
                "uuId": request_id,
            }
        return payload

    def _read_stream(
        self,
        response: requests.Response,
        callback: Optional[Callable[[str], None]],
    ) -> str:
        chunks: list[str] = []
        for raw_line in response.iter_lines():
            if not raw_line:
                continue
            line = raw_line.decode("utf-8").strip()
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            event_type = event.get("event")
            if event_type == "data":
                content = str(event.get("content", ""))
                if content:
                    chunks.append(content)
                    if callback:
                        callback(content)
            elif event_type == "error":
                raise VlmRequestError(f"VLM 流式服务返回错误：{event}")
            elif event_type == "finish":
                break
        answer = "".join(chunks).strip()
        if not answer:
            raise VlmRequestError("VLM 服务未返回有效回答")
        return answer

    def _read_response(self, response: requests.Response) -> str:
        try:
            body = response.json()
        except ValueError as error:
            text = response.text.strip()
            if text:
                return text
            raise VlmRequestError("VLM 服务返回了空响应") from error
        answer = self._extract_text(body.get("result", body))
        if not answer:
            raise VlmRequestError(f"无法从 VLM 响应中提取回答：{body}")
        return answer

    @classmethod
    def _extract_text(cls, value: Any) -> str:
        if isinstance(value, str):
            return value.strip()
        if isinstance(value, list):
            return "".join(filter(None, (cls._extract_text(item) for item in value))).strip()
        if isinstance(value, dict):
            for key in ("answer", "content", "text", "result", "choices"):
                if key in value:
                    answer = cls._extract_text(value[key])
                    if answer:
                        return answer
        return ""

