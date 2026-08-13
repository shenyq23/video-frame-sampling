import time
from typing import List, Dict, Tuple, Any
from uuid import uuid4
import aiohttp
from aiohttp import ClientSession
import asyncio
import base64
import hmac
import hashlib
import uuid
import json
import logging



aiohttp_session: aiohttp.ClientSession = None
logger = logging.getLogger(__name__)


async def get_session() -> aiohttp.ClientSession:
    global aiohttp_session
    if aiohttp_session is None:
        logger.info(f'init aiohttp')
        aiohttp_session = aiohttp.ClientSession()
    return aiohttp_session


async def close_session():
    global aiohttp_session
    if aiohttp_session is not None:
        await aiohttp_session.close()
        aiohttp_session = None


class AsyncMepClient:
    def __init__(self, appid: str, secret_key: str, b_id: str, flow_id: str, elb: str):
        self.appid = appid
        self.secret_key = secret_key
        self.b_id = b_id
        self.flow_id = flow_id
        self.elb = elb
        self._session = None  # 延迟初始化会话

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None:
            self._session = await get_session()
        return self._session

    def __make_headers(self, method: str = "POST") -> Dict[str, str]:
        http_method = method
        url_path = "/service"
        canonical_query_string = ""
        http_payload = ""
        timestamp = str(round(time.time() * 1000))
        string_to_sign = (
            f"{http_method}&{url_path}&{canonical_query_string}&{http_payload}"
            f"&appid={self.appid}&timestamp={timestamp}"
        )
        string_to_sign_bytes = string_to_sign.encode("utf-8")
        secret_key_bytes = self.secret_key.encode("utf-8")
        signature = base64.b64encode(
            hmac.new(secret_key_bytes, string_to_sign_bytes,
                     digestmod=hashlib.sha256).digest()
        ).decode("utf-8")
        access_token = (
            f"CLOUDSOA-HMAC-SHA256 appid={self.appid}, timestamp={timestamp}, "
            f"signmode=easy, signature=\"{signature}\""
        )

        return {
            "Content-Type": "application/json",
            "Authorization": access_token
        }

    def __make_request_body(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "version": "1.0",
            "data": request_data,
            "meta": {
                "bId": self.b_id,
                "flowId": self.flow_id,
                "isPressureTest": False,
                "uuId": str(uuid.uuid4())
            }
        }

    def __parse_response(self, response: Dict[str, Any]) -> Dict[str, Any]:
        result = response.get("result", {})
        if not result:
            logger.error(f"\033[31mERROR: 请求失败，响应无result: {response}\033[0m")
            return {}
        if str(result.get("code", "")) != "0":
            logger.error(f"\033[31mERROR: 请求失败，code非0: {response}\033[0m")
            return {}
        return result.get("content", [{}])[0]

    async def execute(self, request_data, scene='', method: str = "POST") -> Dict[str, Any]:
        session = await self._get_session()
        headers = self.__make_headers(method)
        payload = self.__make_request_body(request_data)
        data_str = json.dumps(payload, ensure_ascii=False)
        try:
            async with session.post(
                url=self.elb,
                data=data_str.encode("utf-8"),
                headers=headers
            ) as resp:
                resp_text = await resp.text(encoding="utf-8")
                if resp.status != 200:
                    msg = f"mep vec[{scene}] server error={resp.status}, resp={resp_text}"
                    logger.error(msg)
                    raise Exception(msg)
                try:
                    response_json = json.loads(resp_text)
                    return self.__parse_response(response_json)
                except json.JSONDecodeError as e:
                    msg = f"mep vec[{scene}] parse response error={str(e)}"
                    logger.error(msg)
                    raise

        except Exception as e:
            import traceback
            logger.error(
                f"mep vec[{scene}] error={str(e)}, {traceback.format_exc()}")
            raise

    async def close(self):
        await close_session()
