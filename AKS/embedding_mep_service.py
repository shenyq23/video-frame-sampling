import json
import logging
import os
from typing import Optional, Tuple, List
from mep_client import AsyncMepClient


logger = logging.getLogger(__name__)


def timing_decorator():
    """No-op fallback that keeps this uploaded example independently usable."""
    def decorate(function):
        return function
    return decorate



DEFAULT_THRESHOLD_VALUE = 1.0


class EmbeddingMepService:
    def __init__(self, mep_config=None):
        if mep_config is None:
            secret_key = os.getenv("MEP_EMBED_SECRET_KEY", "")
            if not secret_key:
                raise ValueError("Set MEP_EMBED_SECRET_KEY or pass mep_config")
            mep_config = {
                "appid": os.getenv("MEP_EMBED_APPID", "shopping_feature_gxq"),
                "secret_key": secret_key,
                "b_id": os.getenv("MEP_EMBED_B_ID", "cloud_gallery_embedding_service_gz"),
                "flow_id": os.getenv("MEP_EMBED_FLOW_ID", "cloud_gallery_embedding_service_gz"),
                "elb": os.getenv("MEP_EMBED_ELB", "http://10.41.1.17:8080/service"),
            }

        self.client = AsyncMepClient(**mep_config)

    @timing_decorator()
    async def get_clip_embedding(
        self,
        text: str,
        model_version: str,
        entity_type: str,
        top_k: int = 5,
    ) -> Tuple[Optional[List[float]], Optional[float]]:
        try:
            data = {
                "task": "text_embedding",
                "text": text,
                "model_version": model_version,
            }
            ret = await self.client.execute(data)
            if ret and int(ret.get("code", -1)) == 200:
                vec = ret.get("es_embedding", "")
                # TODO: 应该为1.0
                thre = ret.get("threshold", DEFAULT_THRESHOLD_VALUE)
                if isinstance(vec, str):
                    return json.loads(vec), thre
                return vec, thre
            else:
                logger.error(f"获取CLIP特征失败: {ret}|text={text}")

        except Exception as e:
            logger.error(f"获取CLIP特征失败: {e}|text={text}")
            return None, None

        return None, None

    @timing_decorator()
    async def get_face_embedding(
        self, text: Optional[str] = None, top_k: int = 5
    ) -> Tuple[Optional[List[float]], Optional[List[float]]]:
        try:
            data = {"task": "text_search_face_embedding", "text": text, "top_k": top_k}
            ret = await self.client.execute(data)
            if ret and int(ret.get("code", -1)) == 200:
                vec = ret.get("es_embedding", "")
                thre = ret.get("threshold", DEFAULT_THRESHOLD_VALUE)
                if isinstance(vec, str):
                    return json.loads(vec), thre
                return vec, thre
            else:
                logger.error(f"获取人脸特征失败: {ret}|text={text}")

        except Exception as e:
            logger.error(f"获取人脸特征失败: {e}|text={text}")
            return None, None

        return None, None

    @timing_decorator()
    async def get_text_embedding(
        self, text: str, model_version: str
    ) -> Tuple[Optional[List[float]], Optional[float]]:
        try:
            data = {"task": "text_embedding", "text": text, "model_version": model_version}    
            ret = await self.client.execute(data)
            if ret and int(ret.get("code", -1)) == 200:
                vec = ret.get("text_embedding", "")
                thre = ret.get("threshold", DEFAULT_THRESHOLD_VALUE)
                if isinstance(vec, str):
                    return json.loads(vec), thre
                return vec, thre
            else:
                logger.error(f"获取文本向量特征失败: {ret}|text={text}")

        except Exception as e:
            logger.error(f"获取文本向量特征失败: {e}|text={text}")
            return None, None

        return None, None

    async def close(self):
        await self.client.close()
