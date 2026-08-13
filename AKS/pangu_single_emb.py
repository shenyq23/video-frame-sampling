import os
import json
import numpy as np
import requests
import mimetypes
from typing import List
from pathlib import Path
from tqdm import tqdm

# ==================== 配置信息 ====================
# Embedding API配置
EMBED_BASE_URL = os.getenv("PANGU_EMBED_BASE_URL", "")
EMBED_API_KEY = os.getenv("PANGU_EMBED_API_KEY", "")

# 默认指令
QUERY_INSTR = "Retrieve relevant images for the user's query."
PASSAGE_INSTR = "Represent this image for text-to-image retrieval."

# 数据路径配置
IMAGE_ROOT = r"C:\Users\j30079095\Downloads\NIGHTS"
TEXT_FILE = r"C:\Users\j30079095\Desktop\.3 服务器\qry_texts.txt"

# 输出路径
OUTPUT_DIR = "./embeddings_output"


# ==================== Embedding API客户端 ====================
class EmbeddingAPIClient:
    """Embedding API客户端"""

    def __init__(self, base_url: str = EMBED_BASE_URL, api_key: str = EMBED_API_KEY):
        if not base_url:
            raise ValueError("Set PANGU_EMBED_BASE_URL or pass base_url")
        if not api_key:
            raise ValueError("Set PANGU_EMBED_API_KEY or pass api_key")
        self.base_url = base_url.rstrip('/')
        self.headers = {"Authorization": f"Bearer {api_key}"}
        self.embedding_dim = None  # 将在第一次调用后确定

    def embed_texts(self, texts: List[str], instruction: str = QUERY_INSTR,
                    batch_size: int = 32) -> np.ndarray:
        """
        批量文本嵌入

        Args:
            texts: 文本列表
            instruction: 指令
            batch_size: 批处理大小

        Returns:
            numpy array of shape (len(texts), embedding_dim)
        """
        all_embeddings = []

        for i in tqdm(range(0, len(texts), batch_size), desc="Embedding texts"):
            batch_texts = texts[i:i + batch_size]
            batch_embeddings = []

            for text in batch_texts:
                embedding = self._embed_single_text(text, instruction)
                batch_embeddings.append(embedding)

            all_embeddings.extend(batch_embeddings)

        return np.array(all_embeddings)

    def embed_images(self, image_paths: List[str], instruction: str = PASSAGE_INSTR,
                     batch_size: int = 16) -> np.ndarray:
        """
        批量图像嵌入

        Args:
            image_paths: 图像路径列表
            instruction: 指令
            batch_size: 批处理大小

        Returns:
            numpy array of shape (len(image_paths), embedding_dim)
        """
        all_embeddings = []

        for i in tqdm(range(0, len(image_paths), batch_size), desc="Embedding images"):
            batch_paths = image_paths[i:i + batch_size]
            batch_embeddings = []

            for img_path in batch_paths:
                embedding = self._embed_single_image(img_path, instruction)
                batch_embeddings.append(embedding)

            all_embeddings.extend(batch_embeddings)

        return np.array(all_embeddings)

    def _embed_single_text(self, text: str, instruction: str) -> List[float]:
        """单个文本嵌入"""
        url = f"{self.base_url}/embed"
        files = {
            "instruction": (None, instruction),
            "text": (None, text)
        }

        try:
            response = requests.post(url, headers=self.headers, files=files, timeout=30)
            response.raise_for_status()
            result = response.json()
            embedding = self._validate_embedding(result.get("embedding"), "text")

            if self.embedding_dim is None:
                self.embedding_dim = len(embedding)

            return embedding
        except Exception as e:
            raise RuntimeError(f"Error embedding text '{text[:50]}...': {e}") from e

    def _embed_single_image(self, image_path: str, instruction: str) -> List[float]:
        """单个图像嵌入"""
        url = f"{self.base_url}/embed"

        # 检测MIME类型
        mime_type, _ = mimetypes.guess_type(image_path)
        if not mime_type:
            mime_type = "image/png"

        try:
            with open(image_path, "rb") as f:
                files = {
                    "instruction": (None, instruction),
                    "image": (image_path, f, mime_type)
                }
                response = requests.post(url, headers=self.headers, files=files, timeout=30)
                response.raise_for_status()
                result = response.json()
                embedding = self._validate_embedding(result.get("embedding"), image_path)

                if self.embedding_dim is None:
                    self.embedding_dim = len(embedding)

                return embedding
        except Exception as e:
            raise RuntimeError(f"Error embedding image '{image_path}': {e}") from e

    def _validate_embedding(self, embedding, source: str) -> List[float]:
        if not isinstance(embedding, list) or not embedding:
            raise ValueError(f"Empty or invalid embedding returned for {source}")
        if not all(isinstance(value, (int, float)) for value in embedding):
            raise ValueError(f"Non-numeric embedding returned for {source}")
        if self.embedding_dim is not None and len(embedding) != self.embedding_dim:
            raise ValueError(
                f"Embedding dimension changed for {source}: "
                f"{len(embedding)} != {self.embedding_dim}"
            )
        return [float(value) for value in embedding]


# ==================== 保存函数 ====================
def save_embeddings(embeddings: np.ndarray, names: List[str], save_name: str):
    """
    保存嵌入向量和对应的名称

    Args:
        embeddings: numpy array, shape (n, dim)
        names: 名称列表（文件名或文本标识）
        save_name: 保存的文件名前缀
    """
    # 创建输出目录
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 保存嵌入向量为 .npy 文件
    emb_path = os.path.join(OUTPUT_DIR, f"{save_name}_embeddings.npy")
    np.save(emb_path, embeddings)
    print(f"✅ 嵌入向量已保存: {emb_path} (shape: {embeddings.shape})")

    # 保存名称列表为 .json 文件
    names_path = os.path.join(OUTPUT_DIR, f"{save_name}_names.json")
    with open(names_path, 'w', encoding='utf-8') as f:
        json.dump(names, f, ensure_ascii=False, indent=2)
    print(f"✅ 名称列表已保存: {names_path} (共 {len(names)} 个)")

    # 可选：保存为 .npz 文件（包含嵌入和名称）
    npz_path = os.path.join(OUTPUT_DIR, f"{save_name}.npz")
    np.savez(npz_path, embeddings=embeddings, names=names)
    print(f"✅ 组合文件已保存: {npz_path}")

    return emb_path, names_path, npz_path


# ==================== 主函数 ====================
def main():
    """批量提取图片和文本的嵌入向量并保存"""

    # 1. 初始化Embedding API客户端
    print("🚀 初始化Embedding API客户端...")
    embed_client = EmbeddingAPIClient()

    # ==================== 图片嵌入 ====================
    print("\n🖼️ 提取图像特征...")

    # 获取所有图像路径
    image_paths = []
    for root, dirs, files in os.walk(IMAGE_ROOT):
        for f in files:
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", ".webp")):
                image_paths.append(os.path.join(root, f))

    print(f"找到 {len(image_paths)} 张图像")

    if image_paths:
        # 提取图像嵌入
        image_embeddings = embed_client.embed_images(image_paths)

        # 获取图像名称列表（不含路径）
        image_names = [os.path.basename(path) for path in image_paths]

        # 保存图像嵌入
        save_embeddings(image_embeddings, image_names, "images")
    else:
        print("⚠️ 未找到图像文件，跳过图像嵌入")

    # ==================== 文本嵌入 ====================
    print("\n📝 提取文本特征...")

    texts = []
    text_names = []

    if TEXT_FILE and os.path.exists(TEXT_FILE):
        with open(TEXT_FILE, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    texts.append(line)
                    text_names.append(f"text_{len(texts)}")
        print(f"从文件加载了 {len(texts)} 条文本")


    if texts:
        text_embeddings = embed_client.embed_texts(texts)

        save_embeddings(text_embeddings, text_names, "texts")
    else:
        print("⚠️ 未提供文本数据，跳过文本嵌入（如需使用，请设置 TEXT_FILE 或添加示例文本）")

    print("\n🎉 所有嵌入提取和保存完成！")
    print(f"📁 输出目录: {os.path.abspath(OUTPUT_DIR)}")


if __name__ == '__main__':
    main()
