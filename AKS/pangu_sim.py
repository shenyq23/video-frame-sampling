#!/usr/bin/env python3
"""
MME /embed API 的 Python 封装，支持：
- 文本嵌入
- 图像嵌入
- 图文联合嵌入
- 任意两种嵌入之间的余弦相似度计算
"""

import os
import requests
import numpy as np
from typing import Optional, Union
from pathlib import Path

# ============ 配置 ============
BASE_URL = os.getenv("PANGU_EMBED_BASE_URL", "")
API_KEY = os.getenv("PANGU_EMBED_API_KEY", "")
QUERY_INSTR = "Retrieve relevant documents for the user's query."
PASSAGE_INSTR = "Represent this document chunk for semantic search and retrieval."


def get_embedding(
        text: Optional[str] = None,
        image_path: Optional[Union[str, Path]] = None,
        instruction: str = QUERY_INSTR,
) -> np.ndarray:
    """
    获取文本/图像/图文联合的嵌入向量。

    Args:
        text: 文本内容（可选）
        image_path: 图像文件路径（可选）
        instruction: 嵌入指令

    Returns:
        归一化的嵌入向量 (numpy array)
    """
    if text is None and image_path is None:
        raise ValueError("至少需要提供 text 或 image_path 之一")
    if not BASE_URL:
        raise ValueError("请设置环境变量 PANGU_EMBED_BASE_URL")
    if not API_KEY:
        raise ValueError("请设置环境变量 PANGU_EMBED_API_KEY")

    url = f"{BASE_URL}/embed"
    headers = {"Authorization": f"Bearer {API_KEY}"}

    # 构建 multipart 请求
    files = {}
    data = {"instruction": instruction}

    if text is not None:
        data["text"] = text

    if image_path is not None:
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"图像文件不存在: {image_path}")
        files["image"] = (image_path.name, open(image_path, "rb"), "image/png")

    try:
        response = requests.post(url, headers=headers, data=data, files=files, timeout=30)
        response.raise_for_status()
        result = response.json()

        embedding = np.array(result["embedding"], dtype=np.float32)

        # 打印元信息
        print(f"  嵌入维度: {result.get('embedding_dim', len(embedding))}")
        print(f"  包含文本: {result.get('has_text', text is not None)}")
        print(f"  包含图像: {result.get('has_image', image_path is not None)}")
        print(f"  前5维预览: {embedding[:5].tolist()}")

        return embedding

    finally:
        # 确保文件句柄被关闭
        if "image" in files:
            files["image"][1].close()


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """计算两个向量的余弦相似度"""
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return float(np.dot(vec1, vec2) / (norm1 * norm2))


def compute_similarities(text_embed: np.ndarray, image_embed: np.ndarray,
                         text_image_embed: np.ndarray) -> dict:
    """计算三种嵌入之间的两两相似度"""
    return {
        "文本 vs 图像": cosine_similarity(text_embed, image_embed),
        "文本 vs 文本+图像": cosine_similarity(text_embed, text_image_embed),
        "图像 vs 文本+图像": cosine_similarity(image_embed, text_image_embed),
    }


def main():
    print("=" * 60)
    print("MME 嵌入 API 测试 & 相似度计算")
    print("=" * 60)

    # 示例 1: 文本嵌入
    print("\n▶ 1. 文本嵌入")
    text_embed = get_embedding(
        text="手机",
        instruction=QUERY_INSTR
    )

    # 示例 2: 图像嵌入（请替换为实际图像路径）
    print("\n▶ 2. 图像嵌入")
    image_path = r"image1.PNG"
    try:
        image_embed = get_embedding(
            image_path=image_path,
            instruction=QUERY_INSTR
        )
    except FileNotFoundError as e:
        print(f"  ⚠ 跳过图像嵌入: {e}")
        image_embed = None

    # 示例 3: 文本+图像联合嵌入
    print("\n▶ 3. 文本+图像联合嵌入")
    text_image_embed = get_embedding(
        text="香烟",
        image_path=image_path if image_embed is not None else None,
        instruction=QUERY_INSTR
    )

    # 计算相似度
    if image_embed is not None:
        print("\n" + "=" * 60)
        print("📊 余弦相似度矩阵")
        print("=" * 60)
        similarities = compute_similarities(text_embed, image_embed, text_image_embed)
        for pair, score in similarities.items():
            print(f"  {pair:20s}: {score:.4f}")

    # 额外：演示文档（passage）嵌入
    print("\n" + "=" * 60)
    print("📄 文档嵌入示例（使用 PASSAGE_INSTR）")
    print("=" * 60)
    doc_embed = get_embedding(
        text="Retrieval-Augmented Generation (RAG) combines retrieval systems "
             "with large language models to generate more accurate responses.",
        instruction=PASSAGE_INSTR
    )

    # 查询 vs 文档相似度
    print(f"\n  查询 \"what is RAG?\" vs 文档 余弦相似度: "
          f"{cosine_similarity(text_embed, doc_embed):.4f}")


if __name__ == "__main__":
    main()
