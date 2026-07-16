"""Gradio interface for uniform and AKS frame extraction."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import gradio as gr

from sampling import extract_frames


DEMO_DIR = Path(__file__).resolve().parent


def run_demo(
    video_path: Optional[str],
    method: str,
    frame_count: int,
    query: str,
    sample_interval: float,
    aks_mode: str,
    device: str,
    progress=gr.Progress(),
):
    try:
        result = extract_frames(
            video=video_path or "",
            method=method,
            frame_count=int(frame_count),
            query=query,
            sample_interval=float(sample_interval),
            aks_mode=aks_mode,
            device=device,
            output_root=DEMO_DIR / "outputs",
            progress=lambda ratio, message: progress(ratio, desc=message),
        )
    except Exception as error:
        raise gr.Error(str(error)) from error
    return (
        result["summary"],
        result["gallery"],
        result["manifest"],
        result["manifest_path"],
        result["archive_path"],
    )


def toggle_aks_options(method: str):
    visible = method == "AKS 关键帧"
    return (
        gr.update(visible=visible),
        gr.update(visible=visible),
        gr.update(visible=visible),
        gr.update(visible=visible),
    )


CSS = """
.gradio-container { max-width: 1180px !important; }
.hero { padding: 10px 4px 4px; }
.hero h1 { margin-bottom: 4px; }
.hint { color: #667085; }
"""


with gr.Blocks(title="视频抽帧 Demo", css=CSS) as demo:
    gr.Markdown(
        """
        <div class="hero">
          <h1>视频抽帧 Demo</h1>
          <p class="hint">均匀抽帧可快速预览；AKS 根据文本问题选择更相关、且兼顾时间覆盖的关键帧。</p>
        </div>
        """
    )
    with gr.Row():
        with gr.Column(scale=5):
            video = gr.Video(label="上传视频", sources=["upload"], type="filepath")
        with gr.Column(scale=4):
            method = gr.Radio(
                ["均匀抽帧", "AKS 关键帧"], value="均匀抽帧", label="抽帧方式"
            )
            frame_count = gr.Slider(
                minimum=1, maximum=64, value=16, step=1, label="输出帧数"
            )
            query = gr.Textbox(
                label="AKS 检索问题 / 画面描述",
                placeholder="例如：the person opens the red suitcase",
                lines=3,
                visible=False,
            )
            sample_interval = gr.Slider(
                minimum=0.25,
                maximum=5.0,
                value=1.0,
                step=0.25,
                label="AKS 候选帧间隔（秒）",
                visible=False,
            )
            aks_mode = gr.Radio(
                ["robust", "original"],
                value="robust",
                label="AKS 配额模式",
                visible=False,
            )
            device = gr.Dropdown(
                ["auto", "cuda", "mps", "cpu"],
                value="auto",
                label="计算设备",
                visible=False,
            )
            run_button = gr.Button("开始抽帧", variant="primary")

    status = gr.Textbox(label="运行结果", interactive=False)
    gallery = gr.Gallery(
        label="关键帧",
        columns=4,
        rows=2,
        height="auto",
        object_fit="contain",
        preview=True,
    )
    with gr.Accordion("运行清单", open=False):
        manifest = gr.JSON(label="Manifest")
        with gr.Row():
            manifest_file = gr.File(label="下载 manifest.json")
            archive_file = gr.File(label="下载完整 ZIP")

    method.change(
        toggle_aks_options,
        inputs=method,
        outputs=[query, sample_interval, aks_mode, device],
    )
    run_button.click(
        run_demo,
        inputs=[video, method, frame_count, query, sample_interval, aks_mode, device],
        outputs=[status, gallery, manifest, manifest_file, archive_file],
    )


if __name__ == "__main__":
    demo.queue(default_concurrency_limit=1).launch(
        server_name="127.0.0.1", server_port=7860, show_error=True
    )
