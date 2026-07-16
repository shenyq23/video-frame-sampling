# Video Frame Sampling Demo

一个与 `AKS/` 平级、以命令行为主要入口的视频抽帧框架：

```text
华为实习/
├── AKS/                    # 独立 AKS 算法模块
└── frame_sampling_demo/    # 视频解码、特征缓存、算法调度和导出
```

输入是一个视频和零个、一个或多个 query，输出为按时间顺序保存的一组 JPG
关键帧和可追溯 `manifest.json`。框架不调用 MLLM。

## 支持的算法

| 算法 | Query | 输出数量 | 说明 |
|---|---|---|---|
| `uniform` | 不需要 | 固定 | 在完整视频上均匀抽帧 |
| `clip_topk` | 需要 | 固定 | 选择 CLIP 相关性最高的 K 帧 |
| `aks_original` | 需要 | 原始配额 | 复用同级 AKS 的 original 模式 |
| `aks_robust` | 需要 | 固定 | 复用同级 AKS，并补足帧预算 |
| `clip_threshold` | 需要 | 自适应 | 保留超过相关性阈值的帧 |

## 安装

建议 Python 3.9～3.12：

```bash
cd /Users/apple/Doc/华为实习/frame_sampling_demo

python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-cli.txt
```

这会以 editable 模式安装：

- `../AKS` 中的 `aks_core.py`；
- Demo CLI；
- CLIP 所需的 Torch 和 Transformers。

视频解码优先使用 Decord；如果当前环境没有 Decord，但安装了 OpenCV，框架会
自动使用 OpenCV 兼容后端。

第一次运行 query-aware 算法时，会下载默认的
`openai/clip-vit-base-patch32`。也可以使用 `--model-name` 指向本地 CLIP。
该默认模型更适合英文 query；中文场景建议使用英文检索描述或兼容的多语 CLIP。

## 查看算法

安装后：

```bash
frame-sampling --list-algorithms
```

或者直接从源码运行：

```bash
python run.py --list-algorithms
```

## 1. 均匀抽帧

不需要 query，也不会加载 CLIP：

```bash
python run.py \
  --video /path/to/video.mp4 \
  --algorithm uniform \
  --max-frames 16
```

## 2. CLIP Top-K

```bash
python run.py \
  --video /path/to/video.mp4 \
  --algorithm clip_topk \
  --query "the person opens the red suitcase" \
  --max-frames 32 \
  --sample-interval 1.0
```

## 3. AKS Robust（推荐应用模式）

```bash
python run.py \
  --video /path/to/video.mp4 \
  --algorithm aks_robust \
  --query "the person opens the red suitcase" \
  --max-frames 32 \
  --candidate-sampling interval \
  --sample-interval 1.0 \
  --threshold 0.8 \
  --max-depth 5
```

`aks_robust` 调用：

```python
from aks_core import select_frames
```

AKS 算法代码仍位于 `../AKS`，Demo 里只有适配器。

## 4. AKS Original（论文仓库规则）

```bash
python run.py \
  --video /path/to/video.mp4 \
  --algorithm aks_original \
  --query "the target event" \
  --max-frames 64 \
  --candidate-sampling original \
  --threshold 0.8 \
  --std-threshold -100 \
  --max-depth 5
```

`candidate-sampling original` 使用原仓库的 `int(FPS)` 步长；
`aks_original` 使用 `int(N / 2**depth)` 配额，不补足非整除预算。

## 5. 自适应数量抽帧

不指定 `--max-frames` 时，输出所有分数超过阈值的帧：

```bash
python run.py \
  --video /path/to/video.mp4 \
  --algorithm clip_threshold \
  --query "a warning dialog appears" \
  --score-threshold 0.28 \
  --min-frames 4 \
  --sample-interval 0.5
```

可选地增加上限：

```bash
--max-frames 48
```

## 多 Query

重复传入 `--query`：

```bash
python run.py \
  --video ./demo.mp4 \
  --algorithm aks_robust \
  --query "When does the red car appear?" \
  --query "Who opens the door?" \
  --max-frames 32 \
  --multi-query-mode independent
```

三种模式：

- `independent`：每个 query 单独输出一组帧；
- `union`：各 query 先独立选择，再合并、去重并应用全局帧数上限；
- `joint`：把多个 query 拼成一个检索文本，只输出一组帧。

从文件读取多个 query：

```bash
python run.py \
  --video ./demo.mp4 \
  --algorithm clip_topk \
  --query-file ./queries.txt \
  --multi-query-mode independent \
  --max-frames 16
```

文本文件每行一个 query，也支持 JSON 字符串数组：

```json
[
  "When does the red car appear?",
  "Who opens the door?"
]
```

## 输出目录

默认输出到：

```text
outputs/<video>_<algorithm>_<run-id>/
├── manifest.json
├── query_001/
│   ├── manifest.json
│   └── frames/
│       ├── 001_t000012.000_f360.jpg
│       └── ...
└── query_002/
    ├── manifest.json
    └── frames/
```

`union`、`joint` 和 `uniform` 分别使用 `union/`、`joint/` 和 `global/`
分组目录。

指定输出根目录：

```bash
--output-dir /path/to/outputs
```

Manifest 记录：

- 视频路径、FPS、时长和总帧数；
- 全部 query 和多 query 模式；
- 抽帧算法与参数；
- 候选帧数量；
- 每个关键帧的原帧号、时间戳和 query 分数；
- AKS 的递归区间、深度和配额。

## CLIP 图像特征缓存

Query-aware 算法会把候选帧图像 embedding 缓存到：

```text
cache/<video-sha256>/clip/<model-hash>/<candidate-hash>.npz
```

同一视频、模型和候选帧配置下：

- 换 query 不需要重新编码视频；
- 多 query 共用一份图像 embedding；
- `clip_topk`、AKS 和 `clip_threshold` 可以共用缓存。

自定义缓存目录：

```bash
--cache-dir /path/to/cache
```

## Python API

```python
from pathlib import Path

from frame_sampling_demo import run_sampling
from frame_sampling_demo.schemas import SamplingRequest

result = run_sampling(
    SamplingRequest(
        video_path=Path("demo.mp4").resolve(),
        queries=["the person opens the door"],
        algorithm="aks_robust",
        max_frames=32,
    )
)

print(result["manifest_path"])
```

## 增加新的抽帧算法

在 `src/frame_sampling_demo/samplers/` 新建实现：

```python
from frame_sampling_demo.registry import register_sampler
from frame_sampling_demo.samplers.base import FrameSampler
from frame_sampling_demo.schemas import Selection


@register_sampler("my_sampler")
class MySampler(FrameSampler):
    query_aware = True
    adaptive_count = True

    def select(self, context, request, query, scores):
        selected_indices = ...
        return Selection(
            algorithm=self.name,
            selected_indices=selected_indices,
            trace={"my_parameter": "value"},
        )
```

然后在 `samplers/__init__.py` 导入这个类，它就会自动出现在：

```bash
frame-sampling --list-algorithms
```

## 测试

```bash
python -m unittest discover -s tests -v
```

测试不下载 CLIP 模型，通过模拟视频和特征服务验证：

- 算法注册；
- 固定与自适应数量选择；
- AKS 共享核心适配；
- 多 query union；
- JPG 和 manifest 导出。

## 可选 UI

原有 `app.py` Gradio 原型仍保留，但不是当前主链路。需要时安装：

```bash
python -m pip install -e ".[ui,clip]"
python app.py
```
