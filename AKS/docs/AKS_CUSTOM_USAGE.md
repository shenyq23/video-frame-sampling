# AKS 共享核心与自定义视频使用指南

## 1. 三条可用链路

仓库现在保留三种入口：

| 入口 | 用途 | 是否使用共享核心 |
|---|---|---|
| `aks_keyframes.py` | 之前的单视频独立链路，原样保留 | 否 |
| `aks_keyframes_v2.py` | 新的单视频、单 query、可追溯链路 | 是，调用 `aks_core.py` |
| `frame_select.py` | 官方 `scores.json + frames.json` 批处理链路 | 是，调用 `aks_core.py` |

`aks_keyframes_v2.py` 只负责选帧和导出 JPG，不会调用 LLaVA、Qwen2-VL
或其他 MLLM。选出的帧可以继续作为多图输入交给任意下游模型。

## 2. 安装依赖

```bash
pip install -r requirements-keyframes.txt
```

默认相关性模型为 `openai/clip-vit-base-patch32`。第一次使用时，
Transformers 会从 Hugging Face 下载权重；也可以通过 `--model-name` 指向
本地兼容的 CLIP checkpoint。

### 2.1 特征提取后端

默认仍使用原 CLIP。本地离线模型路径的用法与之前相同：

```bash
python aks_keyframes_v2.py \
  --video /path/to/video.mp4 \
  --query "the event to retrieve" \
  --feature-backend clip \
  --model-name /path/to/local/clip-model
```

Pangu 后端按 `pangu_sim.py` 的协议调用 Bearer 鉴权 `/embed` 接口：
`instruction/text` 通过普通表单字段发送，图片通过 multipart 文件字段发送。
复制并修改 `configs/features/pangu.example.json`，密钥只放环境变量：

```bash
export PANGU_EMBED_API_KEY='your-key'
python aks_keyframes_v2.py \
  --video /path/to/video.mp4 \
  --query "the event to retrieve" \
  --feature-backend pangu \
  --feature-config configs/features/pangu.json
```

模板默认让文本和图片都使用 `pangu_sim.py` 的 `QUERY_INSTR`，并把视频帧编码为
PNG。`pangu_sim.py` 本身单独运行时还需设置：

```bash
export PANGU_EMBED_BASE_URL='your-base-url'
```

MEP 必须同时提供同一个多模态模型的文本和图片 embedding task。复制
`configs/features/mep.example.json`，按部署服务修改 task、字段和响应路径：

```bash
export MEP_EMBED_APPID='your-app-id'
export MEP_EMBED_SECRET_KEY='your-secret'
python aks_keyframes_v2.py \
  --video /path/to/video.mp4 \
  --query "the event to retrieve" \
  --feature-backend mep \
  --feature-config configs/features/mep.json
```

文本和图片可以分别配置响应字段。当前模板依次兼容文本响应中的
`text_embedding/es_embedding/embedding`，以及图片响应中的
`image_embedding/es_embedding/embedding`。如果错误信息列出了其他实际字段，
将对应字段加入 `text_response_embedding_paths` 或
`image_response_embedding_paths`。

其他 multipart embedding API 可从 `configs/features/http.example.json`
开始配置，并选择 `--feature-backend http`。无论后端类型，文本与图片向量必须
维度相同且位于同一语义空间。

不符合上述 HTTP 协议的本地模型或特殊 API，可从
`configs/features/python.example.json` 开始，指定自定义 Python 类。该类实现
`embed_texts/embed_images`，或直接实现 `prepare_query/score_images` 即可，AKS
主流程无需增加新的模型分支。

## 3. 推荐：自定义视频的 robust 模式

```bash
python aks_keyframes_v2.py \
  --video /absolute/path/to/video.mp4 \
  --query "the person opens the red suitcase" \
  --aks-mode robust \
  --max-num-frames 32 \
  --candidate-sampling interval \
  --sample-interval 1.0 \
  --output-dir ./aks_output_v2/my_video
```

这一模式使用：

- 每 `1.0` 秒一个候选帧；
- CLIP 计算 query 与候选帧的余弦相似度；
- 原论文的归一化、停止条件、递归二分和区间内 Top-K；
- robust 配额：在原二叉树权重基础上补足整数取整造成的缺额。

快速动作或短暂弹窗可将候选间隔改为 0.25～0.5 秒：

```bash
python aks_keyframes_v2.py \
  --video ./demo.mp4 \
  --query "a warning dialog appears" \
  --sample-interval 0.5 \
  --max-num-frames 32
```

也可以从 UTF-8 文本文件读取 query：

```bash
python aks_keyframes_v2.py \
  --video ./demo.mp4 \
  --query-file ./query.txt
```

`--query` 和 `--query-file` 只能使用一个。

## 4. 复现原仓库 CLIP + AKS 选择规则

使用以下组合：

```bash
python aks_keyframes_v2.py \
  --video /absolute/path/to/video.mp4 \
  --query "the target event" \
  --aks-mode original \
  --candidate-sampling original \
  --max-num-frames 64 \
  --threshold 0.8 \
  --std-threshold -100 \
  --max-depth 5 \
  --output-dir ./aks_output_v2/reproduction
```

两个 `original` 的含义不同：

- `--aks-mode original`：每个终止区间严格使用
  `int(N / 2**depth)` 配额，不补足取整缺额；
- `--candidate-sampling original`：严格使用原特征脚本的
  `int(FPS)` 帧步长，而不是按真实秒间隔四舍五入。

标准论文配置 `N=64, max_depth=5` 下，配额通常正好整除。若使用 16、31、
48 等预算，`original` 可能少于请求数量，这是原始整数配额的预期行为。

共享核心修复了两类非算法错误：JSON list 会先转换为 NumPy 数组；所有分数
相同时不会出现除零。其余分段与配额规则保持可对照。

## 5. 输出内容

默认输出目录为：

```text
aks_output_v2/<video_name>/
├── frames/
│   ├── 001_t000012.012_f360.jpg
│   └── ...
└── manifest.json
```

文件名包含：

- 时间顺序；
- 秒级时间戳；
- 原视频绝对帧号。

`manifest.json` 记录：

- 视频路径、FPS 和总帧数；
- query 与 CLIP checkpoint；
- 候选帧采样模式；
- AKS 的 `original/robust` 模式；
- `t1`、`t2`、最大深度；
- 每个终止时间区间的边界、深度和配额；
- 每个关键帧的帧号、时间戳和相关性分数。

因此可以根据 manifest 回溯“候选帧如何产生、区间如何分配预算、最终选中了
哪些帧”。

## 6. 官方 scores.json 批处理链路

特征提取方式仍与 README 相同。例如先生成 BLIP 分数：

```bash
python feature_extract.py \
  --dataset_name longvideobench \
  --dataset_path ./datasets/longvideobench \
  --extract_feature_model blip
```

然后按原配额选择：

```bash
python frame_select.py \
  --dataset_name longvideobench \
  --extract_feature_model blip \
  --score_path ./outscores/longvideobench/blip/scores.json \
  --frame_path ./outscores/longvideobench/blip/frames.json \
  --max_num_frames 64 \
  --aks_mode original
```

输出位置：

```text
selected_frames/longvideobench/blip/selected_frames.json
selected_frames/longvideobench/blip/selection_manifest.json
```

旁路 `selection_manifest.json` 记录两个输入 JSON 的绝对路径、SHA-256、全部
AKS 参数和每条记录的实际输出数量。它不会改变后续评估脚本读取的
`selected_frames.json`。

若要使用补齐配额的应用版本，只需改为：

```bash
python frame_select.py \
  --dataset_name longvideobench \
  --extract_feature_model blip \
  --score_path ./outscores/longvideobench/blip/scores.json \
  --frame_path ./outscores/longvideobench/blip/frames.json \
  --max_num_frames 48 \
  --aks_mode robust
```

`--ratio 2` 表示对已有候选分数再隔一个取一个。默认 `--ratio 1` 不进行二次
降采样。

## 7. 主要参数

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `--aks-mode` | `robust`（V2） | `original` 原配额，`robust` 补足预算 |
| `--max-num-frames` | 32（V2） | 下游模型接收的最大关键帧数 |
| `--candidate-sampling` | `interval` | 候选帧采样规则 |
| `--sample-interval` | 1.0 | `interval` 模式的秒间隔 |
| `--threshold` | 0.8 | AKS 的峰值停止阈值 `t1` |
| `--std-threshold` | -100 | AKS 的标准差阈值 `t2` |
| `--max-depth` | 5 | 最大二分深度 |
| `--feature-backend` | clip | `clip`、`pangu`、`mep`、`http` 或 `python` |
| `--feature-config` | 无 | 远端特征后端的 JSON 配置 |
| `--batch-size` | 16 | 候选图片打分批大小 |
| `--device` | auto | CLIP 后端自动选择 CUDA、MPS 或 CPU |
| `--model-name` | CLIP ViT-B/32 | CLIP 后端的模型名或本地路径 |

完整参数可查看：

```bash
python aks_keyframes_v2.py --help
python frame_select.py --help
```

## 8. 中文 query 与非视觉问题

默认 OpenAI CLIP 对英文 query 更可靠。中文场景建议：

1. 使用英文检索 query；或
2. 通过 `--model-name` 使用兼容的多语 CLIP checkpoint。

当前链路只根据画面选择帧，不处理音频和字幕。对于“说了什么”“字幕何时出现”
等问题，需要额外加入 ASR/字幕相关性，再与视觉分数融合。

## 9. 运行核心测试

```bash
python -m unittest discover -s tests -v
```

测试覆盖：

- 32/64 帧标准预算与原始选择规则的一致性；
- 非整除预算下 `original` 与 `robust` 的差异；
- 常量分数、短视频和非法输入；
- 原始 FPS 采样与真实秒间隔采样的差异。
- 通用余弦评分、向量维度检查、multipart HTTP 与 MEP 请求协议。
