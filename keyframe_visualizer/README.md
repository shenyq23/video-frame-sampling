# Keyframe Visualizer

一个独立于算法仓库输出目录的可视化工作台。当前接入 AKS 和 VSI：左上角可以切换算法；两种算法分别使用自己的视频预处理和 query 参数界面，但共享视频 session、query 历史、帧画廊、任务删除和 VLM 问答能力。

所有运行数据都写入 `keyframe_visualizer/data/`，不会修改 AKS 已有文件或输出。

## 目录

- `backend/`：FastAPI、SQLite 任务队列以及 AKS/VSI Adapter。
- `frontend/`：React + TypeScript 页面。
- `config/feature_models.json`：服务端 Pangu/MEP 配置档案。
- `config/vlm_models.json`：服务端 VLM 配置档案（当前支持 MEP VLM）。
- `data/uploads/`：上传视频，首次启动时自动创建。
- `data/runs/<job-id>/`：关键帧和标准 manifest。

## 安装

推荐在 AKS 已有 Python 环境中安装后端依赖：

```bash
cd keyframe_visualizer/backend
python -m pip install -r requirements.txt
```

安装前端依赖：

```bash
cd keyframe_visualizer/frontend
npm install
```

## 启动

从两个终端分别启动：

```bash
./keyframe_visualizer/scripts/start_backend.sh
```

```bash
./keyframe_visualizer/scripts/start_frontend.sh
```

浏览器打开
```bash
http://127.0.0.1:5173
```

API 文档位于
```bash
http://127.0.0.1:8000/docs
```

## 候选帧间隔

候选帧模式选择“按时间间隔”后，可以输入任意大于 0 的有限数字，例如 `0.333`、`1.27` 或 `120.5`。视频只能在整数帧位置解码，因此结果页会同时显示请求间隔和按视频 FPS 对齐后的实际间隔。小于单帧时长的输入等价于每帧采样。输入框默认留空，灰色文本仅为示例；必须由用户输入数值，且不提供上下箭头调参。

目标帧数也默认留空，只显示灰色示例。用户必须输入 `1～512` 的整数，输入框不提供上下箭头调参。

## Pangu/MEP 密钥配置

密钥不会通过网页上传，也不会写入任务数据库或 manifest。首次使用时复制示例文件：

```bash
cd keyframe_visualizer
cp .env.example .env
```

编辑 `.env`：

```dotenv
PANGU_EMBED_API_KEY=your-pangu-api-key
MEP_EMBED_APPID=your-mep-app-id
MEP_EMBED_SECRET_KEY=your-mep-secret-key
```

修改后重启后端。网页会显示服务配置是否就绪；缺少密钥的配置无法选择。真实 `.env` 已被 `.gitignore` 排除，已有 Shell 环境变量优先于 `.env` 中的同名值。

Pangu/MEP 的服务地址和非敏感参数仍放在 `config/feature_models.json`。如果暂时不希望在网页中显示某个服务，可以把对应 profile 的 `enabled` 改为 `false`。

## CLIP 模型

CLIP 支持三种来源：

- Hugging Face 模型 ID，例如 `openai/clip-vit-base-patch32`；
- 后端机器可访问的绝对目录；
- 网页上传的离线模型压缩包。

上传时选择 ZIP、TAR、TAR.GZ 或 TGZ 文件。压缩包中必须只有一个 Hugging Face CLIP 模型，且至少包含 `config.json`、`preprocessor_config.json` 和 `.safetensors` 或 `.bin` 权重，以及 `tokenizer.json` 或 `vocab.json + merges.txt`。后端会拒绝路径穿越、链接、特殊文件、不完整模型和超过限制的压缩包。
校验成功后模型保存在 `data/models/clip/<model-id>/`，并自动出现在 CLIP 模型下拉框。

如果前后端运行在同一台机器，不上传也可以直接填写模型绝对路径。前后端位于不同机器时，浏览器本机路径对后端无效，应使用压缩包上传。

## 结果 Manifest

除原视频帧号、时间戳和相关性分数外，标准 manifest 还记录：

- `candidate_index/candidate_order`：候选池中的 0-based/1-based 序号；
- `normalized_score`：用于图表的归一化相关性；
- `segment_id/segment_depth/segment_quota`：AKS 分段信息；
- `rank_in_segment`：该候选帧在所属 segment 中的分数排名；
- 全部候选帧的紧凑分数序列，用于绘制相关性曲线。

新任务默认额外保存同数量均匀抽帧和全部候选帧，分别写入 `uniform_frames/` 和 `candidate_frames/`。详情页可以在三组结果之间切换。高级参数中可以关闭任一额外输出以节省磁盘空间；关闭后对应详情页按钮不可用。此设置等价于命令行入口的 `--save-uniform-baseline` 和 `--save-candidate-frames`。

详情页同时展示任务创建时保存的完整参数快照，包括特征后端、模型或服务 profile、AKS 模式、候选采样、阈值、深度、设备和导出选项。同一视频和 query 的不同运行可以据此区分。旧任务不会自动补生成均匀帧和候选帧，需要重新运行后才能切换查看。

成功或失败任务的详情页提供“清除任务及数据”按钮。确认后会删除该任务上传的原视频、`data/runs/<job-id>/` 下的所有帧、manifest 和中间结果，并删除 SQLite 中的任务记录。操作不可恢复。排队中或运行中的任务不能清除，以避免后台写入和删除发生竞争。上传的 CLIP 模型属于多个任务可复用的共享资源，不会随单个任务一起删除。

在 Windows 上，前端会先卸载详情页并终止视频请求，再调用删除接口；后端还会对 WinError 32 文件占用进行短暂重试。如果视频同时被其他浏览器标签页或外部播放器打开，接口会返回明确的 `423 Locked`，关闭占用程序后可直接再次点击清除。

无论任务成功或失败，任务 worker 都会在写入最终状态前显式释放 Decord/OpenCV 的视频解码器句柄，因此失败任务也可以直接从详情页清除。

## 抽帧结果结合 VLM 问答

VLM 配置与抽帧特征配置分开管理。复制 `.env.example` 后补充 MEP VLM 的认证信息：

```dotenv
MEP_VLM_APPID=your-vlm-app-id
MEP_VLM_SECRET_KEY=your-vlm-secret-key
```

然后根据实际环境修改 `config/vlm_models.json` 中的 `elb`、`b_id` 和 `flow_id`。重启后端，网页会在成功任务详情页显示“基于当前帧集合回答 Query”区域。可以在以下三种输入之间切换：

- AKS 抽出帧；
- 同数量均匀抽帧；
- 所有候选帧。

Query 默认使用抽帧任务的 Query，也可以在详情页重新输入。点击“生成 VLM 回答”后，后端会从当前任务的 Manifest 读取图片，按视频时间顺序组成多图请求，并把回答保存到：

```text
data/runs/<job-id>/vlm_results/<frame-set>.json
```

回答区域会展示使用的帧数量、是否因服务的 `max_frames` 限制而均匀缩减，以及实际发送给 VLM 的证据帧和时间戳。服务配置中的 `max_image_dimension`、`jpeg_quality` 和 `max_frames` 可以用来控制请求体大小。认证信息只从后端环境变量读取，不会进入网页、SQLite 或 Manifest。

如果服务配置缺少环境变量，网页会将该配置标记为不可用；补齐 `.env` 后必须重启后端。

## 测试

```bash
cd keyframe_visualizer/backend
python -m unittest discover -s tests -v

cd ../frontend
npm run build
```

后端启动不要求立即加载 CLIP；只有提交 CLIP 任务时才加载模型。GPU/MPS 模型任务由单 worker 顺序执行，避免并发任务同时占满显存。

## VSI 使用

VSI 源码默认从与 `keyframe_visualizer` 平级的 `VSI_VideoFraming/` 目录加载。旧的 `VSI/` 目录不会再被可视化后端引用。进入网页后，在左上角选择 `VSI`。

准备视频时可以选择：

- 烧录字幕 OCR：上传视频后执行 EasyOCR，并把字幕缓存到视频 session；
- 上传字幕：支持 `.srt` 和 `.json`；
- 不使用字幕：仅运行 YOLO-World 视觉分支。

视频准备完成后，每条 query 需要同时输入检测目标 `Objects`，多个目标使用英文逗号分隔。例如：

```text
Query: 视频中什么时候有人骑马经过道路？
Objects: person, horse, road
```

同一个视频的不同 query 可以使用不同 Objects。OCR 或上传字幕只在准备视频时处理一次；YOLO-World 检测和 VSI 自适应采样会针对每条 query 重新执行。

VSI 任务会保存：

- 最终选中的关键帧；
- 同数量均匀抽帧；
- YOLO-World 实际访问过的帧；
- fused score 和 sampling probability；
- visited frame CSV、可选的全部帧 CSV 和采样历史；
- 与 AKS 相同格式的 VLM 回答记录。

新版 `VSI_VideoFraming` 已随仓库提供 YOLO-World、CLIP、EasyOCR 和默认 SentenceTransformer 资源。同步代码后先确保 Git LFS 大文件已经拉取：

```bash
git -C VSI_VideoFraming lfs pull
```

网页的 VSI 视频准备区会显示四项本地资源状态。默认配置会优先使用：

- `VSI_VideoFraming/yolov8s-worldv2.pt`；
- `VSI_VideoFraming/weights/clip/ViT-B-32.pt`；
- `VSI_VideoFraming/output/easyocr_models/`；
- `VSI_VideoFraming/weights/sentence_transformer/paraphrase-multilingual-mpnet-base-v2/` 中的扁平字幕文本模型。

可视化自身产生的运行缓存仍写入 `keyframe_visualizer/data/models/vsi/`。如果选择自定义 YOLO 或字幕模型名称，相关组件仍可能按照第三方库的行为联网下载；绝对路径形式的自定义 YOLO 权重会在任务启动前检查文件完整性。

当前使用的 `VSI_VideoFraming` 不包含 ASR 接口，网页第一版只支持 OCR、上传字幕和纯视觉模式。
