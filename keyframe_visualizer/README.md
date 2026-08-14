# Keyframe Visualizer

一个独立于算法仓库输出目录的可视化工作台。当前接入 AKS、VSI 和 SAGE：左上角可以切换算法；三种算法分别使用自己的视频预处理和 query 参数界面，但共享视频 session、query 历史、帧画廊、任务删除和 VLM 问答能力。

所有运行数据都写入 `keyframe_visualizer/data/`，不会修改 AKS 已有文件或输出。

## 目录

- `backend/`：FastAPI、SQLite 任务队列以及 AKS/VSI/SAGE Adapter。
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

在 Windows 上，后端进程在导入 `app` 包时就会在主线程里加载一次 Torch（`app/torch_runtime.py`）。预处理和抽帧任务运行在 `asyncio.to_thread` 的工作线程里，如果把 Torch 留到那时才由 EasyOCR/Ultralytics 间接导入，`torch\lib\c10.dll` 会以 `[WinError 1114] 动态链接库(DLL)初始化例程失败` 加载失败，而同一环境下的 `examples/run_vsi.py` 因为在主线程模块级导入 Torch 所以正常。预加载失败时后端仍会启动，并在标准错误里打印原因，任务失败信息也会带上同样的提示。用 `KFV_PRELOAD_TORCH=0` 可以关闭预加载，`KFV_PRELOAD_TORCH=1` 可以在非 Windows 平台上强制开启。

## 候选帧间隔

候选帧模式选择“按时间间隔”后，可以输入任意大于 0 的有限数字，例如 `0.333`、`1.27` 或 `120.5`。视频只能在整数帧位置解码，因此结果页会同时显示请求间隔和按视频 FPS 对齐后的实际间隔。小于单帧时长的输入等价于每帧采样。

## 数值参数输入

所有数值参数（候选间隔、目标帧数、Batch size、解码线程、缓存 JPEG 质量、t1/t2 阈值、最大深度，VSI 的 OCR 采样 FPS、OCR 截取区域、Top-K、检测帧预算、每轮采样数量、Text weight、随机种子，以及 SAGE 的关键帧预算）统一使用同一种输入框，行为如下：

- 打开表单时预填推荐值，全选删除后输入框会真正变空，只显示灰色示例，不会残留 `0` 导致出现 `01.2` 这样的拼接结果；
- 一律不提供上下箭头调参；
- 留空提交时沿用该参数的推荐值（即预填的那个值）；
- 填了内容就严格校验，取值范围与 `backend/app/schemas.py` 中的 `AKSParameters` / `VSIParameters` 一致。整数项（目标帧数、Top-K、随机种子等）只接受整数，`1.5` 和 `abc` 一样会被拒绝；小数项只接受十进制写法，`1e3`、`Infinity`、`1.2.3` 均不合法。校验不通过时表单直接报错，不会发起预处理或抽帧请求。

下拉选择类参数（特征模型、设备、字幕来源等）和文本类参数（模型 ID、检测目标）不受影响。

## 结果页布局

页面分两列。左侧一栏依次是算法切换、参数表单、视频记录和当前视频的 Query 历史（含删除整个视频会话的按钮）；右侧整块都留给结果展示。

结果区把两项重点结果并排放在最上面：

- 左边是抽出的关键帧（可以切换抽出帧 / 均匀抽帧 / 候选帧），下面紧接相关性或 VSI 分数折线图；
- 右边是 VLM 的一问一答，包含使用的证据帧缩略图；窗口足够宽时它会随页面滚动固定在视口顶部。

预处理耗时和抽帧耗时以胶囊形式放在结果标题正下方。原视频、本次运行参数快照和帧数统计属于参考信息，收在结果下方的“相关信息”折叠区里，默认展开，视频播放器也限制了尺寸。窗口宽度小于 1240px 时两栏自动堆叠为上下排列。

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

然后根据实际环境修改 `config/vlm_models.json` 中的 `elb`、`b_id` 和 `flow_id`。重启后端，网页会在成功任务详情页显示“VLM 问答”面板。

只要后端至少有一个可用的 VLM 配置，左侧的抽帧按钮就会变成“抽帧并生成 VLM 回答”：点击一次会先提交抽帧任务，任务成功后自动把 `<算法>抽出帧` 和同一条 Query 发给默认可用的 VLM 服务，无需再点第二个按钮。抽帧失败、没有可用 VLM 配置，或这组帧已经存在保存过的回答时，自动问答不会重复触发。

结果页仍然可以手动切换输入帧集合并重新提问，展开面板下方的“VLM 输入”即可换服务或改写 Query。可选的三种输入是：

- AKS/VSI/SAGE 抽出帧；
- 同数量均匀抽帧；
- 所有候选帧 / VSI 访问帧。

切换帧集合时会读取该集合已保存的回答；没有保存过的集合需要手动点“生成 VLM 回答”。后端会从当前任务的 Manifest 读取图片，按视频时间顺序组成多图请求，并把回答保存到：

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

## SAGE 使用

SAGE 源码和模型默认从与 `keyframe_visualizer` 平级的 `SAGE/` 目录加载。Windows 后端需要能读取以下本地资源：

- `SAGE/models/clip/ViT-B-32.pt`；
- `SAGE/models/sentence_transformer/paraphrase-multilingual-mpnet-base-v2/`。

进入网页后选择 `SAGE`，准备视频时可以选择三种 ASR 来源：

- `远程 ASR`：Windows 将视频上传到 `10.97.134.3:8091`，轮询服务器任务，下载 ASR JSON 后再执行本地 SAGE；
- `上传现有 JSON`：随视频上传一个已经生成且能被 `SAGE/sage_frame/io.py` 读取的 JSON；
- `不使用 ASR`：写入空的 ASR 段列表，运行纯视觉 SAGE。

远程模式需要在 Windows 的 `keyframe_visualizer/.env` 中配置：

```dotenv
SAGE_ASR_BASE_URL=http://10.97.134.3:8091
SAGE_ASR_TOKEN=your-real-token
SAGE_ASR_CONNECT_TIMEOUT=15
SAGE_ASR_UPLOAD_TIMEOUT=600
SAGE_ASR_DOWNLOAD_TIMEOUT=120
SAGE_ASR_JOB_TIMEOUT=2400
SAGE_ASR_POLL_INTERVAL=5
SAGE_ASR_DELETE_REMOTE_AFTER_DOWNLOAD=true
```

真实 Token 只保存在后端 `.env`，不会返回前端、写入 SQLite 参数或 Manifest。视频和 ASR 预处理结果保存在：

```text
data/sessions/<session-id>/source/          原视频及上传的 asr.json
data/sessions/<session-id>/preprocess/asr.json
data/sessions/<session-id>/preprocess/metadata.json
data/sessions/<session-id>/preprocess/remote_asr_job.json
```

`remote_asr_job.json` 保存远程任务 ID 和下载状态。后端在上传、轮询或下载期间重启后会恢复同一个远程任务；JSON 已成功下载时会直接复用本地文件。`SAGE_ASR_DELETE_REMOTE_AFTER_DOWNLOAD=true` 时，本地 JSON 校验通过后会尽力清理服务器任务，清理失败不会丢失本地结果。

视频准备完成后可以针对同一视频连续输入不同 Query 和关键帧预算。SAGE 对每条 Query 动态生成候选帧，因此候选帧缓存属于该 Query 的运行目录，而不是视频预处理目录。结果页提供 SAGE 抽出帧、同数量均匀抽帧、SAGE 候选帧，以及视觉相关性和视觉变化曲线。

SAGE 必须通过视频 Session 使用；通用的单次 `/api/jobs` 上传接口会拒绝 SAGE 请求，以免跳过 ASR 准备流程。

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
