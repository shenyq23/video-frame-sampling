# Keyframe Visualizer

一个独立于现有 AKS 输出目录的可视化工作台。当前接入 AKS，支持上传视频、输入 query、选择 AKS 模式、关键帧预算、候选帧采样间隔和 CLIP/Pangu/MEP 特征后端。

所有运行数据都写入 `keyframe_visualizer/data/`，不会修改 AKS 已有文件或输出。

## 目录

- `backend/`：FastAPI、SQLite 任务队列和 AKS Adapter。
- `frontend/`：React + TypeScript 页面。
- `config/feature_models.json`：服务端 Pangu/MEP 配置档案。
- `data/uploads/`：上传视频，首次启动时自动创建。
- `data/runs/<job-id>/`：关键帧和标准 manifest。

## 安装

推荐在 AKS 已有 Python 环境中安装后端依赖：

```bash
cd keyframe_visualizer/backend
python3 -m pip install -r requirements.txt
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
./keyframe_visualizer/scripts/start_frontend.sh
```

浏览器打开 `http://127.0.0.1:5173`。API 文档位于 `http://127.0.0.1:8000/docs`。

## 候选帧间隔

候选帧模式选择“按时间间隔”后，可以输入任意大于 0 的有限数字，例如 `0.333`、
`1.27` 或 `120.5`。视频只能在整数帧位置解码，因此结果页会同时显示请求间隔和
按视频 FPS 对齐后的实际间隔。小于单帧时长的输入等价于每帧采样。

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

修改后重启后端。网页会显示服务配置是否就绪；缺少密钥的配置无法选择。真实 `.env`
已被 `.gitignore` 排除，已有 Shell 环境变量优先于 `.env` 中的同名值。

Pangu/MEP 的服务地址和非敏感参数仍放在 `config/feature_models.json`。如果暂时不希望
在网页中显示某个服务，可以把对应 profile 的 `enabled` 改为 `false`。

## CLIP 模型

CLIP 支持三种来源：

- Hugging Face 模型 ID，例如 `openai/clip-vit-base-patch32`；
- 后端机器可访问的绝对目录；
- 网页上传的离线模型压缩包。

上传时选择 ZIP、TAR、TAR.GZ 或 TGZ 文件。压缩包中必须只有一个 Hugging Face CLIP
模型，且至少包含 `config.json`、`preprocessor_config.json` 和 `.safetensors` 或
`.bin` 权重，以及 `tokenizer.json` 或 `vocab.json + merges.txt`。后端会拒绝路径穿越、
链接、特殊文件、不完整模型和超过限制的压缩包。
校验成功后模型保存在 `data/models/clip/<model-id>/`，并自动出现在 CLIP 模型下拉框。

如果前后端运行在同一台机器，不上传也可以直接填写模型绝对路径。前后端位于不同
机器时，浏览器本机路径对后端无效，应使用压缩包上传。

## 结果 Manifest

除原视频帧号、时间戳和相关性分数外，标准 manifest 还记录：

- `candidate_index/candidate_order`：候选池中的 0-based/1-based 序号；
- `normalized_score`：用于图表的归一化相关性；
- `segment_id/segment_depth/segment_quota`：AKS 分段信息；
- `rank_in_segment`：该候选帧在所属 segment 中的分数排名；
- 全部候选帧的紧凑分数序列，用于绘制相关性曲线。

新任务默认额外保存同数量均匀抽帧和全部候选帧，分别写入 `uniform_frames/` 和
`candidate_frames/`。详情页可以在三组结果之间切换。高级参数中可以关闭任一额外
输出以节省磁盘空间；关闭后对应详情页按钮不可用。此设置等价于命令行入口的
`--save-uniform-baseline` 和 `--save-candidate-frames`。

详情页同时展示任务创建时保存的完整参数快照，包括特征后端、模型或服务 profile、
AKS 模式、候选采样、阈值、深度、设备和导出选项。同一视频和 query 的不同运行可以
据此区分。旧任务不会自动补生成均匀帧和候选帧，需要重新运行后才能切换查看。

成功或失败任务的详情页提供“清除任务及数据”按钮。确认后会删除该任务上传的原视频、
`data/runs/<job-id>/` 下的所有帧、manifest 和中间结果，并删除 SQLite 中的任务记录。
操作不可恢复。排队中或运行中的任务不能清除，以避免后台写入和删除发生竞争。上传的
CLIP 模型属于多个任务可复用的共享资源，不会随单个任务一起删除。

## 测试

```bash
cd keyframe_visualizer/backend
python3 -m unittest discover -s tests -v

cd ../frontend
npm run build
```

后端启动不要求立即加载 CLIP；只有提交 CLIP 任务时才加载模型。GPU/MPS 模型任务由单 worker 顺序执行，避免并发任务同时占满显存。
