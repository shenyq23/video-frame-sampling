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

## 特征模型配置

CLIP 可以直接在页面填写 Hugging Face checkpoint 或本地模型路径。第一次使用远程 checkpoint 时，需要由运行环境下载模型。

Pangu/MEP 的服务地址和非敏感参数放在 `config/feature_models.json`，密钥放环境变量：

```bash
export PANGU_EMBED_API_KEY='...'
export MEP_EMBED_APPID='...'
export MEP_EMBED_SECRET_KEY='...'
```

修改对应档案并把 `enabled` 设为 `true` 后，网页才会显示该档案。API 不会把档案的具体配置或密钥返回浏览器。

## 结果 Manifest

除原视频帧号、时间戳和相关性分数外，标准 manifest 还记录：

- `candidate_index/candidate_order`：候选池中的 0-based/1-based 序号；
- `normalized_score`：用于图表的归一化相关性；
- `segment_id/segment_depth/segment_quota`：AKS 分段信息；
- `rank_in_segment`：该候选帧在所属 segment 中的分数排名；
- 全部候选帧的紧凑分数序列，用于绘制相关性曲线。

## 测试

```bash
cd keyframe_visualizer/backend
python3 -m unittest discover -s tests -v

cd ../frontend
npm run build
```

后端启动不要求立即加载 CLIP；只有提交 CLIP 任务时才加载模型。GPU/MPS 模型任务由单 worker 顺序执行，避免并发任务同时占满显存。
