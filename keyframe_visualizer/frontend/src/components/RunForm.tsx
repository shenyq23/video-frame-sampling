import { useMemo, useState } from "react";
import type { AlgorithmMetadata, ClipModel, RunParameters } from "../types";

interface Props {
  algorithm?: AlgorithmMetadata;
  clipModels: ClipModel[];
  busy: boolean;
  onSubmit: (video: File, query: string, parameters: RunParameters) => Promise<void>;
  onUploadClipModel: (archive: File, name: string) => Promise<ClipModel>;
}

const defaults: RunParameters = {
  aks_mode: "robust",
  max_num_frames: 32,
  candidate_sampling: "interval",
  sample_interval: 1,
  feature_backend: "clip",
  feature_profile: null,
  clip_model_id: null,
  model_name: "openai/clip-vit-base-patch32",
  device: "auto",
  batch_size: 16,
  decode_threads: 2,
  threshold: 0.8,
  std_threshold: -100,
  max_depth: 5,
  jpeg_quality: 92,
  save_uniform_baseline: true,
  save_candidate_frames: true,
};

export function RunForm({ algorithm, clipModels, busy, onSubmit, onUploadClipModel }: Props) {
  const [video, setVideo] = useState<File | null>(null);
  const [query, setQuery] = useState("");
  const [parameters, setParameters] = useState<RunParameters>(defaults);
  const [advanced, setAdvanced] = useState(false);
  const [error, setError] = useState("");
  const [modelArchive, setModelArchive] = useState<File | null>(null);
  const [modelDisplayName, setModelDisplayName] = useState("");
  const [uploadingModel, setUploadingModel] = useState(false);

  const profiles = useMemo(
    () =>
      (algorithm?.parameter_schema.feature_profiles ?? []).filter(
        (profile) => profile.backend === parameters.feature_backend,
      ),
    [algorithm, parameters.feature_backend],
  );

  const update = <K extends keyof RunParameters>(key: K, value: RunParameters[K]) =>
    setParameters((current) => ({ ...current, [key]: value }));

  const submit = async (event: React.FormEvent) => {
    event.preventDefault();
    if (!video || !query.trim()) {
      setError("请选择视频并输入 query。");
      return;
    }
    if (
      parameters.candidate_sampling === "interval" &&
      (!Number.isFinite(parameters.sample_interval) || parameters.sample_interval <= 0)
    ) {
      setError("候选间隔必须是大于 0 的有限数字。");
      return;
    }
    if (parameters.feature_backend !== "clip" && !parameters.feature_profile) {
      setError("远程特征模型需要先在服务端启用一个配置档案。");
      return;
    }
    const selectedProfile = profiles.find((profile) => profile.id === parameters.feature_profile);
    if (selectedProfile && !selectedProfile.credentials_ready) {
      setError(`服务端缺少环境变量：${selectedProfile.missing_environment_variables.join(", ")}`);
      return;
    }
    setError("");
    await onSubmit(video, query.trim(), parameters);
  };

  const uploadModel = async () => {
    if (!modelArchive) {
      setError("请先选择一个 CLIP 模型压缩包。");
      return;
    }
    setError("");
    setUploadingModel(true);
    try {
      const model = await onUploadClipModel(modelArchive, modelDisplayName.trim());
      update("clip_model_id", model.id);
      setModelArchive(null);
      setModelDisplayName("");
    } catch (uploadError) {
      setError(uploadError instanceof Error ? uploadError.message : "CLIP 模型上传失败");
    } finally {
      setUploadingModel(false);
    }
  };

  return (
    <form className="run-form" onSubmit={submit}>
      <div className="section-heading">
        <div>
          <p className="eyebrow">NEW RUN</p>
          <h2>配置抽帧任务</h2>
        </div>
        <span className="algorithm-chip">AKS</span>
      </div>

      <label className="field field-wide">
        <span>视频文件</span>
        <input
          type="file"
          accept="video/mp4,video/quicktime,video/x-matroska,video/webm,video/x-msvideo"
          onChange={(event) => setVideo(event.target.files?.[0] ?? null)}
          required
        />
        <small>{video ? `${video.name} · ${(video.size / 1024 / 1024).toFixed(1)} MB` : "最大 8 GB"}</small>
      </label>

      <label className="field field-wide">
        <span>Query</span>
        <textarea
          rows={3}
          value={query}
          onChange={(event) => setQuery(event.target.value)}
          placeholder="例如：the person opens the red suitcase"
          required
        />
      </label>

      <div className="field-grid">
        <label className="field">
          <span>AKS 模式</span>
          <select value={parameters.aks_mode} onChange={(e) => update("aks_mode", e.target.value as RunParameters["aks_mode"])}>
            <option value="robust">Robust · 补足预算</option>
            <option value="original">Original · 论文配额</option>
          </select>
        </label>
        <label className="field">
          <span>目标帧数</span>
          <input type="number" min="1" max="512" value={parameters.max_num_frames} onChange={(e) => update("max_num_frames", Number(e.target.value))} />
        </label>
        <label className="field">
          <span>候选帧模式</span>
          <select value={parameters.candidate_sampling} onChange={(e) => update("candidate_sampling", e.target.value as RunParameters["candidate_sampling"])}>
            <option value="interval">按时间间隔</option>
            <option value="original">原仓库 int(FPS)</option>
          </select>
        </label>
        <label className="field">
          <span>候选间隔（秒）</span>
          <input type="number" step="any" disabled={parameters.candidate_sampling === "original"} value={parameters.sample_interval} onChange={(e) => update("sample_interval", Number(e.target.value))} />
          <small>支持任意正数；实际间隔会对齐到视频帧。</small>
        </label>
        <label className="field">
          <span>特征模型</span>
          <select value={parameters.feature_backend} onChange={(e) => {
            update("feature_backend", e.target.value as RunParameters["feature_backend"]);
            update("feature_profile", null);
            update("clip_model_id", null);
          }}>
            <option value="clip">CLIP</option>
            <option value="pangu">Pangu</option>
            <option value="mep">MEP</option>
          </select>
        </label>
        {parameters.feature_backend === "clip" ? (
          <label className="field">
            <span>CLIP 模型</span>
            <select value={parameters.clip_model_id ?? ""} onChange={(e) => update("clip_model_id", e.target.value || null)}>
              <option value="">Hugging Face ID / 服务端路径</option>
              {clipModels.map((model) => <option key={model.id} value={model.id}>{model.name}</option>)}
            </select>
          </label>
        ) : (
          <label className="field">
            <span>服务配置</span>
            <select value={parameters.feature_profile ?? ""} onChange={(e) => update("feature_profile", e.target.value || null)}>
              <option value="">请选择已启用配置</option>
              {profiles.map((profile) => <option key={profile.id} value={profile.id} disabled={!profile.credentials_ready}>{profile.name}{profile.credentials_ready ? "" : " · 缺少密钥"}</option>)}
            </select>
            {parameters.feature_profile && <small>{profiles.find((profile) => profile.id === parameters.feature_profile)?.credentials_ready ? "服务端凭据已就绪" : "服务端凭据未配置"}</small>}
          </label>
        )}
      </div>

      {parameters.feature_backend === "clip" && !parameters.clip_model_id && (
        <label className="field field-wide">
          <span>模型 ID 或服务端绝对路径</span>
          <input value={parameters.model_name} onChange={(e) => update("model_name", e.target.value)} placeholder="openai/clip-vit-base-patch32" />
        </label>
      )}

      {parameters.feature_backend === "clip" && (
        <div className="model-uploader">
          <div className="model-uploader-title"><strong>上传离线 CLIP 模型</strong><small>ZIP / TAR / TAR.GZ / TGZ，最大 12 GB</small></div>
          <div className="model-upload-fields">
            <input aria-label="模型显示名称" value={modelDisplayName} onChange={(e) => setModelDisplayName(e.target.value)} placeholder="模型显示名称（可选）" />
            <input aria-label="CLIP 模型压缩包" type="file" accept=".zip,.tar,.gz,.tgz,application/zip,application/gzip" onChange={(e) => setModelArchive(e.target.files?.[0] ?? null)} />
            <button className="secondary-button" type="button" disabled={uploadingModel} onClick={uploadModel}>{uploadingModel ? "上传并校验中…" : "上传模型"}</button>
          </div>
          {modelArchive && <small>已选择：{modelArchive.name} · {(modelArchive.size / 1024 / 1024).toFixed(1)} MB</small>}
        </div>
      )}

      <button className="text-button" type="button" onClick={() => setAdvanced((value) => !value)} aria-expanded={advanced}>
        {advanced ? "收起高级参数" : "展开高级参数"}
      </button>

      {advanced && (
        <div className="field-grid advanced-fields">
          <label className="field"><span>t1 threshold</span><input type="number" step="0.05" value={parameters.threshold} onChange={(e) => update("threshold", Number(e.target.value))} /></label>
          <label className="field"><span>t2 std threshold</span><input type="number" step="0.1" value={parameters.std_threshold} onChange={(e) => update("std_threshold", Number(e.target.value))} /></label>
          <label className="field"><span>最大深度</span><input type="number" min="0" max="16" value={parameters.max_depth} onChange={(e) => update("max_depth", Number(e.target.value))} /></label>
          <label className="field"><span>Batch size</span><input type="number" min="1" max="256" value={parameters.batch_size} onChange={(e) => update("batch_size", Number(e.target.value))} /></label>
          <label className="field"><span>设备</span><select value={parameters.device} onChange={(e) => update("device", e.target.value as RunParameters["device"])}><option value="auto">Auto</option><option value="cuda">CUDA</option><option value="mps">MPS</option><option value="cpu">CPU</option></select></label>
          <label className="field"><span>JPEG 质量</span><input type="number" min="1" max="100" value={parameters.jpeg_quality} onChange={(e) => update("jpeg_quality", Number(e.target.value))} /></label>
          <label className="check-field"><input type="checkbox" checked={parameters.save_uniform_baseline} onChange={(e) => update("save_uniform_baseline", e.target.checked)} /><span>保存同数量均匀抽帧</span></label>
          <label className="check-field"><input type="checkbox" checked={parameters.save_candidate_frames} onChange={(e) => update("save_candidate_frames", e.target.checked)} /><span>保存全部候选帧</span></label>
        </div>
      )}

      {error && <p className="form-error" role="alert">{error}</p>}
      <button className="primary-button" type="submit" disabled={busy}>{busy ? "正在上传…" : "开始抽帧"}</button>
    </form>
  );
}
