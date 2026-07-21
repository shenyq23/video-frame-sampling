import { useMemo, useState } from "react";
import type { AlgorithmMetadata, RunParameters } from "../types";

interface Props {
  algorithm?: AlgorithmMetadata;
  busy: boolean;
  onSubmit: (video: File, query: string, parameters: RunParameters) => Promise<void>;
}

const defaults: RunParameters = {
  aks_mode: "robust",
  max_num_frames: 32,
  candidate_sampling: "interval",
  sample_interval: 1,
  feature_backend: "clip",
  feature_profile: null,
  model_name: "openai/clip-vit-base-patch32",
  device: "auto",
  batch_size: 16,
  decode_threads: 2,
  threshold: 0.8,
  std_threshold: -100,
  max_depth: 5,
  jpeg_quality: 92,
};

export function RunForm({ algorithm, busy, onSubmit }: Props) {
  const [video, setVideo] = useState<File | null>(null);
  const [query, setQuery] = useState("");
  const [parameters, setParameters] = useState<RunParameters>(defaults);
  const [advanced, setAdvanced] = useState(false);
  const [error, setError] = useState("");

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
    if (parameters.feature_backend !== "clip" && !parameters.feature_profile) {
      setError("远程特征模型需要先在服务端启用一个配置档案。");
      return;
    }
    setError("");
    await onSubmit(video, query.trim(), parameters);
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
          <input type="number" min="0.01" max="60" step="0.05" disabled={parameters.candidate_sampling === "original"} value={parameters.sample_interval} onChange={(e) => update("sample_interval", Number(e.target.value))} />
        </label>
        <label className="field">
          <span>特征模型</span>
          <select value={parameters.feature_backend} onChange={(e) => {
            update("feature_backend", e.target.value as RunParameters["feature_backend"]);
            update("feature_profile", null);
          }}>
            <option value="clip">CLIP</option>
            <option value="pangu">Pangu</option>
            <option value="mep">MEP</option>
          </select>
        </label>
        {parameters.feature_backend === "clip" ? (
          <label className="field">
            <span>CLIP checkpoint</span>
            <input value={parameters.model_name} onChange={(e) => update("model_name", e.target.value)} />
          </label>
        ) : (
          <label className="field">
            <span>服务配置</span>
            <select value={parameters.feature_profile ?? ""} onChange={(e) => update("feature_profile", e.target.value || null)}>
              <option value="">请选择已启用配置</option>
              {profiles.map((profile) => <option key={profile.id} value={profile.id}>{profile.name}</option>)}
            </select>
          </label>
        )}
      </div>

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
        </div>
      )}

      {error && <p className="form-error" role="alert">{error}</p>}
      <button className="primary-button" type="submit" disabled={busy}>{busy ? "正在上传…" : "开始抽帧"}</button>
    </form>
  );
}

