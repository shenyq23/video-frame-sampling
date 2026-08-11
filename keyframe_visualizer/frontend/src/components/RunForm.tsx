import { useEffect, useMemo, useState } from "react";
import type { AlgorithmMetadata, ClipModel, RunParameters, Session } from "../types";
import type { NumericSpecMap } from "../numeric";
import {
  numericDefaults,
  numericInputDefaults,
  numericInputsFrom,
  resolveNumericFields,
} from "../numeric";
import { NumberField } from "./NumberField";

interface Props {
  algorithm?: AlgorithmMetadata;
  clipModels: ClipModel[];
  currentSession: Session | null;
  preparingNewSession: boolean;
  busy: boolean;
  onPrepareSession: (video: File, parameters: RunParameters) => Promise<void>;
  onRunQuery: (query: string, parameters: RunParameters) => Promise<boolean>;
  onUploadClipModel: (archive: File, name: string) => Promise<ClipModel>;
  onPreparingNewSessionChange: (preparing: boolean) => void;
}

// Bounds mirror AKSParameters in backend/app/schemas.py.
const sessionNumericSpecs = {
  sample_interval: {
    label: "候选间隔（秒）",
    kind: "decimal",
    default: 1,
    min: 0,
    exclusiveMin: true,
    hint: "支持任意正数；实际间隔会对齐到视频帧。",
  },
  batch_size: { label: "Batch size", kind: "integer", default: 16, min: 1, max: 256 },
  decode_threads: { label: "解码线程", kind: "integer", default: 2, min: 1, max: 32 },
  jpeg_quality: { label: "缓存 JPEG 质量", kind: "integer", default: 92, min: 1, max: 100 },
} satisfies NumericSpecMap;

const queryNumericSpecs = {
  max_num_frames: { label: "目标帧数", kind: "integer", default: 32, min: 1, max: 512 },
  threshold: { label: "t1 threshold", kind: "decimal", default: 0.8 },
  std_threshold: { label: "t2 std threshold", kind: "decimal", default: -100 },
  max_depth: { label: "最大深度", kind: "integer", default: 5, min: 0, max: 16 },
} satisfies NumericSpecMap;

const defaults: RunParameters = {
  ...numericDefaults(sessionNumericSpecs),
  ...numericDefaults(queryNumericSpecs),
  aks_mode: "robust",
  candidate_sampling: "interval",
  feature_backend: "clip",
  feature_profile: null,
  clip_model_id: null,
  model_name: "openai/clip-vit-base-patch32",
  device: "auto",
  save_uniform_baseline: true,
  save_candidate_frames: true,
};

const sessionLockedKeys: Array<keyof RunParameters> = [
  "candidate_sampling",
  "sample_interval",
  "feature_backend",
  "feature_profile",
  "clip_model_id",
  "model_name",
  "device",
];

function labelFeatureBackend(value?: string) {
  if (value === "clip") return "CLIP";
  if (value === "pangu") return "Pangu";
  if (value === "mep") return "86M";
  return value ?? "—";
}

function sessionModelLabel(session: Session | null, clipModels: ClipModel[]) {
  const parameters = session?.parameters;
  if (!parameters) return "—";
  if (parameters.feature_backend === "clip") {
    const uploaded = clipModels.find((model) => model.id === parameters.clip_model_id);
    return uploaded?.name ?? parameters.clip_model_id ?? parameters.model_name ?? "—";
  }
  return parameters.feature_profile ?? "—";
}

export function RunForm({
  algorithm,
  clipModels,
  currentSession,
  preparingNewSession,
  busy,
  onPrepareSession,
  onRunQuery,
  onUploadClipModel,
  onPreparingNewSessionChange,
}: Props) {
  const [video, setVideo] = useState<File | null>(null);
  const [query, setQuery] = useState("");
  const [sessionNumbers, setSessionNumbers] = useState(() => numericInputDefaults(sessionNumericSpecs));
  const [queryNumbers, setQueryNumbers] = useState(() => numericInputDefaults(queryNumericSpecs));
  const [parameters, setParameters] = useState<RunParameters>(defaults);
  const [advanced, setAdvanced] = useState(false);
  const [error, setError] = useState("");
  const [modelArchive, setModelArchive] = useState<File | null>(null);
  const [modelDisplayName, setModelDisplayName] = useState("");
  const [uploadingModel, setUploadingModel] = useState(false);

  useEffect(() => {
    if (!currentSession?.parameters) return;
    setParameters((current) => ({
      ...current,
      ...currentSession.parameters,
    }) as RunParameters);
    setSessionNumbers(numericInputsFrom(sessionNumericSpecs, currentSession.parameters));
    setQuery("");
    setError("");
  }, [currentSession?.id]);

  const profiles = useMemo(
    () =>
      (algorithm?.parameter_schema.feature_profiles ?? []).filter(
        (profile) => profile.backend === parameters.feature_backend,
      ),
    [algorithm, parameters.feature_backend],
  );

  const update = <K extends keyof RunParameters>(key: K, value: RunParameters[K]) =>
    setParameters((current) => ({ ...current, [key]: value }));

  const updateSessionNumber = (key: string, value: string) =>
    setSessionNumbers((current) => ({ ...current, [key]: value }));

  const updateQueryNumber = (key: string, value: string) =>
    setQueryNumbers((current) => ({ ...current, [key]: value }));

  const prepareSession = async (event: React.FormEvent) => {
    event.preventDefault();
    if (!video) {
      setError("请先选择视频。");
      return;
    }
    // The interval box is disabled in int(FPS) mode, so leave it unvalidated.
    const resolved = resolveNumericFields(
      sessionNumericSpecs,
      sessionNumbers,
      parameters.candidate_sampling === "original" ? ["sample_interval"] : [],
    );
    if (!resolved.ok) {
      setError(resolved.message);
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
    await onPrepareSession(video, { ...parameters, ...resolved.values });
    setVideo(null);
  };

  const submitQuery = async (event: React.FormEvent) => {
    event.preventDefault();
    if (!currentSession || currentSession.status !== "succeeded") {
      setError("请先等待视频预处理完成。");
      return;
    }
    if (!query.trim()) {
      setError("请输入 query。");
      return;
    }
    const resolved = resolveNumericFields(queryNumericSpecs, queryNumbers);
    if (!resolved.ok) {
      setError(resolved.message);
      return;
    }
    const locked = { ...parameters, ...resolved.values };
    for (const key of sessionLockedKeys) {
      if (currentSession.parameters[key] !== undefined) {
        locked[key] = currentSession.parameters[key] as never;
      }
    }
    setError("");
    const submitted = await onRunQuery(query.trim(), locked);
    if (submitted) setQuery("");
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

  const showPreparationForm =
    preparingNewSession || !currentSession || currentSession.status === "failed";
  const sessionReady = currentSession?.status === "succeeded";

  if (showPreparationForm) {
    return (
      <form className="run-form" onSubmit={prepareSession}>
        <div className="section-heading">
          <div>
            <p className="eyebrow">VIDEO SESSION</p>
            <h2>准备视频</h2>
          </div>
          <span className="algorithm-chip">AKS</span>
        </div>

        {currentSession?.status === "failed" && (
          <p className="form-error" role="alert">上一个视频预处理失败：{currentSession.error}</p>
        )}

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

        <div className="field-grid">
          <label className="field">
            <span>候选帧模式</span>
            <select value={parameters.candidate_sampling} onChange={(e) => update("candidate_sampling", e.target.value as RunParameters["candidate_sampling"])}>
              <option value="interval">按时间间隔</option>
              <option value="original">原仓库 int(FPS)</option>
            </select>
          </label>
          <NumberField
            spec={sessionNumericSpecs.sample_interval}
            value={sessionNumbers.sample_interval}
            onChange={(value) => updateSessionNumber("sample_interval", value)}
            disabled={parameters.candidate_sampling === "original"}
          />
          <label className="field">
            <span>特征模型</span>
            <select value={parameters.feature_backend} onChange={(e) => {
              update("feature_backend", e.target.value as RunParameters["feature_backend"]);
              update("feature_profile", null);
              update("clip_model_id", null);
            }}>
              <option value="clip">CLIP</option>
              <option value="pangu">Pangu</option>
              <option value="mep">86M</option>
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
          {advanced ? "收起预处理参数" : "展开预处理参数"}
        </button>

        {advanced && (
          <div className="field-grid advanced-fields">
            <NumberField
              spec={sessionNumericSpecs.batch_size}
              value={sessionNumbers.batch_size}
              onChange={(value) => updateSessionNumber("batch_size", value)}
            />
            <label className="field"><span>设备</span><select value={parameters.device} onChange={(e) => update("device", e.target.value as RunParameters["device"])}><option value="auto">Auto</option><option value="cuda">CUDA</option><option value="mps">MPS</option><option value="cpu">CPU</option></select></label>
            <NumberField
              spec={sessionNumericSpecs.decode_threads}
              value={sessionNumbers.decode_threads}
              onChange={(value) => updateSessionNumber("decode_threads", value)}
            />
            <NumberField
              spec={sessionNumericSpecs.jpeg_quality}
              value={sessionNumbers.jpeg_quality}
              onChange={(value) => updateSessionNumber("jpeg_quality", value)}
            />
          </div>
        )}

        {error && <p className="form-error" role="alert">{error}</p>}
        <button className="primary-button" type="submit" disabled={busy}>{busy ? "正在提交…" : "准备视频"}</button>
        {sessionReady && (
          <button className="text-button" type="button" onClick={() => onPreparingNewSessionChange(false)}>
            返回当前视频
          </button>
        )}
      </form>
    );
  }

  if (!sessionReady) {
    return (
      <section className="run-form">
        <div className="section-heading">
          <div>
            <p className="eyebrow">VIDEO SESSION</p>
            <h2>准备视频</h2>
          </div>
          <span className="algorithm-chip">AKS</span>
        </div>
        <div className="session-panel">
          <strong>{currentSession?.original_filename}</strong>
          <span>{currentSession?.stage}</span>
          <div className="large-progress"><i style={{ width: `${Math.round((currentSession?.progress ?? 0) * 100)}%` }} /></div>
          <small>{Math.round((currentSession?.progress ?? 0) * 100)}%</small>
        </div>
      </section>
    );
  }

  return (
    <form className="run-form" onSubmit={submitQuery}>
      <div className="section-heading">
        <div>
          <p className="eyebrow">SESSION QUERY</p>
          <h2>连续提问</h2>
        </div>
        <span className="algorithm-chip">{currentSession.candidate_count} 帧</span>
      </div>

      <div className="session-panel">
        <strong>{currentSession.original_filename}</strong>
        <span>{labelFeatureBackend(typeof currentSession.parameters.feature_backend === "string" ? currentSession.parameters.feature_backend : undefined)} · {sessionModelLabel(currentSession, clipModels)}</span>
        <small>
          候选采样：
          {currentSession.parameters.candidate_sampling === "interval"
            ? `${String(currentSession.parameters.sample_interval ?? "—")} 秒`
            : "原仓库 int(FPS)"}
        </small>
        <button className="text-button" type="button" onClick={() => {
          setQuery("");
          setError("");
          onPreparingNewSessionChange(true);
        }}>
          准备另一个视频
        </button>
      </div>

      <label className="field field-wide">
        <span>Query</span>
        <textarea
          rows={3}
          value={query}
          onChange={(event) => setQuery(event.target.value)}
          placeholder="例如：视频中的人物正在做什么？"
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
        <NumberField
          spec={queryNumericSpecs.max_num_frames}
          value={queryNumbers.max_num_frames}
          onChange={(value) => updateQueryNumber("max_num_frames", value)}
        />
      </div>

      <button className="text-button" type="button" onClick={() => setAdvanced((value) => !value)} aria-expanded={advanced}>
        {advanced ? "收起高级参数" : "展开高级参数"}
      </button>

      {advanced && (
        <div className="field-grid advanced-fields">
          <NumberField
            spec={queryNumericSpecs.threshold}
            value={queryNumbers.threshold}
            onChange={(value) => updateQueryNumber("threshold", value)}
          />
          <NumberField
            spec={queryNumericSpecs.std_threshold}
            value={queryNumbers.std_threshold}
            onChange={(value) => updateQueryNumber("std_threshold", value)}
          />
          <NumberField
            spec={queryNumericSpecs.max_depth}
            value={queryNumbers.max_depth}
            onChange={(value) => updateQueryNumber("max_depth", value)}
          />
          <label className="check-field"><input type="checkbox" checked={parameters.save_uniform_baseline} onChange={(e) => update("save_uniform_baseline", e.target.checked)} /><span>显示同数量均匀抽帧</span></label>
          <label className="check-field"><input type="checkbox" checked={parameters.save_candidate_frames} onChange={(e) => update("save_candidate_frames", e.target.checked)} /><span>显示全部候选帧</span></label>
        </div>
      )}

      {error && <p className="form-error" role="alert">{error}</p>}
      <button className="primary-button" type="submit" disabled={busy}>{busy ? "正在提交…" : "运行 Query 抽帧"}</button>
    </form>
  );
}
