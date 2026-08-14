import { useEffect, useState } from "react";
import type {
  AlgorithmMetadata,
  SageQueryParameters,
  SageSessionParameters,
  Session,
} from "../types";
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
  currentSession: Session | null;
  preparingNewSession: boolean;
  busy: boolean;
  onPrepareSession: (
    video: File,
    parameters: SageQueryParameters,
    asr: File | null,
  ) => Promise<void>;
  onRunQuery: (query: string, parameters: SageQueryParameters) => Promise<boolean>;
  onPreparingNewSessionChange: (value: boolean) => void;
  vlmReady: boolean;
}

const queryNumericSpecs = {
  budget: { label: "关键帧预算", kind: "integer", default: 8, min: 1, max: 512 },
} satisfies NumericSpecMap;

const sessionDefaults: SageSessionParameters = {
  asr_mode: "remote",
  device: "cpu",
};

const queryDefaults: SageQueryParameters = {
  ...sessionDefaults,
  ...numericDefaults(queryNumericSpecs),
  save_uniform_baseline: true,
  save_candidate_frames: true,
};

const asrModeLabel = (mode: SageSessionParameters["asr_mode"]) => {
  if (mode === "remote") return "远程 ASR";
  if (mode === "upload") return "上传 JSON";
  return "纯视觉";
};

export function SageForm({
  algorithm,
  currentSession,
  preparingNewSession,
  busy,
  onPrepareSession,
  onRunQuery,
  onPreparingNewSessionChange,
  vlmReady,
}: Props) {
  const [video, setVideo] = useState<File | null>(null);
  const [asr, setAsr] = useState<File | null>(null);
  const [query, setQuery] = useState("");
  const [sessionParameters, setSessionParameters] =
    useState<SageSessionParameters>(sessionDefaults);
  const [parameters, setParameters] = useState<SageQueryParameters>(queryDefaults);
  const [queryNumbers, setQueryNumbers] = useState(() => numericInputDefaults(queryNumericSpecs));
  const [advanced, setAdvanced] = useState(false);
  const [error, setError] = useState("");
  const assets = algorithm?.parameter_schema.assets ?? {};
  const assetReady = (key: string) => assets[key]?.ready !== false;

  useEffect(() => {
    if (!currentSession?.parameters) {
      setQuery("");
      return;
    }
    const nextSession = {
      ...sessionDefaults,
      ...currentSession.parameters,
    } as SageSessionParameters;
    setSessionParameters(nextSession);
    setParameters((current) => ({ ...current, ...currentSession.parameters, ...nextSession }));
    setQueryNumbers(numericInputsFrom(queryNumericSpecs, currentSession.parameters));
    setQuery("");
    setError("");
  }, [currentSession?.id]);

  const updateSession = <K extends keyof SageSessionParameters>(
    key: K,
    value: SageSessionParameters[K],
  ) => {
    setSessionParameters((current) => ({ ...current, [key]: value }));
    setParameters((current) => ({ ...current, [key]: value }));
    if (key === "asr_mode" && value !== "upload") setAsr(null);
  };

  const update = <K extends keyof SageQueryParameters>(key: K, value: SageQueryParameters[K]) =>
    setParameters((current) => ({ ...current, [key]: value }));

  const prepare = async (event: React.FormEvent) => {
    event.preventDefault();
    if (!video) return setError("请先选择视频。");
    if (sessionParameters.asr_mode === "remote" && !assetReady("remote_asr")) {
      return setError("后端尚未配置 SAGE_ASR_BASE_URL 或 SAGE_ASR_TOKEN。");
    }
    if (sessionParameters.asr_mode === "upload" && !asr) {
      return setError("请选择 SAGE 可读取的 ASR JSON 文件。");
    }
    setError("");
    await onPrepareSession(video, { ...parameters, ...sessionParameters }, asr);
    setVideo(null);
    setAsr(null);
  };

  const submit = async (event: React.FormEvent) => {
    event.preventDefault();
    if (!currentSession || currentSession.status !== "succeeded") {
      return setError("请先等待视频和 ASR 预处理完成。");
    }
    if (!query.trim()) return setError("请输入 query。");
    if (!assetReady("clip")) return setError("SAGE CLIP 模型不存在或未完整复制。");
    if (sessionParameters.asr_mode !== "none" && !assetReady("text_model")) {
      return setError("SAGE 字幕文本模型不存在或未完整复制。");
    }
    const resolved = resolveNumericFields(queryNumericSpecs, queryNumbers);
    if (!resolved.ok) return setError(resolved.message);
    setError("");
    const submitted = await onRunQuery(query.trim(), {
      ...parameters,
      ...resolved.values,
      asr_mode: currentSession.parameters.asr_mode as SageSessionParameters["asr_mode"],
      device: currentSession.parameters.device as SageSessionParameters["device"],
    });
    if (submitted) setQuery("");
  };

  const preparing = preparingNewSession || !currentSession || currentSession.status === "failed";
  if (preparing) {
    return (
      <form className="run-form" onSubmit={prepare}>
        <div className="section-heading"><div><p className="eyebrow">VIDEO SESSION · SAGE</p><h2>准备 SAGE 视频</h2></div><span className="algorithm-chip">SAGE</span></div>
        {Object.keys(assets).length > 0 && <div className="vsi-assets" aria-label="SAGE 资源状态">{Object.entries(assets).map(([key, asset]) => <span key={key} className={asset.ready ? "ready" : "missing"}><i />{asset.label}：{asset.ready ? "已就绪" : "未就绪"}</span>)}</div>}
        {currentSession?.status === "failed" && <p className="form-error" role="alert">上一个视频预处理失败：{currentSession.error}</p>}
        <label className="field field-wide"><span>视频文件</span><input type="file" accept="video/*" onChange={(event) => setVideo(event.target.files?.[0] ?? null)} required /><small>{video ? `${video.name} · ${(video.size / 1024 / 1024).toFixed(1)} MB` : "最大 8 GB"}</small></label>
        <div className="field-grid">
          <label className="field"><span>ASR 来源</span><select value={sessionParameters.asr_mode} onChange={(event) => updateSession("asr_mode", event.target.value as SageSessionParameters["asr_mode"])}><option value="remote">远程 ASR</option><option value="upload">上传现有 JSON</option><option value="none">不使用 ASR</option></select></label>
          <label className="field"><span>运行设备</span><select value={sessionParameters.device} onChange={(event) => updateSession("device", event.target.value as SageSessionParameters["device"])}><option value="cpu">CPU</option><option value="cuda">CUDA</option><option value="mps">MPS</option></select></label>
        </div>
        {sessionParameters.asr_mode === "upload" && <label className="field field-wide"><span>ASR JSON</span><input type="file" accept=".json,application/json" onChange={(event) => setAsr(event.target.files?.[0] ?? null)} required /><small>{asr?.name ?? "选择由 SAGE load_asr_json 支持的 JSON"}</small></label>}
        {error && <p className="form-error" role="alert">{error}</p>}
        <button className="primary-button" type="submit" disabled={busy}>{busy ? "正在提交…" : "准备 SAGE 视频"}</button>
        {currentSession?.status === "succeeded" && <button className="text-button" type="button" onClick={() => onPreparingNewSessionChange(false)}>返回当前视频</button>}
      </form>
    );
  }

  if (!currentSession || currentSession.status !== "succeeded") {
    return <section className="run-form"><div className="section-heading"><div><p className="eyebrow">VIDEO SESSION · SAGE</p><h2>准备 SAGE 视频</h2></div><span className="algorithm-chip">SAGE</span></div><div className="session-panel"><strong>{currentSession?.original_filename}</strong><span>{currentSession?.stage}</span><div className="large-progress"><i style={{ width: `${Math.round((currentSession?.progress ?? 0) * 100)}%` }} /></div><small>{Math.round((currentSession?.progress ?? 0) * 100)}%</small></div></section>;
  }

  return (
    <form className="run-form" onSubmit={submit}>
      <div className="section-heading"><div><p className="eyebrow">SESSION QUERY · SAGE</p><h2>连续提问</h2></div><span className="algorithm-chip">SAGE</span></div>
      <div className="session-panel"><strong>{currentSession.original_filename}</strong><span>ASR：{asrModeLabel(currentSession.parameters.asr_mode as SageSessionParameters["asr_mode"])}</span><span>设备：{String(currentSession.parameters.device ?? "cpu").toUpperCase()}</span><button className="text-button" type="button" onClick={() => { setQuery(""); setError(""); onPreparingNewSessionChange(true); }}>准备另一个视频</button></div>
      <label className="field field-wide"><span>Query</span><textarea rows={3} value={query} onChange={(event) => setQuery(event.target.value)} placeholder="例如：找到有人举起奖杯的时刻" /></label>
      <div className="field-grid"><NumberField spec={queryNumericSpecs.budget} value={queryNumbers.budget} onChange={(value) => setQueryNumbers((current) => ({ ...current, budget: value }))} /></div>
      <button className="text-button" type="button" onClick={() => setAdvanced((value) => !value)}>{advanced ? "收起高级参数" : "展开高级参数"}</button>
      {advanced && <div className="field-grid advanced-fields"><label className="check-field"><input type="checkbox" checked={parameters.save_uniform_baseline} onChange={(event) => update("save_uniform_baseline", event.target.checked)} /><span>显示同数量均匀抽帧</span></label><label className="check-field"><input type="checkbox" checked={parameters.save_candidate_frames} onChange={(event) => update("save_candidate_frames", event.target.checked)} /><span>显示 SAGE 候选帧</span></label></div>}
      {error && <p className="form-error" role="alert">{error}</p>}
      <button className="primary-button" type="submit" disabled={busy}>{busy ? "正在提交…" : vlmReady ? "抽帧并生成 VLM 回答" : "运行 SAGE 抽帧"}</button>
      {vlmReady && <p className="submit-hint">抽帧完成后会自动把抽出帧交给 VLM 回答同一条 Query。</p>}
    </form>
  );
}
