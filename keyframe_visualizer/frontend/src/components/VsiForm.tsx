import { useEffect, useState } from "react";
import type { AlgorithmMetadata, Session, VsiQueryParameters, VsiSessionParameters } from "../types";
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
  onPrepareSession: (video: File, parameters: VsiSessionParameters, subtitle: File | null) => Promise<void>;
  onRunQuery: (query: string, parameters: VsiQueryParameters) => Promise<boolean>;
  onPreparingNewSessionChange: (value: boolean) => void;
}

// Bounds mirror VSIParameters in backend/app/schemas.py.
const sessionNumericSpecs = {
  ocr_fps: {
    label: "OCR 采样 FPS",
    kind: "decimal",
    default: 2,
    min: 0,
    exclusiveMin: true,
    max: 10,
  },
  ocr_crop_top: {
    label: "OCR 截取区域",
    kind: "decimal",
    default: 0.62,
    min: 0,
    max: 1,
    exclusiveMax: true,
  },
} satisfies NumericSpecMap;

const queryNumericSpecs = {
  top_k: { label: "Top-K 关键帧数", kind: "integer", default: 8, min: 1, max: 512 },
  detection_budget: { label: "检测帧预算", kind: "integer", default: 64, min: 1, max: 10000 },
  samples_per_round: { label: "每轮采样数量", kind: "integer", default: 16, min: 1, max: 10000 },
  text_weight: { label: "Text weight", kind: "decimal", default: 0.3, min: 0, max: 1 },
  seed: { label: "随机种子", kind: "integer", default: 0, min: 0 },
} satisfies NumericSpecMap;

const sessionDefaults: VsiSessionParameters = {
  ...numericDefaults(sessionNumericSpecs),
  subtitle_mode: "ocr",
  text_model: "weights/sentence_transformer/paraphrase-multilingual-mpnet-base-v2",
  device: "cpu",
};

const queryDefaults: VsiQueryParameters = {
  ...sessionDefaults,
  ...numericDefaults(queryNumericSpecs),
  objects: [],
  model: "yolov8s-worldv2.pt",
  save_uniform_baseline: true,
  save_candidate_frames: true,
};

export function VsiForm({
  algorithm,
  currentSession,
  preparingNewSession,
  busy,
  onPrepareSession,
  onRunQuery,
  onPreparingNewSessionChange,
}: Props) {
  const [video, setVideo] = useState<File | null>(null);
  const [subtitle, setSubtitle] = useState<File | null>(null);
  const [query, setQuery] = useState("");
  const [objects, setObjects] = useState("");
  const [sessionParameters, setSessionParameters] = useState<VsiSessionParameters>(sessionDefaults);
  const [parameters, setParameters] = useState<VsiQueryParameters>(queryDefaults);
  const [sessionNumbers, setSessionNumbers] = useState(() => numericInputDefaults(sessionNumericSpecs));
  const [queryNumbers, setQueryNumbers] = useState(() => numericInputDefaults(queryNumericSpecs));
  const [advanced, setAdvanced] = useState(false);
  const [error, setError] = useState("");
  const assets = algorithm?.parameter_schema.assets ?? {};
  const assetReady = (key: string) => assets[key]?.ready !== false;

  useEffect(() => {
    if (!currentSession?.parameters) {
      setQuery("");
      setObjects("");
      return;
    }
    const next = { ...sessionDefaults, ...currentSession.parameters } as VsiSessionParameters;
    setSessionParameters(next);
    setParameters((current) => ({ ...current, ...next }));
    setSessionNumbers(numericInputsFrom(sessionNumericSpecs, currentSession.parameters));
    setQuery("");
    setObjects("");
    setError("");
  }, [currentSession?.id]);

  const updateSession = <K extends keyof VsiSessionParameters>(key: K, value: VsiSessionParameters[K]) => {
    setSessionParameters((current) => ({ ...current, [key]: value }));
    setParameters((current) => ({ ...current, [key]: value }));
  };

  const update = <K extends keyof VsiQueryParameters>(key: K, value: VsiQueryParameters[K]) =>
    setParameters((current) => ({ ...current, [key]: value }));

  const updateSessionNumber = (key: string, value: string) =>
    setSessionNumbers((current) => ({ ...current, [key]: value }));

  const updateQueryNumber = (key: string, value: string) =>
    setQueryNumbers((current) => ({ ...current, [key]: value }));

  const prepare = async (event: React.FormEvent) => {
    event.preventDefault();
    if (!video) return setError("请先选择视频。");
    if (sessionParameters.subtitle_mode === "upload" && !subtitle) {
      return setError("请选择 .srt 或 .json 字幕文件。");
    }
    if (sessionParameters.subtitle_mode === "ocr" && !assetReady("easyocr")) {
      return setError("EasyOCR 本地权重未完整下载，请先在 VSI_VideoFraming 目录执行 git lfs pull。");
    }
    if (sessionParameters.subtitle_mode !== "none" && sessionParameters.text_model === sessionDefaults.text_model && !assetReady("text_model")) {
      return setError("默认字幕文本模型未完整下载，请先在 VSI_VideoFraming 目录执行 git lfs pull。");
    }
    // Both OCR boxes are disabled outside OCR mode, so leave them unvalidated.
    const resolved = resolveNumericFields(
      sessionNumericSpecs,
      sessionNumbers,
      sessionParameters.subtitle_mode === "ocr" ? [] : ["ocr_fps", "ocr_crop_top"],
    );
    if (!resolved.ok) return setError(resolved.message);
    setError("");
    await onPrepareSession(video, { ...sessionParameters, ...resolved.values }, subtitle);
    setVideo(null);
    setSubtitle(null);
  };

  const submit = async (event: React.FormEvent) => {
    event.preventDefault();
    if (!currentSession || currentSession.status !== "succeeded") return setError("请先等待视频预处理完成。");
    const parsedObjects = objects.split(",").map((item) => item.trim()).filter(Boolean);
    if (!query.trim()) return setError("请输入 query。");
    if (!parsedObjects.length) return setError("请输入至少一个检测目标，例如 person, car。");
    if (parameters.model === queryDefaults.model && !assetReady("yolo")) {
      return setError("YOLO-World 本地权重未完整下载，请先在 VSI_VideoFraming 目录执行 git lfs pull。");
    }
    if (parameters.model === queryDefaults.model && !assetReady("clip")) {
      return setError("YOLO-World 使用的 CLIP 本地权重未完整下载，请先在 VSI_VideoFraming 目录执行 git lfs pull。");
    }
    const resolved = resolveNumericFields(queryNumericSpecs, queryNumbers);
    if (!resolved.ok) return setError(resolved.message);
    const values = resolved.values;
    if ((values.samples_per_round ?? 0) > (values.detection_budget ?? 0)) {
      return setError("每轮采样数量不能大于检测帧预算。");
    }
    setError("");
    const submitted = await onRunQuery(query.trim(), { ...parameters, ...values, objects: parsedObjects });
    if (submitted) {
      setQuery("");
      setObjects("");
    }
  };

  const preparing = preparingNewSession || !currentSession || currentSession.status === "failed";
  if (preparing) {
    return (
      <form className="run-form" onSubmit={prepare}>
        <div className="section-heading"><div><p className="eyebrow">VIDEO SESSION · VSI</p><h2>准备 VSI 视频</h2></div><span className="algorithm-chip">VSI</span></div>
        {Object.keys(assets).length > 0 && <div className="vsi-assets" aria-label="VSI 本地资源状态">{Object.entries(assets).map(([key, asset]) => <span key={key} className={asset.ready ? "ready" : "missing"}><i />{asset.label}：{asset.ready ? "已就绪" : "未就绪"}</span>)}</div>}
        {currentSession?.status === "failed" && <p className="form-error" role="alert">上一个视频预处理失败：{currentSession.error}</p>}
        <label className="field field-wide"><span>视频文件</span><input type="file" accept="video/*" onChange={(e) => setVideo(e.target.files?.[0] ?? null)} required /><small>{video ? `${video.name} · ${(video.size / 1024 / 1024).toFixed(1)} MB` : "最大 8 GB"}</small></label>
        <div className="field-grid">
          <label className="field"><span>字幕来源</span><select value={sessionParameters.subtitle_mode} onChange={(e) => updateSession("subtitle_mode", e.target.value as VsiSessionParameters["subtitle_mode"])}><option value="ocr">烧录字幕 OCR</option><option value="upload">上传字幕</option><option value="none">不使用字幕</option></select></label>
          <NumberField
            spec={sessionNumericSpecs.ocr_fps}
            value={sessionNumbers.ocr_fps}
            onChange={(value) => updateSessionNumber("ocr_fps", value)}
            disabled={sessionParameters.subtitle_mode !== "ocr"}
          />
          <NumberField
            spec={sessionNumericSpecs.ocr_crop_top}
            value={sessionNumbers.ocr_crop_top}
            onChange={(value) => updateSessionNumber("ocr_crop_top", value)}
            disabled={sessionParameters.subtitle_mode !== "ocr"}
          />
          <label className="field"><span>运行设备</span><select value={sessionParameters.device} onChange={(e) => updateSession("device", e.target.value as VsiSessionParameters["device"])}><option value="cpu">CPU</option><option value="cuda">CUDA</option><option value="mps">MPS</option></select></label>
        </div>
        {sessionParameters.subtitle_mode === "upload" && <label className="field field-wide"><span>字幕文件</span><input type="file" accept=".srt,.json" onChange={(e) => setSubtitle(e.target.files?.[0] ?? null)} required /><small>{subtitle?.name ?? "支持 .srt / .json"}</small></label>}
        <label className="field field-wide"><span>字幕文本模型</span><input value={sessionParameters.text_model} onChange={(e) => updateSession("text_model", e.target.value)} /></label>
        {error && <p className="form-error" role="alert">{error}</p>}
        <button className="primary-button" type="submit" disabled={busy}>{busy ? "正在提交…" : "准备 VSI 视频"}</button>
        {currentSession?.status === "succeeded" && <button className="text-button" type="button" onClick={() => onPreparingNewSessionChange(false)}>返回当前视频</button>}
      </form>
    );
  }

  if (!currentSession || currentSession.status !== "succeeded") {
    return <section className="run-form"><div className="section-heading"><div><p className="eyebrow">VIDEO SESSION · VSI</p><h2>准备 VSI 视频</h2></div><span className="algorithm-chip">VSI</span></div><div className="session-panel"><strong>{currentSession?.original_filename}</strong><span>{currentSession?.stage}</span><div className="large-progress"><i style={{ width: `${Math.round((currentSession?.progress ?? 0) * 100)}%` }} /></div><small>{Math.round((currentSession?.progress ?? 0) * 100)}%</small></div></section>;
  }

  return (
    <form className="run-form" onSubmit={submit}>
      <div className="section-heading"><div><p className="eyebrow">SESSION QUERY · VSI</p><h2>连续提问</h2></div><span className="algorithm-chip">VSI</span></div>
      <div className="session-panel"><strong>{currentSession.original_filename}</strong><span>字幕：{String(currentSession.parameters.subtitle_mode ?? "—")}</span><button className="text-button" type="button" onClick={() => { setQuery(""); setObjects(""); setError(""); onPreparingNewSessionChange(true); }}>准备另一个视频</button></div>
      <label className="field field-wide"><span>Query</span><textarea rows={3} value={query} onChange={(e) => setQuery(e.target.value)} placeholder="例如：视频中的人物正在做什么？" /></label>
      <label className="field field-wide"><span>检测目标 Objects</span><input value={objects} onChange={(e) => setObjects(e.target.value)} placeholder="例如：person, horse, road" /><small>多个目标用英文逗号分隔；每条 query 可以不同。</small></label>
      <div className="field-grid">
        <NumberField
          spec={queryNumericSpecs.top_k}
          value={queryNumbers.top_k}
          onChange={(value) => updateQueryNumber("top_k", value)}
        />
        <NumberField
          spec={queryNumericSpecs.detection_budget}
          value={queryNumbers.detection_budget}
          onChange={(value) => updateQueryNumber("detection_budget", value)}
        />
        <NumberField
          spec={queryNumericSpecs.samples_per_round}
          value={queryNumbers.samples_per_round}
          onChange={(value) => updateQueryNumber("samples_per_round", value)}
        />
        <NumberField
          spec={queryNumericSpecs.text_weight}
          value={queryNumbers.text_weight}
          onChange={(value) => updateQueryNumber("text_weight", value)}
        />
        <NumberField
          spec={queryNumericSpecs.seed}
          value={queryNumbers.seed}
          onChange={(value) => updateQueryNumber("seed", value)}
        />
      </div>
      <button className="text-button" type="button" onClick={() => setAdvanced((value) => !value)}>{advanced ? "收起高级参数" : "展开高级参数"}</button>
      {advanced && <div className="field-grid advanced-fields"><label className="field field-wide"><span>YOLO-World 模型</span><input value={parameters.model} onChange={(e) => update("model", e.target.value)} /></label><label className="check-field"><input type="checkbox" checked={parameters.save_uniform_baseline} onChange={(e) => update("save_uniform_baseline", e.target.checked)} /><span>显示同数量均匀抽帧</span></label><label className="check-field"><input type="checkbox" checked={parameters.save_candidate_frames} onChange={(e) => update("save_candidate_frames", e.target.checked)} /><span>显示 VSI 访问帧</span></label></div>}
      {error && <p className="form-error" role="alert">{error}</p>}
      <button className="primary-button" type="submit" disabled={busy}>{busy ? "正在提交…" : "运行 VSI 抽帧"}</button>
    </form>
  );
}
