import { useEffect, useMemo, useRef, useState } from "react";
import { api } from "../api";
import type {
  ClipModel,
  FrameRecord,
  Job,
  Manifest,
  RunParameters,
} from "../types";
import { ScoreChart } from "./ScoreChart";

interface Props {
  job: Job | null;
  manifest: Manifest | null;
  clipModels: ClipModel[];
  deleting: boolean;
  onDelete: (job: Job) => Promise<void>;
}

type FrameSetKey = "selected" | "uniform" | "candidates";

function formatTime(seconds: number) {
  const minutes = Math.floor(seconds / 60);
  const remainder = seconds - minutes * 60;
  return `${minutes}:${remainder.toFixed(3).padStart(6, "0")}`;
}

function display(value: unknown, fallback = "—") {
  if (value === null || value === undefined || value === "") return fallback;
  if (typeof value === "boolean") return value ? "是" : "否";
  return String(value);
}

function ParameterSummary({
  parameters,
  clipModels,
}: {
  parameters: Partial<RunParameters>;
  clipModels: ClipModel[];
}) {
  const uploadedModel = clipModels.find((model) => model.id === parameters.clip_model_id);
  const model =
    parameters.feature_backend === "clip"
      ? uploadedModel?.name ?? parameters.clip_model_id ?? parameters.model_name
      : parameters.feature_profile;
  const sampling =
    parameters.candidate_sampling === "interval"
      ? `时间间隔 · ${display(parameters.sample_interval)} 秒`
      : "原仓库 int(FPS)";
  const items = [
    ["特征后端", parameters.feature_backend?.toUpperCase()],
    ["特征模型 / 服务", model],
    ["AKS 模式", parameters.aks_mode],
    ["目标帧数", parameters.max_num_frames],
    ["候选采样", sampling],
    ["t1 threshold", parameters.threshold],
    ["t2 std threshold", parameters.std_threshold],
    ["最大深度", parameters.max_depth],
    ["Batch size", parameters.batch_size],
    ["运行设备", parameters.device],
    ["解码线程", parameters.decode_threads],
    ["JPEG 质量", parameters.jpeg_quality],
    ["保存均匀抽帧", parameters.save_uniform_baseline],
    ["保存候选帧", parameters.save_candidate_frames],
  ];

  return (
    <section className="parameter-panel" aria-labelledby="parameter-title">
      <div className="parameter-heading">
        <h3 id="parameter-title">本次运行参数</h3>
        <span>任务创建时的参数快照</span>
      </div>
      <dl className="parameter-grid">
        {items.map(([label, value]) => (
          <div key={String(label)}>
            <dt>{label}</dt>
            <dd>{display(value)}</dd>
          </div>
        ))}
      </dl>
    </section>
  );
}

function DeleteTaskButton({
  job,
  deleting,
  onDelete,
}: {
  job: Job;
  deleting: boolean;
  onDelete: (job: Job) => Promise<void>;
}) {
  if (job.status !== "succeeded" && job.status !== "failed") return null;
  const confirmDelete = () => {
    const confirmed = window.confirm(
      `确定彻底清除任务“${job.original_filename}”吗？\n\n` +
        "上传视频、所有抽帧图片、manifest、中间结果和任务记录都会被永久删除，且无法恢复。",
    );
    if (confirmed) void onDelete(job);
  };
  return (
    <button
      className="destructive-button"
      type="button"
      disabled={deleting}
      onClick={confirmDelete}
    >
      {deleting ? "正在清除…" : "清除任务及数据"}
    </button>
  );
}

function normalizeSelectedFrames(manifest: Manifest): FrameRecord[] {
  if (manifest.frame_sets?.selected.frames.length) {
    return manifest.frame_sets.selected.frames;
  }
  return manifest.selected_frames.map((frame) => ({
    ...frame,
    order: frame.selected_order,
    selected_by_aks: true,
  }));
}

export function ResultView({ job, manifest, clipModels, deleting, onDelete }: Props) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const [frameSet, setFrameSet] = useState<FrameSetKey>("selected");

  useEffect(() => setFrameSet("selected"), [job?.id]);

  useEffect(() => {
    const video = videoRef.current;
    return () => {
      if (!video) return;
      video.pause();
      video.removeAttribute("src");
      video.load();
    };
  }, [job?.id, manifest?.run_id]);

  const frameSets = useMemo(() => {
    if (!manifest) return null;
    const selected = normalizeSelectedFrames(manifest);
    const uniform = manifest.frame_sets?.uniform.frames ?? manifest.uniform_frames ?? [];
    const candidates: FrameRecord[] =
      manifest.frame_sets?.candidates.frames ??
      manifest.candidates
        .filter((frame): frame is typeof frame & { file: string; order: number } =>
          Boolean(frame.file && frame.order),
        )
        .map((frame): FrameRecord => ({
          ...frame,
          file: frame.file,
          order: frame.order,
          selected_by_aks: frame.selected_by_aks ?? frame.selected,
        }));
    return {
      selected: { available: true, frames: selected },
      uniform: {
        available: manifest.frame_sets?.uniform.available ?? uniform.length > 0,
        frames: uniform,
      },
      candidates: {
        available: manifest.frame_sets?.candidates.available ?? candidates.length > 0,
        frames: candidates,
      },
    };
  }, [manifest]);

  const seek = (frame: FrameRecord) => {
    const video = videoRef.current;
    if (!video) return;

    const seekAndPlay = () => {
      const upperBound = Number.isFinite(video.duration)
        ? Math.max(0, video.duration - 0.001)
        : frame.timestamp_seconds;
      const target = Math.max(0, Math.min(frame.timestamp_seconds, upperBound));
      video.pause();
      video.currentTime = target;
      const playAfterSeek = () => {
        void video.play().catch(() => undefined);
      };
      if (Math.abs(video.currentTime - target) < 0.01) {
        playAfterSeek();
      } else {
        video.addEventListener("seeked", playAfterSeek, { once: true });
      }
      video.scrollIntoView({ behavior: "smooth", block: "center" });
    };

    if (video.readyState >= HTMLMediaElement.HAVE_METADATA) {
      seekAndPlay();
    } else {
      video.addEventListener("loadedmetadata", seekAndPlay, { once: true });
      video.load();
    }
  };

  if (!job) {
    return <section className="empty-state"><div className="empty-orbit" /><h2>等待一次运行</h2><p>上传视频并设置 query，结果会在这里沿时间线展开。</p></section>;
  }

  if (job.status !== "succeeded" || !manifest) {
    return (
      <section className="job-detail-state">
        <div className="job-state">
          <div className={`state-icon state-${job.status}`}><span>{job.status === "failed" ? "!" : Math.round(job.progress * 100)}</span>{job.status !== "failed" && <small>%</small>}</div>
          <p className="eyebrow">{job.algorithm.toUpperCase()} RUN</p>
          <h2>{job.stage}</h2>
          <p className="state-query">“{job.query}”</p>
          <p>{job.status === "failed" ? job.error : `${job.original_filename} 正在处理，页面会自动更新。`}</p>
          <div className="large-progress"><i style={{ width: `${job.progress * 100}%` }} /></div>
        </div>
        <ParameterSummary parameters={job.parameters} clipModels={clipModels} />
        {(job.status === "failed" || job.status === "succeeded") && (
          <div className="delete-row">
            <DeleteTaskButton job={job} deleting={deleting} onDelete={onDelete} />
          </div>
        )}
      </section>
    );
  }

  const activeFrames = frameSets?.[frameSet].frames ?? [];
  const tabs: Array<{ key: FrameSetKey; label: string }> = [
    { key: "selected", label: "AKS 抽出帧" },
    { key: "uniform", label: "同数量均匀抽帧" },
    { key: "candidates", label: "所有候选帧" },
  ];

  return (
    <section className="results">
      <div className="result-title">
        <div><p className="eyebrow">RESULT · {manifest.algorithm.mode.toUpperCase()}</p><h2>{manifest.video.filename}</h2><p className="result-query">“{manifest.query}”</p></div>
        <div className="result-actions">
          <a className="secondary-button" href={api.manifestDownloadUrl(job.id)}>下载 Manifest</a>
          <DeleteTaskButton job={job} deleting={deleting} onDelete={onDelete} />
        </div>
      </div>

      <ParameterSummary parameters={job.parameters} clipModels={clipModels} />

      <div className="summary-grid">
        <div><span>已选帧</span><strong>{manifest.summary.selected_keyframes}</strong><small>/ 请求 {manifest.summary.requested_keyframes}</small></div>
        <div><span>候选帧</span><strong>{manifest.summary.candidate_frames}</strong><small>{manifest.candidate_sampling.interval_seconds ? `请求 ${manifest.candidate_sampling.interval_seconds}s · 实际 ${manifest.candidate_sampling.effective_interval_seconds?.toFixed(6) ?? "—"}s` : "原始采样"}</small></div>
        <div><span>视频时长</span><strong>{formatTime(manifest.video.duration_seconds)}</strong><small>{manifest.video.fps.toFixed(2)} FPS</small></div>
      </div>

      <video ref={videoRef} className="video-player" controls preload="metadata" src={api.videoUrl(job.id)} />
      <ScoreChart candidates={manifest.candidates} />

      <div className="frame-set-toolbar">
        <div className="frame-tabs" role="group" aria-label="切换帧集合">
          {tabs.map((tab) => {
            const entry = frameSets?.[tab.key];
            const available = Boolean(entry?.available);
            return (
              <button
                key={tab.key}
                type="button"
                className={frameSet === tab.key ? "active" : ""}
                aria-pressed={frameSet === tab.key}
                disabled={!available}
                onClick={() => setFrameSet(tab.key)}
              >
                {tab.label}<span>{entry?.frames.length ?? 0}</span>
              </button>
            );
          })}
        </div>
        <span>点击任意帧跳转到原视频位置</span>
      </div>

      {activeFrames.length ? (
        <div className="frame-grid">
          {activeFrames.map((frame) => (
            <button className="frame-card" key={`${frameSet}-${frame.order}-${frame.original_frame_index}`} onClick={() => seek(frame)}>
              <div className="frame-image-wrap">
                <img src={api.mediaUrl(job.id, frame.file)} alt={`${tabs.find((tab) => tab.key === frameSet)?.label}第 ${frame.order} 帧，时间 ${formatTime(frame.timestamp_seconds)}`} loading="lazy" />
                <span className="frame-order">{String(frame.order).padStart(2, "0")}</span>
                <span className="frame-time">{formatTime(frame.timestamp_seconds)}</span>
              </div>
              <dl>
                <div><dt>原视频帧</dt><dd>{frame.original_frame_index}</dd></div>
                <div><dt>候选序号</dt><dd>#{frame.candidate_order}</dd></div>
                <div><dt>相关性</dt><dd>{frame.relevance_score.toFixed(4)}</dd></div>
                {frameSet === "selected" ? (
                  <div><dt>Segment</dt><dd>{typeof frame.segment_id === "number" && frame.segment_id >= 0 ? `${frame.segment_id} · d${frame.segment_depth}` : "—"}</dd></div>
                ) : (
                  <div><dt>被 AKS 选中</dt><dd>{frame.selected_by_aks ? "是" : "否"}</dd></div>
                )}
              </dl>
            </button>
          ))}
        </div>
      ) : (
        <p className="frame-set-empty">该任务没有保存这组帧。旧任务需要重新运行后才能查看。</p>
      )}
    </section>
  );
}
