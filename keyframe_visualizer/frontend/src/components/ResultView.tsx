import { useEffect, useMemo, useRef, useState } from "react";
import { api } from "../api";
import type {
  ClipModel,
  FrameRecord,
  Job,
  Manifest,
  Session,
  VlmAnswer,
  VlmProfile,
  ParameterSnapshot,
} from "../types";
import { ScoreChart } from "./ScoreChart";

interface Props {
  job: Job | null;
  currentSession: Session | null;
  manifest: Manifest | null;
  clipModels: ClipModel[];
  vlmProfiles: VlmProfile[];
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

function formatDuration(seconds: number | null | undefined) {
  if (seconds === null || seconds === undefined || !Number.isFinite(seconds) || seconds < 0) return "—";
  if (seconds < 10) return `${seconds.toFixed(2)} 秒`;
  if (seconds < 60) return `${seconds.toFixed(1)} 秒`;
  const hours = Math.floor(seconds / 3600);
  const minutes = Math.floor((seconds % 3600) / 60);
  const remainder = Math.floor(seconds % 60);
  return hours > 0
    ? `${hours} 小时 ${minutes} 分 ${remainder} 秒`
    : `${minutes} 分 ${remainder} 秒`;
}

function recordDuration(record: Pick<Job | Session, "created_at" | "updated_at" | "status"> | null) {
  if (!record) return null;
  const started = Date.parse(record.created_at);
  const terminal = record.status === "succeeded" || record.status === "failed";
  const ended = terminal ? Date.parse(record.updated_at) : Date.now();
  if (!Number.isFinite(started) || !Number.isFinite(ended)) return null;
  return Math.max(0, (ended - started) / 1000);
}

function RunTiming({ session, job }: { session: Session | null; job: Job | null }) {
  return (
    <div className="run-timing" aria-label="任务耗时">
      {session && <div><span>预处理耗时</span><strong>{formatDuration(recordDuration(session))}</strong></div>}
      {job && <div><span>抽帧耗时</span><strong>{formatDuration(recordDuration(job))}</strong></div>}
    </div>
  );
}

function ParameterSummary({
  algorithm,
  parameters,
  clipModels,
}: {
  algorithm: string;
  parameters: ParameterSnapshot;
  clipModels: ClipModel[];
}) {
  const uploadedModel = clipModels.find((model) => model.id === parameters.clip_model_id);
  const isMep = parameters.feature_backend === "mep";
  const backendDisplayName = isMep
    ? "86M"
    : typeof parameters.feature_backend === "string"
      ? parameters.feature_backend.toUpperCase()
      : parameters.feature_backend;
  const model =
    parameters.feature_backend === "clip"
      ? uploadedModel?.name ?? parameters.clip_model_id ?? parameters.model_name
      : isMep
        ? "86M-default"
        : parameters.feature_profile;
  const sampling =
    parameters.candidate_sampling === "interval"
      ? `时间间隔 · ${display(parameters.sample_interval)} 秒`
      : "原仓库 int(FPS)";
  const aksItems = [
    ["特征后端", backendDisplayName],
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
  const vsiItems = [
    ["检测目标", Array.isArray(parameters.objects) ? parameters.objects.join(", ") : parameters.objects],
    ["Top-K", parameters.top_k],
    ["检测帧预算", parameters.detection_budget],
    ["每轮采样数量", parameters.samples_per_round],
    ["文本权重", parameters.text_weight],
    ["YOLO-World 模型", parameters.model],
    ["字幕模式", parameters.subtitle_mode],
    ["字幕文本模型", parameters.text_model],
    ["运行设备", parameters.device],
    ["随机种子", parameters.seed],
    ["保存均匀抽帧", parameters.save_uniform_baseline],
    ["保存访问帧", parameters.save_candidate_frames],
  ];
  const items = algorithm === "vsi" ? vsiItems : aksItems;

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
  beforeDelete,
}: {
  job: Job;
  deleting: boolean;
  onDelete: (job: Job) => Promise<void>;
  beforeDelete?: () => void;
}) {
  if (job.status !== "succeeded" && job.status !== "failed") return null;
  const confirmDelete = () => {
    const title = job.owns_video ? "彻底清除任务" : "删除这条 query 记录";
    const body = job.owns_video
      ? "上传视频、所有抽帧图片、manifest、中间结果和任务记录都会被永久删除，且无法恢复。"
      : "只会删除这条 query 的抽帧结果、manifest 和 VLM 回答，不会删除视频预处理缓存。";
    const confirmed = window.confirm(
      `确定${title}“${job.query}”吗？\n\n${body}`,
    );
    if (confirmed) {
      beforeDelete?.();
      void onDelete(job);
    }
  };
  return (
    <button
      className="destructive-button"
      type="button"
      disabled={deleting}
      onClick={confirmDelete}
    >
      {deleting ? "正在删除…" : job.owns_video ? "清除任务及数据" : "删除 query 记录"}
    </button>
  );
}

function normalizeSelectedFrames(manifest: Manifest): FrameRecord[] {
  if (manifest.frame_sets?.selected.frames.length) {
    return manifest.frame_sets.selected.frames;
  }
  return manifest.selected_frames.map((frame) => ({
    ...frame,
    order: frame.selected_order ?? frame.order ?? 0,
    selected_by_aks: manifest.algorithm.id === "aks" ? true : frame.selected_by_aks,
    selected_by_vsi: manifest.algorithm.id === "vsi" ? true : frame.selected_by_vsi,
  }));
}

export function ResultView({ job, currentSession, manifest, clipModels, vlmProfiles, deleting, onDelete }: Props) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const [frameSet, setFrameSet] = useState<FrameSetKey>("selected");
  const [vlmQuery, setVlmQuery] = useState("");
  const [vlmProfile, setVlmProfile] = useState("");
  const [vlmAnswer, setVlmAnswer] = useState<VlmAnswer | null>(null);
  const [vlmBusy, setVlmBusy] = useState(false);
  const [vlmError, setVlmError] = useState("");

  useEffect(() => {
    setFrameSet("selected");
    setVlmQuery(job?.query ?? "");
    setVlmAnswer(null);
    setVlmError("");
  }, [job?.id, job?.query]);

  useEffect(() => {
    const ready = vlmProfiles.find((profile) => profile.enabled && profile.credentials_ready);
    setVlmProfile((current) =>
      vlmProfiles.some((profile) => profile.id === current && profile.enabled)
        ? current
        : ready?.id ?? "",
    );
  }, [vlmProfiles]);

  useEffect(() => {
    let active = true;
    setVlmAnswer(null);
    setVlmError("");
    if (job?.status === "succeeded") {
      api.savedVlmAnswer(job.id, frameSet)
        .then((answer) => {
          if (!active) return;
          setVlmAnswer(answer);
          if (answer) {
            setVlmQuery(answer.query);
            setVlmProfile(answer.profile_id);
          }
        })
        .catch((error: Error) => active && setVlmError(error.message));
    }
    return () => { active = false; };
  }, [job?.id, job?.status, frameSet]);

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

  const releaseMedia = () => {
    const video = videoRef.current;
    if (!video) return;
    video.pause();
    video.removeAttribute("src");
    video.load();
  };

  const askVlm = async () => {
    if (!job || !vlmQuery.trim() || !vlmProfile) return;
    setVlmBusy(true);
    setVlmError("");
    try {
      const answer = await api.createVlmAnswer(
        job.id,
        frameSet,
        vlmQuery.trim(),
        vlmProfile,
      );
      setVlmAnswer(answer);
    } catch (error) {
      setVlmError(error instanceof Error ? error.message : "VLM 问答失败");
    } finally {
      setVlmBusy(false);
    }
  };

  if (!job) {
    if (currentSession?.status === "succeeded") {
      return (
        <section className="empty-state">
          <div className="empty-orbit" />
          <h2>视频已准备完成</h2>
          <p>候选帧和特征已经缓存好了。现在可以在左侧连续输入 query，每次都会直接复用同一个视频会话。</p>
          <p className="standalone-timing">预处理耗时：<strong>{formatDuration(recordDuration(currentSession))}</strong></p>
        </section>
      );
    }
    if (currentSession?.status === "running" || currentSession?.status === "queued") {
      return (
        <section className="job-detail-state">
          <div className="job-state">
            <div className={`state-icon state-${currentSession.status}`}>
              <span>{Math.round(currentSession.progress * 100)}</span>
              <small>%</small>
            </div>
            <p className="eyebrow">VIDEO SESSION</p>
            <h2>{currentSession.stage}</h2>
            <p className="state-query">“{currentSession.original_filename}”</p>
            <p>{currentSession.algorithm === "vsi" ? "视频信息和字幕正在预处理，完成后就可以开始连续提问。" : "候选帧和特征正在缓存，完成后就可以开始连续提问。"}</p>
            <p className="standalone-timing">预处理已用时：<strong>{formatDuration(recordDuration(currentSession))}</strong></p>
            <div className="large-progress"><i style={{ width: `${currentSession.progress * 100}%` }} /></div>
          </div>
        </section>
      );
    }
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
        <ParameterSummary algorithm={job.algorithm} parameters={job.parameters} clipModels={clipModels} />
        <RunTiming session={currentSession} job={job} />
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
    { key: "selected", label: `${job.algorithm.toUpperCase()} 抽出帧` },
    { key: "uniform", label: "同数量均匀抽帧" },
    { key: "candidates", label: job.algorithm === "vsi" ? "VSI 访问帧" : "所有候选帧" },
  ];

  return (
    <section className="results">
      <div className="result-title">
        <div><p className="eyebrow">RESULT · {manifest.algorithm.mode.toUpperCase()}</p><h2>{manifest.video.filename}</h2><p className="result-query">“{manifest.query}”</p></div>
        <div className="result-actions">
          <a className="secondary-button" href={api.manifestDownloadUrl(job.id)}>下载 Manifest</a>
          <DeleteTaskButton
            job={job}
            deleting={deleting}
            onDelete={onDelete}
            beforeDelete={releaseMedia}
          />
        </div>
      </div>

      <ParameterSummary algorithm={job.algorithm} parameters={job.parameters} clipModels={clipModels} />
      <RunTiming session={currentSession} job={job} />

      <div className="summary-grid">
        <div><span>已选帧</span><strong>{manifest.summary.selected_keyframes}</strong><small>/ 请求 {manifest.summary.requested_keyframes}</small></div>
        <div><span>{job.algorithm === "vsi" ? "访问帧" : "候选帧"}</span><strong>{manifest.summary.candidate_frames}</strong><small>{job.algorithm === "vsi" ? "YOLO-World 实际检测" : manifest.candidate_sampling?.interval_seconds ? `请求 ${manifest.candidate_sampling.interval_seconds}s · 实际 ${manifest.candidate_sampling.effective_interval_seconds?.toFixed(6) ?? "—"}s` : "原始采样"}</small></div>
        <div><span>视频时长</span><strong>{formatTime(manifest.video.duration_seconds)}</strong><small>{manifest.video.fps.toFixed(2)} FPS</small></div>
      </div>

      <video ref={videoRef} className="video-player" controls preload="metadata" src={api.videoUrl(job.id)} />
      <ScoreChart candidates={manifest.candidates} algorithm={job.algorithm} />

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
      </div>

      <section className="vlm-panel" aria-labelledby="vlm-title">
        <div className="vlm-heading">
          <div>
            <p className="eyebrow">MULTI-FRAME VLM</p>
            <h3 id="vlm-title">基于当前帧集合回答 Query</h3>
          </div>
          <span>当前使用：{tabs.find((tab) => tab.key === frameSet)?.label}</span>
        </div>
        <div className="vlm-fields">
          <label className="field">
            <span>VLM 服务</span>
            <select
              value={vlmProfile}
              onChange={(event) => setVlmProfile(event.target.value)}
              disabled={vlmBusy}
            >
              {!vlmProfiles.length && <option value="">没有 VLM 配置</option>}
              {vlmProfiles.filter((profile) => profile.enabled).map((profile) => (
                <option
                  key={profile.id}
                  value={profile.id}
                  disabled={!profile.credentials_ready}
                >
                  {profile.name}{profile.credentials_ready ? "" : "（缺少环境变量）"}
                </option>
              ))}
            </select>
          </label>
          <label className="field vlm-query-field">
            <span>VLM Query</span>
            <textarea
              rows={3}
              value={vlmQuery}
              onChange={(event) => setVlmQuery(event.target.value)}
              disabled={vlmBusy}
              placeholder="例如：视频中的人物正在做什么？"
            />
          </label>
        </div>
        <div className="vlm-submit-row">
          <p>
            当前集合共 {activeFrames.length} 帧；服务会按时间顺序读取，并在超过配置上限时均匀缩减。
          </p>
          <button
            className="primary-button vlm-submit"
            type="button"
            disabled={vlmBusy || !vlmProfile || !vlmQuery.trim() || activeFrames.length === 0}
            onClick={() => void askVlm()}
          >
            {vlmBusy ? "VLM 正在分析…" : vlmAnswer ? "重新生成回答" : "生成 VLM 回答"}
          </button>
        </div>
        {vlmError && <p className="vlm-error" role="alert">{vlmError}</p>}
        {vlmAnswer && (
          <article className="vlm-answer" aria-live="polite">
            <div className="vlm-answer-meta">
              <strong>{vlmAnswer.profile_name}</strong>
              <span>
                使用 {vlmAnswer.used_frame_count}/{vlmAnswer.source_frame_count} 帧
                {vlmAnswer.frames_limited ? "（已按时间均匀缩减）" : ""}
              </span>
              <span>VLM 回答耗时：{formatDuration(vlmAnswer.generation_duration_seconds)}</span>
              <time dateTime={vlmAnswer.created_at}>
                {new Date(vlmAnswer.created_at).toLocaleString()}
              </time>
            </div>
            <p className="vlm-answer-query">“{vlmAnswer.query}”</p>
            <div className="vlm-answer-text">{vlmAnswer.answer}</div>
            <div className="vlm-evidence" aria-label="VLM 使用的关键帧">
              {vlmAnswer.used_frames.map((frame, index) => (
                <figure key={`${frame.file}-${index}`}>
                  <img
                    src={api.mediaUrl(job.id, frame.file)}
                    alt={`VLM 证据帧 ${index + 1}，时间 ${frame.timestamp_seconds ?? 0} 秒`}
                    loading="lazy"
                  />
                  <figcaption>#{index + 1} · {formatTime(frame.timestamp_seconds ?? 0)}</figcaption>
                </figure>
              ))}
            </div>
          </article>
        )}
      </section>

      {activeFrames.length ? (
        <div className="frame-grid">
          {activeFrames.map((frame) => (
            <article className="frame-card" key={`${frameSet}-${frame.order}-${frame.original_frame_index}`}>
              <div className="frame-image-wrap">
                <img src={api.mediaUrl(job.id, frame.file)} alt={`${tabs.find((tab) => tab.key === frameSet)?.label}第 ${frame.order} 帧，时间 ${formatTime(frame.timestamp_seconds)}`} loading="lazy" />
                <span className="frame-order">{String(frame.order).padStart(2, "0")}</span>
                <span className="frame-time">{formatTime(frame.timestamp_seconds)}</span>
              </div>
              <dl>
                <div><dt>原视频帧</dt><dd>{frame.original_frame_index}</dd></div>
                <div><dt>{job.algorithm === "vsi" ? "访问序号" : "候选序号"}</dt><dd>{frame.candidate_order ? `#${frame.candidate_order}` : "—"}</dd></div>
                <div><dt>{job.algorithm === "vsi" ? "融合分数" : "相关性"}</dt><dd>{typeof frame.relevance_score === "number" ? frame.relevance_score.toFixed(4) : "—"}</dd></div>
                {job.algorithm === "vsi" ? (
                  <><div><dt>采样概率</dt><dd>{typeof frame.sampling_probability === "number" ? frame.sampling_probability.toFixed(6) : "—"}</dd></div><div><dt>访问顺序</dt><dd>{frame.visited_order ?? "—"}</dd></div></>
                ) : frameSet === "selected" ? (
                  <>
                    {/* Temporarily hidden from the AKS selected-frame cards.
                    <div><dt>Segment</dt><dd>{typeof frame.segment_id === "number" && frame.segment_id >= 0 ? `${frame.segment_id} · d${frame.segment_depth}` : "—"}</dd></div>
                    */}
                  </>
                ) : (
                  <div><dt>被 AKS 选中</dt><dd>{frame.selected_by_aks ? "是" : "否"}</dd></div>
                )}
              </dl>
            </article>
          ))}
        </div>
      ) : (
        <p className="frame-set-empty">该任务没有保存这组帧。旧任务需要重新运行后才能查看。</p>
      )}
    </section>
  );
}
