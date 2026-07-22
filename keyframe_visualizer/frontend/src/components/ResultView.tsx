import { useRef } from "react";
import { api } from "../api";
import type { Job, Manifest, SelectedFrame } from "../types";
import { ScoreChart } from "./ScoreChart";

interface Props {
  job: Job | null;
  manifest: Manifest | null;
}

function formatTime(seconds: number) {
  const minutes = Math.floor(seconds / 60);
  const remainder = seconds - minutes * 60;
  return `${minutes}:${remainder.toFixed(3).padStart(6, "0")}`;
}

export function ResultView({ job, manifest }: Props) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const seek = (frame: SelectedFrame) => {
    if (videoRef.current) {
      videoRef.current.currentTime = frame.timestamp_seconds;
      void videoRef.current.play();
    }
  };

  if (!job) {
    return <section className="empty-state"><div className="empty-orbit" /><h2>等待一次运行</h2><p>上传视频并设置 query，结果会在这里沿时间线展开。</p></section>;
  }

  if (job.status !== "succeeded" || !manifest) {
    return (
      <section className="job-state">
        <div className={`state-icon state-${job.status}`}><span>{job.status === "failed" ? "!" : Math.round(job.progress * 100)}</span>{job.status !== "failed" && <small>%</small>}</div>
        <p className="eyebrow">{job.algorithm.toUpperCase()} RUN</p>
        <h2>{job.stage}</h2>
        <p>{job.status === "failed" ? job.error : `${job.original_filename} 正在处理，页面会自动更新。`}</p>
        <div className="large-progress"><i style={{ width: `${job.progress * 100}%` }} /></div>
      </section>
    );
  }

  return (
    <section className="results">
      <div className="result-title">
        <div><p className="eyebrow">RESULT · {manifest.algorithm.mode.toUpperCase()}</p><h2>{manifest.video.filename}</h2><p className="result-query">“{manifest.query}”</p></div>
        <a className="secondary-button" href={api.manifestDownloadUrl(job.id)}>下载 Manifest</a>
      </div>

      <div className="summary-grid">
        <div><span>已选帧</span><strong>{manifest.summary.selected_keyframes}</strong><small>/ 请求 {manifest.summary.requested_keyframes}</small></div>
        <div><span>候选帧</span><strong>{manifest.summary.candidate_frames}</strong><small>{manifest.candidate_sampling.interval_seconds ? `请求 ${manifest.candidate_sampling.interval_seconds}s · 实际 ${manifest.candidate_sampling.effective_interval_seconds?.toFixed(6) ?? "—"}s` : "原始采样"}</small></div>
        <div><span>视频时长</span><strong>{formatTime(manifest.video.duration_seconds)}</strong><small>{manifest.video.fps.toFixed(2)} FPS</small></div>
      </div>

      <video ref={videoRef} className="video-player" controls preload="metadata" src={api.videoUrl(job.id)} />
      <ScoreChart candidates={manifest.candidates} />

      <div className="frames-heading"><h3>选中的全部帧</h3><span>点击任意帧跳转到原视频位置</span></div>
      <div className="frame-grid">
        {manifest.selected_frames.map((frame) => (
          <button className="frame-card" key={frame.selected_order} onClick={() => seek(frame)}>
            <div className="frame-image-wrap">
              <img src={api.mediaUrl(job.id, frame.file)} alt={`第 ${frame.selected_order} 个关键帧，时间 ${formatTime(frame.timestamp_seconds)}`} loading="lazy" />
              <span className="frame-order">{String(frame.selected_order).padStart(2, "0")}</span>
              <span className="frame-time">{formatTime(frame.timestamp_seconds)}</span>
            </div>
            <dl>
              <div><dt>原视频帧</dt><dd>{frame.original_frame_index}</dd></div>
              <div><dt>候选序号</dt><dd>#{frame.candidate_order}</dd></div>
              <div><dt>相关性</dt><dd>{frame.relevance_score.toFixed(4)}</dd></div>
              <div><dt>Segment</dt><dd>{frame.segment_id >= 0 ? `${frame.segment_id} · d${frame.segment_depth}` : "—"}</dd></div>
            </dl>
          </button>
        ))}
      </div>
    </section>
  );
}
