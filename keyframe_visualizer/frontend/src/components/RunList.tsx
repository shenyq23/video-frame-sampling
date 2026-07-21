import type { Job } from "../types";

interface Props {
  jobs: Job[];
  selectedId: string | null;
  onSelect: (job: Job) => void;
}

const labels: Record<Job["status"], string> = {
  queued: "排队中",
  running: "运行中",
  succeeded: "已完成",
  failed: "失败",
};

export function RunList({ jobs, selectedId, onSelect }: Props) {
  return (
    <section className="run-list">
      <div className="section-heading compact-heading">
        <div>
          <p className="eyebrow">RUN HISTORY</p>
          <h2>运行记录</h2>
        </div>
        <span className="count-label">{jobs.length}</span>
      </div>
      {jobs.length === 0 ? (
        <p className="empty-copy">还没有任务。提交第一个视频后，运行记录会保存在这里。</p>
      ) : (
        <div className="run-items">
          {jobs.map((job) => (
            <button key={job.id} className={`run-item ${selectedId === job.id ? "selected" : ""}`} onClick={() => onSelect(job)}>
              <span className="run-item-top"><strong>{job.original_filename}</strong><span className={`status status-${job.status}`}>{labels[job.status]}</span></span>
              <span className="run-query">{job.query}</span>
              <span className="run-progress"><i style={{ width: `${Math.round(job.progress * 100)}%` }} /></span>
              <span className="run-meta">{job.stage} · {Math.round(job.progress * 100)}%</span>
            </button>
          ))}
        </div>
      )}
    </section>
  );
}

