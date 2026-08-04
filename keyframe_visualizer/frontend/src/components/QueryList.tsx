import type { Job, Session } from "../types";

interface Props {
  session: Session | null;
  jobs: Job[];
  selectedJobId: string | null;
  deletingJobId: string | null;
  deletingSessionId: string | null;
  hasActiveJobs: boolean;
  onSelect: (job: Job) => void;
  onDeleteJob: (job: Job) => Promise<void>;
  onDeleteSession: (session: Session) => Promise<void>;
}

const labels: Record<Job["status"], string> = {
  queued: "排队中",
  running: "运行中",
  succeeded: "已完成",
  failed: "失败",
};

export function QueryList({
  session,
  jobs,
  selectedJobId,
  deletingJobId,
  deletingSessionId,
  hasActiveJobs,
  onSelect,
  onDeleteJob,
  onDeleteSession,
}: Props) {
  const confirmDeleteSession = () => {
    if (!session) return;
    const confirmed = window.confirm(
      `确定彻底删除视频会话“${session.original_filename}”吗？\n\n` +
        "这会删除该视频的预处理缓存、源视频、所有 query 任务、manifest 和 VLM 回答，且无法恢复。",
    );
    if (confirmed) void onDeleteSession(session);
  };

  const confirmDeleteJob = (event: React.MouseEvent, job: Job) => {
    event.stopPropagation();
    const confirmed = window.confirm(
      `确定删除这条 query 记录吗？\n\n“${job.query}”\n\n` +
        "只会删除这条 query 的抽帧结果、manifest 和 VLM 回答，不会删除视频预处理缓存。",
    );
    if (confirmed) void onDeleteJob(job);
  };

  if (!session) {
    return (
      <aside className="query-panel">
        <div className="query-panel-heading">
          <p className="eyebrow">QUERY HISTORY</p>
          <h2>选择一个视频</h2>
        </div>
        <p className="empty-copy">左侧选择视频后，这里会显示该视频下的所有 query。</p>
      </aside>
    );
  }

  return (
    <aside className="query-panel">
      <div className="query-panel-heading">
        <div>
          <p className="eyebrow">QUERY HISTORY</p>
          <h2>{session.original_filename}</h2>
        </div>
        <span className="count-label">{jobs.length}</span>
      </div>
      <div className="query-panel-actions">
        <button
          className="destructive-button"
          type="button"
          disabled={
            deletingSessionId === session.id ||
            hasActiveJobs ||
            session.status === "queued" ||
            session.status === "running"
          }
          onClick={confirmDeleteSession}
        >
          {deletingSessionId === session.id ? "正在删除…" : "删除整个视频会话"}
        </button>
      </div>

      {jobs.length === 0 ? (
        <p className="empty-copy">这个视频下还没有 query。视频预处理会保留，可以继续在左侧输入 query。</p>
      ) : (
        <div className="query-items">
          {jobs.map((job) => (
            <article
              key={job.id}
              className={`query-item ${selectedJobId === job.id ? "selected" : ""}`}
              onClick={() => onSelect(job)}
              onKeyDown={(event) => {
                if (event.key === "Enter" || event.key === " ") onSelect(job);
              }}
              role="button"
              tabIndex={0}
            >
              <span className="run-item-top">
                <strong>{job.query}</strong>
                <span className={`status status-${job.status}`}>{labels[job.status]}</span>
              </span>
              <span className="run-meta">
                {job.stage} · {Math.round(job.progress * 100)}% · {String(job.parameters.max_num_frames ?? job.parameters.top_k ?? "—")} 帧
              </span>
              <span className="run-progress">
                <i style={{ width: `${Math.round(job.progress * 100)}%` }} />
              </span>
              {(job.status === "succeeded" || job.status === "failed") && (
                <span className="query-item-actions">
                  <button
                    className="text-danger-button"
                    type="button"
                    disabled={deletingJobId === job.id}
                    onClick={(event) => confirmDeleteJob(event, job)}
                  >
                    {deletingJobId === job.id ? "删除中…" : "删除 query"}
                  </button>
                </span>
              )}
            </article>
          ))}
        </div>
      )}
    </aside>
  );
}
