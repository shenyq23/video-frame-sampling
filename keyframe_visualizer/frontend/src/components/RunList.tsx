import { useMemo } from "react";
import type { Job, Session } from "../types";

interface Props {
  sessions: Session[];
  jobs: Job[];
  selectedSessionId: string | null;
  onSelect: (session: Session) => void;
}

const labels: Record<Session["status"], string> = {
  queued: "排队中",
  running: "预处理中",
  succeeded: "已准备",
  failed: "失败",
};

export function RunList({ sessions, jobs, selectedSessionId, onSelect }: Props) {
  const queryCounts = useMemo(() => {
    const counts = new Map<string, number>();
    for (const job of jobs) {
      if (!job.session_id) continue;
      counts.set(job.session_id, (counts.get(job.session_id) ?? 0) + 1);
    }
    return counts;
  }, [jobs]);

  return (
    <section className="run-list">
      <div className="section-heading compact-heading">
        <div>
          <p className="eyebrow">VIDEO HISTORY</p>
          <h2>视频记录</h2>
        </div>
        <span className="count-label">{sessions.length}</span>
      </div>
      {sessions.length === 0 ? (
        <p className="empty-copy">还没有视频会话。先准备一个视频，后续 query 会归到这个视频下面。</p>
      ) : (
        <div className="run-items">
          {sessions.map((session) => {
            const queryCount = queryCounts.get(session.id) ?? 0;
            return (
              <button
                key={session.id}
                className={`run-item ${selectedSessionId === session.id ? "selected" : ""}`}
                onClick={() => onSelect(session)}
              >
                <span className="run-item-top">
                  <strong>{session.original_filename}</strong>
                  <span className={`status status-${session.status}`}>{labels[session.status]}</span>
                </span>
                <span className="run-query">
                  {queryCount} 条 query · {session.candidate_count || "—"} 个候选帧
                </span>
                <span className="run-progress">
                  <i style={{ width: `${Math.round(session.progress * 100)}%` }} />
                </span>
                <span className="run-meta">{session.stage} · {Math.round(session.progress * 100)}%</span>
              </button>
            );
          })}
        </div>
      )}
    </section>
  );
}
