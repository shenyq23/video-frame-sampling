import type { CandidateFrame } from "../types";

export function ScoreChart({ candidates }: { candidates: CandidateFrame[] }) {
  if (candidates.length < 2) return null;
  const width = 1000;
  const height = 190;
  const padX = 28;
  const padY = 20;
  const maxTime = candidates[candidates.length - 1].timestamp_seconds || 1;
  const x = (value: number) => padX + (value / maxTime) * (width - padX * 2);
  const y = (value: number) => height - padY - value * (height - padY * 2);
  const points = candidates.map((frame) => `${x(frame.timestamp_seconds)},${y(frame.normalized_score)}`).join(" ");

  return (
    <div className="chart-wrap">
      <div className="chart-heading"><h3>候选帧相关性</h3><span>归一化分数 · 横轴为视频时间</span></div>
      <svg className="score-chart" viewBox={`0 0 ${width} ${height}`} role="img" aria-label="候选帧相关性曲线，选中的帧以圆点标记">
        {[0, 0.5, 1].map((tick) => <line key={tick} x1={padX} x2={width - padX} y1={y(tick)} y2={y(tick)} className="grid-line" />)}
        <polyline points={points} className="score-line" />
        {candidates.filter((frame) => frame.selected).map((frame) => (
          <circle key={frame.candidate_index} cx={x(frame.timestamp_seconds)} cy={y(frame.normalized_score)} r="6" className="selected-dot">
            <title>{`${frame.timestamp_seconds.toFixed(2)}s · ${frame.relevance_score.toFixed(4)}`}</title>
          </circle>
        ))}
        <text x={padX} y={height - 2} className="axis-label">0s</text>
        <text x={width - padX} y={height - 2} textAnchor="end" className="axis-label">{maxTime.toFixed(1)}s</text>
      </svg>
    </div>
  );
}

