import type { CandidateFrame } from "../types";

export function ScoreChart({ candidates, algorithm = "aks" }: { candidates: CandidateFrame[]; algorithm?: string }) {
  if (candidates.length < 2) return null;
  const width = 1000;
  const height = 190;
  const padX = 28;
  const padY = 20;
  const maxTime = candidates[candidates.length - 1].timestamp_seconds || 1;
  const x = (value: number) => padX + (value / maxTime) * (width - padX * 2);
  const values = algorithm === "vsi"
    ? candidates.flatMap((frame) => [frame.fused_score ?? 0, frame.sampling_probability ?? 0])
    : candidates.map((frame) => frame.normalized_score ?? 0);
  const minValue = algorithm === "vsi" ? Math.min(0, ...values) : 0;
  const maxValue = Math.max(1e-6, ...values);
  const y = (value: number) => height - padY - ((value - minValue) / (maxValue - minValue || 1)) * (height - padY * 2);
  const points = candidates.map((frame) => `${x(frame.timestamp_seconds)},${y(frame.normalized_score ?? 0)}`).join(" ");
  const line = (field: "fused_score" | "sampling_probability") => candidates.map((frame) => `${x(frame.timestamp_seconds)},${y(frame[field] ?? 0)}`).join(" ");

  return (
    <div className="chart-wrap">
      <div className="chart-heading"><h3>{algorithm === "vsi" ? "VSI 访问帧分数" : "候选帧相关性"}</h3><span>{algorithm === "vsi" ? "融合分数 / 最终采样概率 · 横轴为视频时间" : "归一化分数 · 横轴为视频时间"}</span></div>
      <svg className="score-chart" viewBox={`0 0 ${width} ${height}`} role="img" aria-label="候选帧相关性曲线，选中的帧以圆点标记">
        {[minValue, (minValue + maxValue) / 2, maxValue].map((tick) => <line key={tick} x1={padX} x2={width - padX} y1={y(tick)} y2={y(tick)} className="grid-line" />)}
        {algorithm === "vsi" ? <><polyline points={line("fused_score")} className="score-line vsi-fused-line" /><polyline points={line("sampling_probability")} className="score-line vsi-probability-line" /></> : <polyline points={points} className="score-line" />}
        {candidates.filter((frame) => frame.selected).map((frame) => (
          <circle key={frame.candidate_index} cx={x(frame.timestamp_seconds)} cy={y(algorithm === "vsi" ? frame.fused_score ?? 0 : frame.normalized_score ?? 0)} r="6" className="selected-dot">
            <title>{`${frame.timestamp_seconds.toFixed(2)}s · ${(frame.relevance_score ?? 0).toFixed(4)}`}</title>
          </circle>
        ))}
        <text x={padX} y={height - 2} className="axis-label">0s</text>
        <text x={width - padX} y={height - 2} textAnchor="end" className="axis-label">{maxTime.toFixed(1)}s</text>
      </svg>
    </div>
  );
}
