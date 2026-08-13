import type { CandidateFrame } from "../types";

export function ScoreChart({ candidates, algorithm = "aks" }: { candidates: CandidateFrame[]; algorithm?: string }) {
  if (candidates.length < 2) return null;
  const width = 1000;
  const height = 190;
  const padX = 28;
  const padY = 20;
  const maxTime = candidates[candidates.length - 1].timestamp_seconds || 1;
  const x = (value: number) => padX + (value / maxTime) * (width - padX * 2);

  if (algorithm === "vsi") {
    const renderVsiChart = (
      field: "fused_score" | "sampling_probability",
      title: string,
      description: string,
    ) => {
      const values = candidates.map((frame) => frame[field] ?? 0);
      const minValue = field === "fused_score" ? Math.min(0, ...values) : 0;
      const maxValue = Math.max(1e-6, ...values);
      const y = (value: number) => height - padY - ((value - minValue) / (maxValue - minValue || 1)) * (height - padY * 2);
      const points = candidates
        .map((frame) => `${x(frame.timestamp_seconds)},${y(frame[field] ?? 0)}`)
        .join(" ");

      return (
        <section className="vsi-score-plot" key={field}>
          <div className="vsi-score-heading"><h4>{title}</h4><span>{description}</span></div>
          <svg className="score-chart" viewBox={`0 0 ${width} ${height}`} role="img" aria-label={`${title}曲线，最终选中的关键帧以圆点标记`}>
            {[minValue, (minValue + maxValue) / 2, maxValue].map((tick) => <line key={tick} x1={padX} x2={width - padX} y1={y(tick)} y2={y(tick)} className="grid-line" />)}
            <polyline points={points} className="score-line" />
            {candidates.filter((frame) => frame.selected).map((frame) => (
              <circle key={frame.candidate_index} cx={x(frame.timestamp_seconds)} cy={y(frame[field] ?? 0)} r="6" className="selected-dot">
                <title>{`${frame.timestamp_seconds.toFixed(2)}s · ${(frame[field] ?? 0).toFixed(6)}`}</title>
              </circle>
            ))}
            <text x={padX} y={height - 2} className="axis-label">0s</text>
            <text x={width - padX} y={height - 2} textAnchor="end" className="axis-label">{maxTime.toFixed(1)}s</text>
          </svg>
        </section>
      );
    };

    return (
      <div className="chart-wrap">
        <div className="chart-heading"><h3>VSI 访问帧分数</h3><span>横轴为视频时间 · 橙色圆点为最终选中帧</span></div>
        <div className="vsi-score-plots">
          {renderVsiChart("fused_score", "融合分数", "目标检测与字幕相关性的加权结果")}
          {renderVsiChart("sampling_probability", "最终采样概率", "算法最后一轮对视频时间位置的采样倾向")}
        </div>
      </div>
    );
  }

  const values = candidates.map((frame) => frame.normalized_score ?? 0);
  const minValue = 0;
  const maxValue = Math.max(1e-6, ...values);
  const y = (value: number) => height - padY - ((value - minValue) / (maxValue - minValue || 1)) * (height - padY * 2);
  const points = candidates.map((frame) => `${x(frame.timestamp_seconds)},${y(frame.normalized_score ?? 0)}`).join(" ");

  return (
    <div className="chart-wrap">
      <div className="chart-heading"><h3>候选帧相关性</h3><span>归一化分数 · 横轴为视频时间</span></div>
      <svg className="score-chart" viewBox={`0 0 ${width} ${height}`} role="img" aria-label="候选帧相关性曲线，选中的帧以圆点标记">
        {[minValue, (minValue + maxValue) / 2, maxValue].map((tick) => <line key={tick} x1={padX} x2={width - padX} y1={y(tick)} y2={y(tick)} className="grid-line" />)}
        <polyline points={points} className="score-line" />
        {candidates.filter((frame) => frame.selected).map((frame) => (
          <circle key={frame.candidate_index} cx={x(frame.timestamp_seconds)} cy={y(frame.normalized_score ?? 0)} r="6" className="selected-dot">
            <title>{`${frame.timestamp_seconds.toFixed(2)}s · ${(frame.relevance_score ?? 0).toFixed(4)}`}</title>
          </circle>
        ))}
        <text x={padX} y={height - 2} className="axis-label">0s</text>
        <text x={width - padX} y={height - 2} textAnchor="end" className="axis-label">{maxTime.toFixed(1)}s</text>
      </svg>
    </div>
  );
}
