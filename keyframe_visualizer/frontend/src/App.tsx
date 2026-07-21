import { useCallback, useEffect, useState } from "react";
import { api } from "./api";
import { ResultView } from "./components/ResultView";
import { RunForm } from "./components/RunForm";
import { RunList } from "./components/RunList";
import type { AlgorithmMetadata, Job, Manifest, RunParameters } from "./types";

export default function App() {
  const [algorithms, setAlgorithms] = useState<AlgorithmMetadata[]>([]);
  const [jobs, setJobs] = useState<Job[]>([]);
  const [selected, setSelected] = useState<Job | null>(null);
  const [manifest, setManifest] = useState<Manifest | null>(null);
  const [submitting, setSubmitting] = useState(false);
  const [globalError, setGlobalError] = useState("");

  const refresh = useCallback(async () => {
    const next = await api.jobs();
    setJobs(next);
    setSelected((current) => current ? next.find((job) => job.id === current.id) ?? current : next[0] ?? null);
  }, []);

  useEffect(() => {
    Promise.all([api.algorithms(), api.jobs()])
      .then(([algorithmData, jobData]) => {
        setAlgorithms(algorithmData);
        setJobs(jobData);
        setSelected(jobData[0] ?? null);
      })
      .catch((error: Error) => setGlobalError(`无法连接后端：${error.message}`));
  }, []);

  useEffect(() => {
    if (!selected || selected.status === "succeeded" || selected.status === "failed") return;
    const events = new EventSource(api.eventsUrl(selected.id));
    events.onmessage = (event) => {
      const update = JSON.parse(event.data) as Job;
      setSelected(update);
      setJobs((current) => current.map((job) => job.id === update.id ? update : job));
      if (update.status === "succeeded" || update.status === "failed") {
        events.close();
        void refresh();
      }
    };
    return () => events.close();
  }, [selected?.id, selected?.status, refresh]);

  useEffect(() => {
    setManifest(null);
    if (selected?.status === "succeeded") {
      api.manifest(selected.id).then(setManifest).catch((error: Error) => setGlobalError(error.message));
    }
  }, [selected?.id, selected?.status]);

  const create = async (video: File, query: string, parameters: RunParameters) => {
    setSubmitting(true);
    setGlobalError("");
    try {
      const job = await api.createJob(video, query, parameters);
      setJobs((current) => [job, ...current]);
      setSelected(job);
      setManifest(null);
    } catch (error) {
      setGlobalError(error instanceof Error ? error.message : "提交任务失败");
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <div className="app-shell">
      <header className="topbar">
        <div className="brand"><span className="brand-mark"><i /><i /><i /></span><div><h1>Keyframe Lab</h1><p>可解释的视频抽帧工作台</p></div></div>
        <div className="topbar-status"><span className="live-dot" />AKS connected</div>
      </header>

      {globalError && <div className="global-error" role="alert">{globalError}<button onClick={() => setGlobalError("")}>关闭</button></div>}

      <main className="workspace">
        <aside className="control-column">
          <RunForm algorithm={algorithms.find((item) => item.id === "aks")} busy={submitting} onSubmit={create} />
          <RunList jobs={jobs} selectedId={selected?.id ?? null} onSelect={setSelected} />
        </aside>
        <div className="result-column"><ResultView job={selected} manifest={manifest} /></div>
      </main>
    </div>
  );
}

