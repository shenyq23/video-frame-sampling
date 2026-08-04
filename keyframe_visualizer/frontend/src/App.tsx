import { useCallback, useEffect, useMemo, useState } from "react";
import { api } from "./api";
import { QueryList } from "./components/QueryList";
import { ResultView } from "./components/ResultView";
import { RunForm } from "./components/RunForm";
import { VsiForm } from "./components/VsiForm";
import { RunList } from "./components/RunList";
import type {
  AlgorithmMetadata,
  ClipModel,
  Job,
  Manifest,
  RunParameters,
  AlgorithmId,
  VsiQueryParameters,
  VsiSessionParameters,
  Session,
  VlmProfile,
  ParameterSnapshot,
} from "./types";

export default function App() {
  const [algorithms, setAlgorithms] = useState<AlgorithmMetadata[]>([]);
  const [sessions, setSessions] = useState<Session[]>([]);
  const [currentSession, setCurrentSession] = useState<Session | null>(null);
  const [jobs, setJobs] = useState<Job[]>([]);
  const [clipModels, setClipModels] = useState<ClipModel[]>([]);
  const [vlmProfiles, setVlmProfiles] = useState<VlmProfile[]>([]);
  const [selected, setSelected] = useState<Job | null>(null);
  const [manifest, setManifest] = useState<Manifest | null>(null);
  const [submitting, setSubmitting] = useState(false);
  const [preparingNewSession, setPreparingNewSession] = useState(false);
  const [deletingJobId, setDeletingJobId] = useState<string | null>(null);
  const [deletingSessionId, setDeletingSessionId] = useState<string | null>(null);
  const [pendingDelete, setPendingDelete] = useState<Job | null>(null);
  const [pendingSessionDelete, setPendingSessionDelete] = useState<Session | null>(null);
  const [globalError, setGlobalError] = useState("");
  const [activeAlgorithm, setActiveAlgorithm] = useState<AlgorithmId>("aks");

  const latestJobForSession = useCallback((items: Job[], sessionId: string | null) => {
    if (!sessionId) return null;
    return items.find((job) => job.session_id === sessionId) ?? null;
  }, []);

  const currentSessionJobs = useMemo(
    () => currentSession ? jobs.filter((job) => job.session_id === currentSession.id) : [],
    [jobs, currentSession?.id],
  );

  const refresh = useCallback(async () => {
    const [nextJobs, nextSessions] = await Promise.all([api.jobs(), api.sessions()]);
    setJobs(nextJobs);
    setSessions(nextSessions);
    setSelected((current) => current ? nextJobs.find((job) => job.id === current.id) ?? current : null);
    setCurrentSession((current) => {
      if (current) return nextSessions.find((session) => session.id === current.id) ?? current;
      return nextSessions.find((session) => session.status === "succeeded") ?? nextSessions[0] ?? null;
    });
  }, []);

  useEffect(() => {
    Promise.all([api.algorithms(), api.sessions(), api.jobs(), api.clipModels(), api.vlmProfiles()])
      .then(([algorithmData, sessionData, jobData, modelData, vlmProfileData]) => {
        setAlgorithms(algorithmData);
        setSessions(sessionData);
        setJobs(jobData);
        setClipModels(modelData);
        setVlmProfiles(Object.values(vlmProfileData));
        const initialSession =
          sessionData.find((session) => session.algorithm === "aks" && session.status === "succeeded") ??
          sessionData.find((session) => session.algorithm === "aks") ??
          null;
        setCurrentSession(initialSession);
        setSelected(initialSession ? latestJobForSession(jobData, initialSession.id) : null);
      })
      .catch((error: Error) => setGlobalError(`无法连接后端：${error.message}`));
  }, [latestJobForSession]);

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
    if (!currentSession || currentSession.status === "succeeded" || currentSession.status === "failed") return;
    const events = new EventSource(api.sessionEventsUrl(currentSession.id));
    events.onmessage = (event) => {
      const update = JSON.parse(event.data) as Session;
      setCurrentSession(update);
      setSessions((current) => current.map((session) => session.id === update.id ? update : session));
      if (update.status === "succeeded" || update.status === "failed") {
        events.close();
        void refresh();
      }
    };
    return () => events.close();
  }, [currentSession?.id, currentSession?.status, refresh]);

  useEffect(() => {
    setManifest(null);
    if (selected?.status === "succeeded") {
      api.manifest(selected.id).then(setManifest).catch((error: Error) => setGlobalError(error.message));
    }
  }, [selected?.id, selected?.status]);

  const prepareSession = async (
    algorithm: AlgorithmId,
    video: File,
    parameters: ParameterSnapshot,
    subtitle: File | null = null,
  ) => {
    setSubmitting(true);
    setGlobalError("");
    try {
      const session = await api.createSession(algorithm, video, parameters, subtitle);
      setSessions((current) => [session, ...current.filter((item) => item.id !== session.id)]);
      setCurrentSession(session);
      setSelected(null);
      setManifest(null);
      setPreparingNewSession(false);
    } catch (error) {
      setGlobalError(error instanceof Error ? error.message : "提交视频预处理失败");
    } finally {
      setSubmitting(false);
    }
  };

  const runSessionQuery = async (query: string, parameters: ParameterSnapshot) => {
    if (!currentSession || currentSession.status !== "succeeded") {
      setGlobalError("请先等待视频预处理完成");
      return false;
    }
    setSubmitting(true);
    setGlobalError("");
    try {
      const job = await api.createSessionJob(currentSession.id, activeAlgorithm, query, parameters);
      setJobs((current) => [job, ...current]);
      setSelected(job);
      setManifest(null);
      return true;
    } catch (error) {
      setGlobalError(error instanceof Error ? error.message : "提交 query 任务失败");
      return false;
    } finally {
      setSubmitting(false);
    }
  };

  const uploadClipModel = async (archive: File, name: string) => {
    const model = await api.uploadClipModel(archive, name);
    setClipModels((current) => [model, ...current.filter((item) => item.id !== model.id)]);
    return model;
  };

  const deleteJob = async (job: Job) => {
    setDeletingJobId(job.id);
    setGlobalError("");
    const deletingSelected = selected?.id === job.id;
    if (deletingSelected) {
      setPendingDelete(job);
      setSelected(null);
      setManifest(null);
      await new Promise<void>((resolve) => window.setTimeout(resolve, 350));
    }
    let deleted = false;
    try {
      await api.deleteJob(job.id);
      deleted = true;
      let remaining: Job[];
      try {
        remaining = await api.jobs();
      } catch {
        remaining = jobs.filter((item) => item.id !== job.id);
      }
      setJobs(remaining);
      if (deletingSelected) {
        setSelected(latestJobForSession(remaining, job.session_id ?? currentSession?.id ?? null));
        setPendingDelete(null);
      }
    } catch (error) {
      setGlobalError(error instanceof Error ? error.message : "清除任务失败");
      if (!deleted && deletingSelected) setSelected(job);
      if (deletingSelected) setPendingDelete(null);
    } finally {
      setDeletingJobId(null);
    }
  };

  const deleteSession = async (session: Session) => {
    setDeletingSessionId(session.id);
    setGlobalError("");
    setPendingSessionDelete(session);
    setCurrentSession(null);
    setSelected(null);
    setManifest(null);
    await new Promise<void>((resolve) => window.setTimeout(resolve, 350));
    try {
      await api.deleteSession(session.id);
      const [remainingJobs, remainingSessions] = await Promise.all([api.jobs(), api.sessions()]);
      setJobs(remainingJobs);
      setSessions(remainingSessions);
      const nextSession =
        remainingSessions.find((item) => item.status === "succeeded") ??
        remainingSessions[0] ??
        null;
      setCurrentSession(nextSession);
      setSelected(nextSession ? latestJobForSession(remainingJobs, nextSession.id) : null);
      setPendingSessionDelete(null);
    } catch (error) {
      setGlobalError(error instanceof Error ? error.message : "删除视频会话失败");
      setCurrentSession(session);
      setSelected(latestJobForSession(jobs, session.id));
      setPendingSessionDelete(null);
    } finally {
      setDeletingSessionId(null);
    }
  };

  const selectSession = (session: Session) => {
    setActiveAlgorithm(session.algorithm);
    setPreparingNewSession(false);
    setCurrentSession(session);
    setSelected(latestJobForSession(jobs, session.id));
    setManifest(null);
  };

  const selectJob = (job: Job) => {
    if (job.algorithm === "aks" || job.algorithm === "vsi") setActiveAlgorithm(job.algorithm);
    setPreparingNewSession(false);
    setSelected(job);
    if (job.session_id) {
      const session = sessions.find((item) => item.id === job.session_id);
      if (session) setCurrentSession(session);
    }
  };

  const switchAlgorithm = (algorithm: AlgorithmId) => {
    setActiveAlgorithm(algorithm);
    setSelected(null);
    setManifest(null);
    const next = sessions.find((session) => session.algorithm === algorithm && session.status === "succeeded")
      ?? sessions.find((session) => session.algorithm === algorithm)
      ?? null;
    setCurrentSession(next);
    setPreparingNewSession(!next);
    setGlobalError("");
  };

  const visibleSessions = useMemo(
    () => sessions.filter((session) => session.algorithm === activeAlgorithm),
    [sessions, activeAlgorithm],
  );

  return (
    <div className="app-shell">
      <header className="topbar">
        <div className="brand"><span className="brand-mark"><i /><i /><i /></span><div><h1>Keyframe Lab</h1><p>可解释的视频抽帧工作台</p></div></div>
        <div className="topbar-status"><span className="live-dot" />服务已连接</div>
      </header>

      {globalError && <div className="global-error" role="alert">{globalError}<button onClick={() => setGlobalError("")}>关闭</button></div>}

      <main className="workspace">
        <aside className="control-column">
          <div className="algorithm-switcher" role="tablist" aria-label="抽帧算法">
            <span className="algorithm-switcher-label">抽帧算法</span>
            {(["aks", "vsi"] as AlgorithmId[]).map((algorithm) => (
              <button key={algorithm} className={activeAlgorithm === algorithm ? "selected" : ""} type="button" role="tab" aria-selected={activeAlgorithm === algorithm} onClick={() => switchAlgorithm(algorithm)}>
                {algorithm.toUpperCase()}
              </button>
            ))}
          </div>
          {activeAlgorithm === "aks" ? (
            <RunForm
              algorithm={algorithms.find((item) => item.id === "aks")}
              clipModels={clipModels}
              currentSession={currentSession?.algorithm === "aks" ? currentSession : null}
              preparingNewSession={preparingNewSession}
              busy={submitting}
              onPrepareSession={(video, parameters) => prepareSession("aks", video, parameters)}
              onRunQuery={(query, parameters) => runSessionQuery(query, parameters)}
              onUploadClipModel={uploadClipModel}
              onPreparingNewSessionChange={setPreparingNewSession}
            />
          ) : (
            <VsiForm
              algorithm={algorithms.find((item) => item.id === "vsi")}
              currentSession={currentSession?.algorithm === "vsi" ? currentSession : null}
              preparingNewSession={preparingNewSession}
              busy={submitting}
              onPrepareSession={(video, parameters, subtitle) => prepareSession("vsi", video, parameters, subtitle)}
              onRunQuery={(query, parameters) => runSessionQuery(query, parameters)}
              onPreparingNewSessionChange={setPreparingNewSession}
            />
          )}
          <RunList
            sessions={visibleSessions}
            jobs={jobs}
            selectedSessionId={preparingNewSession ? null : currentSession?.algorithm === activeAlgorithm ? currentSession.id : null}
            onSelect={selectSession}
          />
        </aside>
        <div className="result-column">
          {preparingNewSession ? (
            <section className="empty-state">
              <div className="empty-orbit" />
              <h2>等待视频上传</h2>
              <p>请在左侧选择并准备一个新视频。完成预处理后，可以针对它连续输入 query。</p>
            </section>
          ) : pendingSessionDelete ? (
            <section className="deleting-state" aria-live="polite">
              <div className="empty-orbit" />
              <h2>正在删除视频会话</h2>
              <p>{pendingSessionDelete.original_filename}</p>
              <p>正在删除视频预处理缓存和全部 query 历史…</p>
            </section>
          ) : (
            <div className="session-detail-layout">
              <QueryList
                session={currentSession}
                jobs={currentSessionJobs}
                selectedJobId={selected?.id ?? null}
                deletingJobId={deletingJobId}
                deletingSessionId={deletingSessionId}
                hasActiveJobs={currentSessionJobs.some((job) => job.status === "queued" || job.status === "running")}
                onSelect={selectJob}
                onDeleteJob={deleteJob}
                onDeleteSession={deleteSession}
              />
              <div className="job-result-pane">
                {pendingDelete ? (
                  <section className="deleting-state" aria-live="polite">
                    <div className="empty-orbit" />
                    <h2>正在删除 query 记录</h2>
                    <p>{pendingDelete.query}</p>
                    <p>正在释放视频连接并删除这条 query 的结果…</p>
                  </section>
                ) : (
                  <ResultView
                    job={selected}
                    currentSession={currentSession}
                    manifest={manifest}
                    clipModels={clipModels}
                    vlmProfiles={vlmProfiles}
                    deleting={selected?.id === deletingJobId}
                    onDelete={deleteJob}
                  />
                )}
              </div>
            </div>
          )}
        </div>
      </main>
    </div>
  );
}
