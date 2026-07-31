import type {
  AlgorithmMetadata,
  ClipModel,
  Job,
  Manifest,
  RunParameters,
  Session,
  VlmAnswer,
  VlmProfile,
} from "./types";

const API_BASE = import.meta.env.VITE_API_BASE ?? "";

async function checked<T>(responsePromise: Promise<Response>): Promise<T> {
  const response = await responsePromise;
  if (!response.ok) {
    let message = `${response.status} ${response.statusText}`;
    try {
      const body = await response.json();
      message = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail);
    } catch {
      // Keep the HTTP status when the response is not JSON.
    }
    throw new Error(message);
  }
  return response.json() as Promise<T>;
}

export const api = {
  algorithms: () => checked<AlgorithmMetadata[]>(fetch(`${API_BASE}/api/algorithms`)),
  clipModels: () => checked<ClipModel[]>(fetch(`${API_BASE}/api/models/clip`)),
  vlmProfiles: () =>
    checked<Record<string, VlmProfile>>(fetch(`${API_BASE}/api/settings/vlm-profiles`)),
  sessions: () => checked<Session[]>(fetch(`${API_BASE}/api/sessions`)),
  session: (id: string) => checked<Session>(fetch(`${API_BASE}/api/sessions/${id}`)),
  jobs: () => checked<Job[]>(fetch(`${API_BASE}/api/jobs`)),
  job: (id: string) => checked<Job>(fetch(`${API_BASE}/api/jobs/${id}`)),
  createSession: async (video: File, parameters: RunParameters) => {
    const body = new FormData();
    body.append("video", video);
    body.append("config", JSON.stringify({ algorithm: "aks", parameters }));
    return checked<Session>(fetch(`${API_BASE}/api/sessions`, { method: "POST", body }));
  },
  createSessionJob: (sessionId: string, query: string, parameters: RunParameters) =>
    checked<Job>(
      fetch(`${API_BASE}/api/sessions/${sessionId}/jobs`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ algorithm: "aks", query, parameters }),
      }),
    ),
  deleteJob: async (id: string) => {
    const response = await fetch(`${API_BASE}/api/jobs/${id}`, { method: "DELETE" });
    if (!response.ok) {
      let message = `${response.status} ${response.statusText}`;
      try {
        const body = await response.json();
        message = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail);
      } catch {
        // Keep the HTTP status for non-JSON failures.
      }
      throw new Error(message);
    }
  },
  deleteSession: async (id: string) => {
    const response = await fetch(`${API_BASE}/api/sessions/${id}`, { method: "DELETE" });
    if (!response.ok) {
      let message = `${response.status} ${response.statusText}`;
      try {
        const body = await response.json();
        message = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail);
      } catch {
        // Keep the HTTP status for non-JSON failures.
      }
      throw new Error(message);
    }
  },
  manifest: (id: string) =>
    checked<Manifest>(fetch(`${API_BASE}/api/jobs/${id}/manifest`)),
  savedVlmAnswer: async (id: string, frameSet: string) => {
    const response = await fetch(
      `${API_BASE}/api/jobs/${id}/vlm-answer?frame_set=${encodeURIComponent(frameSet)}`,
    );
    if (response.status === 404) return null;
    return checked<VlmAnswer>(Promise.resolve(response));
  },
  createVlmAnswer: (id: string, frameSet: string, query: string, vlmProfile: string) =>
    checked<VlmAnswer>(
      fetch(`${API_BASE}/api/jobs/${id}/vlm-answer`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ frame_set: frameSet, query, vlm_profile: vlmProfile }),
      }),
    ),
  createJob: async (video: File, query: string, parameters: RunParameters) => {
    const body = new FormData();
    body.append("video", video);
    body.append("config", JSON.stringify({ algorithm: "aks", query, parameters }));
    return checked<Job>(fetch(`${API_BASE}/api/jobs`, { method: "POST", body }));
  },
  uploadClipModel: async (archive: File, name: string) => {
    const body = new FormData();
    body.append("archive", archive);
    body.append("name", name);
    return checked<ClipModel>(fetch(`${API_BASE}/api/models/clip`, { method: "POST", body }));
  },
  eventsUrl: (id: string) => `${API_BASE}/api/jobs/${id}/events`,
  sessionEventsUrl: (id: string) => `${API_BASE}/api/sessions/${id}/events`,
  videoUrl: (id: string) => `${API_BASE}/api/jobs/${id}/video`,
  mediaUrl: (id: string, path: string) =>
    `${API_BASE}/api/jobs/${id}/media/${path.split("/").map(encodeURIComponent).join("/")}`,
  manifestDownloadUrl: (id: string) => `${API_BASE}/api/jobs/${id}/manifest/download`,
};
