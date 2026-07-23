import type { AlgorithmMetadata, ClipModel, Job, Manifest, RunParameters } from "./types";

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
  jobs: () => checked<Job[]>(fetch(`${API_BASE}/api/jobs`)),
  job: (id: string) => checked<Job>(fetch(`${API_BASE}/api/jobs/${id}`)),
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
  manifest: (id: string) =>
    checked<Manifest>(fetch(`${API_BASE}/api/jobs/${id}/manifest`)),
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
  videoUrl: (id: string) => `${API_BASE}/api/jobs/${id}/video`,
  mediaUrl: (id: string, path: string) =>
    `${API_BASE}/api/jobs/${id}/media/${path.split("/").map(encodeURIComponent).join("/")}`,
  manifestDownloadUrl: (id: string) => `${API_BASE}/api/jobs/${id}/manifest/download`,
};
