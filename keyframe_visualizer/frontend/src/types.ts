export type JobStatus = "queued" | "running" | "succeeded" | "failed";

export interface Job {
  id: string;
  created_at: string;
  updated_at: string;
  status: JobStatus;
  stage: string;
  progress: number;
  algorithm: string;
  query: string;
  original_filename: string;
  error: string | null;
  manifest_available: boolean;
}

export interface FeatureProfile {
  id: string;
  name: string;
  backend: string;
}

export interface AlgorithmMetadata {
  id: string;
  name: string;
  description: string;
  parameter_schema: {
    feature_profiles: FeatureProfile[];
    defaults: Record<string, unknown>;
  };
}

export interface CandidateFrame {
  candidate_index: number;
  candidate_order: number;
  original_frame_index: number;
  timestamp_seconds: number;
  relevance_score: number;
  normalized_score: number;
  selected: boolean;
}

export interface SelectedFrame extends Omit<CandidateFrame, "selected"> {
  selected_order: number;
  file: string;
  segment_id: number;
  segment_depth: number | null;
  segment_quota: number | null;
  rank_in_segment: number | null;
}

export interface Manifest {
  schema_version: string;
  run_id: string;
  algorithm: { id: string; name: string; mode: string };
  video: {
    filename: string;
    fps: number;
    duration_seconds: number;
    total_frames: number;
  };
  query: string;
  parameters: Record<string, unknown>;
  candidate_sampling: {
    mode: string;
    interval_seconds: number | null;
    candidate_count: number;
  };
  feature_extraction: Record<string, unknown>;
  summary: {
    requested_keyframes: number;
    selected_keyframes: number;
    candidate_frames: number;
  };
  selected_frames: SelectedFrame[];
  candidates: CandidateFrame[];
}

export interface RunParameters {
  aks_mode: "robust" | "original";
  max_num_frames: number;
  candidate_sampling: "interval" | "original";
  sample_interval: number;
  feature_backend: "clip" | "pangu" | "mep";
  feature_profile: string | null;
  model_name: string;
  device: "auto" | "cuda" | "mps" | "cpu";
  batch_size: number;
  decode_threads: number;
  threshold: number;
  std_threshold: number;
  max_depth: number;
  jpeg_quality: number;
}

