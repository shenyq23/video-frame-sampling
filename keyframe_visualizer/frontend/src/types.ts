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
  parameters: Partial<RunParameters>;
  error: string | null;
  manifest_available: boolean;
}

export interface FeatureProfile {
  id: string;
  name: string;
  backend: string;
  enabled: boolean;
  credentials_ready: boolean;
  missing_environment_variables: string[];
}

export interface ClipModel {
  id: string;
  name: string;
  source_filename: string;
  size_bytes: number;
  created_at: string;
}

export interface VlmProfile {
  id: string;
  name: string;
  backend: string;
  enabled: boolean;
  credentials_ready: boolean;
  required_environment_variables: string[];
  missing_environment_variables: string[];
  max_frames: number;
}

export interface VlmUsedFrame {
  order: number | null;
  file: string;
  timestamp_seconds: number | null;
  original_frame_index: number | null;
  candidate_order: number | null;
}

export interface VlmAnswer {
  schema_version: string;
  job_id: string;
  created_at: string;
  profile_id: string;
  profile_name: string;
  frame_set: "selected" | "uniform" | "candidates";
  frame_set_name: string;
  query: string;
  answer: string;
  source_frame_count: number;
  used_frame_count: number;
  frames_limited: boolean;
  used_frames: VlmUsedFrame[];
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
  file?: string;
  order?: number;
  selected_by_aks?: boolean;
}

export interface SelectedFrame extends Omit<CandidateFrame, "selected"> {
  selected_order: number;
  file: string;
  segment_id: number;
  segment_depth: number | null;
  segment_quota: number | null;
  rank_in_segment: number | null;
}

export interface FrameRecord {
  order: number;
  selected_order?: number;
  file: string;
  original_frame_index: number;
  timestamp_seconds: number;
  candidate_index: number;
  candidate_order: number;
  relevance_score: number;
  normalized_score: number;
  selected_by_aks: boolean;
  selected?: boolean;
  segment_id?: number;
  segment_depth?: number | null;
  segment_quota?: number | null;
  rank_in_segment?: number | null;
}

export interface FrameSet {
  available: boolean;
  count: number;
  selection_rule?: string;
  frames: FrameRecord[];
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
    effective_interval_seconds: number | null;
    candidate_count: number;
  };
  feature_extraction: Record<string, unknown>;
  summary: {
    requested_keyframes: number;
    selected_keyframes: number;
    candidate_frames: number;
  };
  selected_frames: SelectedFrame[];
  uniform_frames?: FrameRecord[];
  candidates: CandidateFrame[];
  frame_sets?: {
    selected: FrameSet;
    uniform: FrameSet;
    candidates: FrameSet;
  };
}

export interface RunParameters {
  aks_mode: "robust" | "original";
  max_num_frames: number;
  candidate_sampling: "interval" | "original";
  sample_interval: number;
  feature_backend: "clip" | "pangu" | "mep";
  feature_profile: string | null;
  clip_model_id: string | null;
  model_name: string;
  device: "auto" | "cuda" | "mps" | "cpu";
  batch_size: number;
  decode_threads: number;
  threshold: number;
  std_threshold: number;
  max_depth: number;
  jpeg_quality: number;
  save_uniform_baseline: boolean;
  save_candidate_frames: boolean;
}
