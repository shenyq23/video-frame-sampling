export type JobStatus = "queued" | "running" | "succeeded" | "failed";
export type AlgorithmId = "aks" | "vsi" | "sage";
export type ParameterSnapshot = Record<string, any>;

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
  session_id: string | null;
  owns_video: boolean;
  parameters: ParameterSnapshot;
  error: string | null;
  manifest_available: boolean;
}

export interface Session {
  id: string;
  created_at: string;
  updated_at: string;
  status: JobStatus;
  stage: string;
  progress: number;
  original_filename: string;
  algorithm: AlgorithmId;
  parameters: ParameterSnapshot;
  candidate_count: number;
  error: string | null;
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
  generation_duration_seconds?: number | null;
  source_frame_count: number;
  used_frame_count: number;
  frames_limited: boolean;
  used_frames: VlmUsedFrame[];
}

export interface AlgorithmMetadata {
  id: string;
  name: string;
  description: string;
  parameter_schema: Record<string, unknown> & {
    feature_profiles?: FeatureProfile[];
    defaults?: Record<string, unknown>;
    assets?: Record<string, {
      label: string;
      ready: boolean;
    }>;
  };
}

export interface SessionConfig {
  algorithm: AlgorithmId;
  parameters: Record<string, unknown>;
}

export interface CandidateFrame {
  candidate_index: number;
  candidate_order: number;
  original_frame_index: number;
  timestamp_seconds: number;
  relevance_score: number;
  normalized_score?: number;
  selected?: boolean;
  file?: string;
  order?: number;
  selected_by_aks?: boolean;
  selected_by_vsi?: boolean;
  selected_by_sage?: boolean;
  fused_score?: number;
  sage_score?: number;
  change_score?: number;
  base_score?: number;
  sampling_probability?: number;
  visited_order?: number;
}

export interface SelectedFrame extends Omit<CandidateFrame, "selected"> {
  selected_order: number;
  file: string;
  segment_id?: number;
  segment_depth?: number | null;
  segment_quota?: number | null;
  rank_in_segment?: number | null;
}

export interface FrameRecord {
  order: number;
  selected_order?: number;
  file: string;
  original_frame_index: number;
  timestamp_seconds: number;
  candidate_index: number;
  candidate_order: number;
  relevance_score?: number;
  normalized_score?: number;
  selected_by_aks?: boolean;
  selected_by_vsi?: boolean;
  selected_by_sage?: boolean;
  object_score?: number;
  text_score?: number;
  fused_score?: number;
  sampling_probability?: number;
  visited_order?: number;
  sage_score?: number;
  change_score?: number;
  base_score?: number;
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
  parameters: ParameterSnapshot;
  candidate_sampling?: {
    mode: string;
    interval_seconds: number | null;
    effective_interval_seconds: number | null;
    candidate_count: number;
  };
  feature_extraction?: Record<string, unknown>;
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

export interface VsiSessionParameters {
  subtitle_mode: "ocr" | "upload" | "none";
  ocr_fps: number;
  ocr_crop_top: number;
  text_model: string;
  device: "cuda" | "mps" | "cpu";
}

export interface VsiQueryParameters extends VsiSessionParameters {
  objects: string[];
  top_k: number;
  detection_budget: number;
  samples_per_round: number;
  text_weight: number;
  model: string;
  seed: number;
  save_uniform_baseline: boolean;
  save_candidate_frames: boolean;
}

export interface SageSessionParameters {
  asr_mode: "remote" | "upload" | "none";
  device: "cuda" | "mps" | "cpu";
}

export interface SageQueryParameters extends SageSessionParameters {
  budget: number;
  save_uniform_baseline: boolean;
  save_candidate_frames: boolean;
}
