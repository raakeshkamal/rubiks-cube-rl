export interface TrainingConfig {
  scramble_depth: number;
  max_steps_per_episode: number;
  episodes_per_epoch: number;
  checkpoint_freq: number;
  hidden_size: number;
  learning_rate: number;
  max_scramble_depth: number;
  batch_size?: number;
  states_per_update?: number;
  train_epochs_per_update?: number;
  loss_thresh?: number;
  back_max?: number;
  search_weight?: number;
  artifact_root?: string;
  epsilon_start?: number;
  epsilon_end?: number;
  epsilon_decay?: number;
  adi_states_per_step?: number;
  adi_steps_per_epoch?: number;
  target_update_freq?: number;
  loss_threshold?: number;
}

export interface TrainingMetrics {
  episode: number;
  total_steps: number;
  epoch: number;
  solved: number;
  total_reward: number;
  avg_reward: number;
  solve_rate: number;
  epsilon: number;
  avg_loss: number;
  epoch_time: number;
  episodes_per_sec: number;
  current_scramble_depth: number;
  device: string;
  adi_steps: number;
  lr?: number;
  update_num?: number;
}

export interface EpisodeResult {
  episode: number;
  reward: number;
  solved: boolean;
  steps: number;
  scramble_depth: number;
}

export interface ScrambleResult {
  state: number[];
  moves: string[];
  depth: number;
}

export interface CubeStepResult {
  state: number[];
  move: string;
  step: number;
  solved: boolean;
  distance: number;
}

export interface LogEntry {
  timestamp: number;
  type: string;
  message: string;
}

export type WsMessageType =
  | "connected"
  | "disconnected"
  | "training_started"
  | "training_stopped"
  | "training_complete"
  | "metrics_update"
  | "adi_progress"
  | "episode_complete"
  | "checkpoint_saved"
  | "status"
  | "scramble_result"
  | "cube_step"
  | "solve_result"
  | "error"
  | "pong";

export interface WsMessage {
  type: WsMessageType;
  payload: any;
}
