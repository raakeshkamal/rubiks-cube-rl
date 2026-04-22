import { create } from "zustand";
import type { TrainingMetrics, LogEntry, ScrambleResult, CubeStepResult } from "../types/api";

interface TrainingStore {
  // Connection state
  isConnected: boolean;
  connectionStatus: string;
  
  // Training state
  isTraining: boolean;
  metrics: TrainingMetrics | null;
  
  // Cube state
  cubeState: number[];
  scrambleMoves: string[];
  
  // Live eval step tracking
  lastMove: string;
  lastStepNum: number;
  lastStepSolved: boolean;
  lastStepDistance: number;
  evalMoveHistory: string[];
  
  // Logs
  logs: LogEntry[];
  
  // Metrics history for charts
  metricsHistory: Array<{
    episode: number;
    solve_rate: number;
    avg_reward: number;
    update_num: number;
    avg_loss: number;
  }>;
  
  // Actions
  setConnected: (connected: boolean) => void;
  setConnectionStatus: (status: string) => void;
  setTraining: (training: boolean) => void;
  setMetrics: (metrics: TrainingMetrics) => void;
  setCubeState: (state: number[]) => void;
  setScrambleMoves: (moves: string[]) => void;
  setCubeStep: (step: CubeStepResult) => void;
  addLog: (type: string, message: string) => void;
  addMetricsHistory: (metrics: TrainingMetrics) => void;
  clearLogs: () => void;
}

export const useTrainingStore = create<TrainingStore>((set, get) => ({
  // Initial state
  isConnected: false,
  connectionStatus: "disconnected",
  isTraining: false,
  metrics: null,
  cubeState: Array.from({ length: 54 }, (_, i) => Math.floor(i / 9)),
  scrambleMoves: [],
  lastMove: "",
  lastStepNum: 0,
  lastStepSolved: false,
  lastStepDistance: 0,
  evalMoveHistory: [],
  logs: [],
  metricsHistory: [],
  
  // Actions
  setConnected: (connected) => set({ isConnected: connected }),
  setConnectionStatus: (status) => set({ connectionStatus: status }),
  setTraining: (training) => set({ isTraining: training }),
  setMetrics: (metrics) => set({ metrics }),
  setCubeState: (state) => set({ cubeState: state }),
  setScrambleMoves: (moves) => set({ scrambleMoves: moves }),
  setCubeStep: (step) => set((state) => {
    // Reset move history when a new episode starts (step goes back to 1)
    const history = step.step <= state.lastStepNum
      ? [step.move]
      : [...state.evalMoveHistory.slice(-49), step.move];
    return {
      cubeState: step.state,
      lastMove: step.move,
      lastStepNum: step.step,
      lastStepSolved: step.solved,
      lastStepDistance: step.distance,
      evalMoveHistory: history,
    };
  }),
  
  addLog: (type, message) => set(state => ({
    logs: [
      { timestamp: Date.now(), type, message },
      ...state.logs.slice(0, 99), // Keep last 100
    ],
  })),
  
  addMetricsHistory: (metrics) => set(state => ({
    metricsHistory: [
      ...state.metricsHistory.slice(-199), // Keep last 200 points
      {
        episode: metrics.episode,
        solve_rate: metrics.solve_rate,
        avg_reward: metrics.avg_reward,
        update_num: metrics.update_num ?? 0,
        avg_loss: metrics.avg_loss,
      },
    ],
  })),
  
  clearLogs: () => set({ logs: [] }),
}));
