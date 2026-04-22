import { useTrainingStore } from "../store/training";

export function TrainingDashboard({ send }: { send: (type: string, payload?: any) => void }) {
  const { isTraining, metrics, connectionStatus } = useTrainingStore();
  
  const handleStart = () => {
    send("start_training", {});
  };
  
  const handleResume = () => {
    send("start_training", { resume: true });
  };
  
  const handleStop = () => {
    send("stop_training");
  };
  
  const handleScramble = () => {
    send("scramble", { depth: 5 });
  };
  
  return (
    <div className="training-dashboard">
      <h2>Training Controls</h2>
      
      <div className="status-bar">
        <span className={`status ${connectionStatus}`}>
          {connectionStatus === "connected" ? "🟢 Connected" : 
           connectionStatus === "connecting" ? "🟡 Connecting..." : 
           "🔴 Disconnected"}
        </span>
        <span className={`training-status ${isTraining ? "training" : ""}`}>
          {isTraining ? "🏃 Training (DeepCubeA)" : "⏸️ Idle"}
        </span>
      </div>
      
      <div className="control-buttons">
        <button
          onClick={handleStart}
          disabled={isTraining}
          className="btn btn-start"
        >
          ▶️ Start Training
        </button>
        <button
          onClick={handleResume}
          disabled={isTraining}
          className="btn btn-start"
          style={{ marginLeft: '8px' }}
        >
          ⏯️ Resume Training
        </button>
        <button
          onClick={handleStop}
          disabled={!isTraining}
          className="btn btn-stop"
        >
          ⏹️ Stop Training
        </button>
        <button
          onClick={handleScramble}
          className="btn btn-scramble"
        >
          🔄 Scramble Cube
        </button>
      </div>
      
      {metrics && (
        <div className="metrics-grid">
          <div className="metric">
            <span className="metric-label">Epoch</span>
            <span className="metric-value">{metrics.epoch}</span>
          </div>
          <div className="metric">
            <span className="metric-label">Solve Rate</span>
            <span className="metric-value">{((metrics.solve_rate ?? 0) * 100).toFixed(1)}%</span>
          </div>
          <div className="metric">
            <span className="metric-label">Avg Reward</span>
            <span className="metric-value">{(metrics.avg_reward ?? 0).toFixed(2)}</span>
          </div>
          <div className="metric">
            <span className="metric-label">Updates</span>
            <span className="metric-value">{metrics.update_num ?? 0}</span>
          </div>
          <div className="metric">
            <span className="metric-label">Scramble Depth</span>
            <span className="metric-value">{metrics.current_scramble_depth}</span>
          </div>
          <div className="metric">
            <span className="metric-label">Train Steps</span>
            <span className="metric-value">{metrics.adi_steps?.toLocaleString() ?? "—"}</span>
          </div>
          <div className="metric">
            <span className="metric-label">Avg Loss</span>
            <span className="metric-value">{(metrics.avg_loss ?? 0).toFixed(4)}</span>
          </div>
          <div className="metric">
            <span className="metric-label">Epoch Time</span>
            <span className="metric-value">{(metrics.epoch_time ?? 0).toFixed(2)}s</span>
          </div>
          <div className="metric">
            <span className="metric-label">Learning Rate</span>
            <span className="metric-value">{(metrics.lr ?? 0).toExponential(2)}</span>
          </div>
          <div className="metric">
            <span className="metric-label">Device</span>
            <span className="metric-value">{metrics.device}</span>
          </div>
        </div>
      )}
    </div>
  );
}
