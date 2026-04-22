import { useTrainingStore } from "../store/training";

export function TrainingLogs() {
  const { logs, clearLogs } = useTrainingStore();
  
  const getLogColor = (type: string) => {
    switch (type) {
      case "training": return "#4ade80";
      case "episode": return "#3b82f6";
      case "adi": return "#f59e0b";
      case "checkpoint": return "#a855f7";
      case "error": return "#ef4444";
      case "warning": return "#f97316";
      case "scramble": return "#06b6d4";
      case "solve": return "#10b981";
      default: return "#888888";
    }
  };
  
  const formatTime = (timestamp: number) => {
    return new Date(timestamp).toLocaleTimeString();
  };
  
  return (
    <div className="training-logs">
      <div className="logs-header">
        <h3>Training Logs</h3>
        <button onClick={clearLogs} className="btn btn-clear">
          Clear
        </button>
      </div>
      <div className="logs-content">
        {logs.length === 0 ? (
          <p className="no-logs">No logs yet...</p>
        ) : (
          logs.map((log, index) => (
            <div key={`${log.timestamp}-${index}`} className="log-entry">
              <span className="log-time">{formatTime(log.timestamp)}</span>
              <span
                className="log-type"
                style={{ color: getLogColor(log.type) }}
              >
                [{log.type}]
              </span>
              <span className="log-message">{log.message}</span>
            </div>
          ))
        )}
      </div>
    </div>
  );
}
