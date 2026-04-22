import { CubeVisualizer } from "./components/CubeVisualizer";
import { TrainingDashboard } from "./components/TrainingDashboard";
import { TrainingLogs } from "./components/TrainingLogs";
import { useWebSocket } from "./api/client";
import "./App.css";

import { MetricsChart } from "./components/MetricsChart";

function App() {
  const { send } = useWebSocket();
  console.log("Rendering App component with core logic...");
  
  return (
    <div className="app">
      <header className="app-header">
        <h1>🧊 Rubik's Cube DeepCubeA</h1>
        <p className="subtitle">
          DeepCubeA-style cost-to-go training and search
        </p>
      </header>
      
      <main className="app-main">
        <div className="left-panel">
          <CubeVisualizer />
          <TrainingLogs />
        </div>
        
        <div className="center-panel">
          <TrainingDashboard send={send} />
          <MetricsChart />
        </div>
      </main>
    </div>
  );
}

export default App;
