import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  ReferenceLine,
} from "recharts";
import { useTrainingStore } from "../store/training";

export function MetricsChart() {
  const { metricsHistory } = useTrainingStore();
  
  if (metricsHistory.length === 0) {
    return (
      <div className="metrics-chart">
        <h3>Training Metrics</h3>
        <p className="no-data">No training data yet. Start training to see metrics.</p>
      </div>
    );
  }
  
  return (
    <div className="metrics-chart">
      <h3>Training Metrics</h3>
      
      <div className="chart-container">
        <h4>Solve Rate</h4>
        <ResponsiveContainer width="100%" height={200}>
          <LineChart data={metricsHistory}>
            <CartesianGrid strokeDasharray="3 3" stroke="#333" />
            <XAxis 
              dataKey="episode" 
              stroke="#888"
              tick={{ fill: '#888', fontSize: 10 }}
            />
            <YAxis 
              domain={[0, 1]} 
              stroke="#888"
              tick={{ fill: '#888', fontSize: 10 }}
            />
            <Tooltip 
              contentStyle={{ backgroundColor: '#1a1a1a', border: '1px solid #333' }}
              labelStyle={{ color: '#fff' }}
            />
            <Legend />
            <Line 
              type="monotone" 
              dataKey="solve_rate" 
              stroke="#4ade80" 
              name="Solve Rate"
              dot={false}
              strokeWidth={2}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>
      
      <div className="chart-container">
        <h4>Average Reward</h4>
        <ResponsiveContainer width="100%" height={200}>
          <LineChart data={metricsHistory}>
            <CartesianGrid strokeDasharray="3 3" stroke="#333" />
            <XAxis 
              dataKey="episode" 
              stroke="#888"
              tick={{ fill: '#888', fontSize: 10 }}
            />
            <YAxis 
              stroke="#888"
              tick={{ fill: '#888', fontSize: 10 }}
            />
            <Tooltip 
              contentStyle={{ backgroundColor: '#1a1a1a', border: '1px solid #333' }}
              labelStyle={{ color: '#fff' }}
            />
            <Legend />
            <Line 
              type="monotone" 
              dataKey="avg_reward" 
              stroke="#3b82f6" 
              name="Avg Reward"
              dot={false}
              strokeWidth={2}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>
      
      <div className="chart-container">
        <h4>Training Loss</h4>
        <ResponsiveContainer width="100%" height={200}>
          <LineChart data={metricsHistory}>
            <CartesianGrid strokeDasharray="3 3" stroke="#333" />
            <XAxis 
              dataKey="episode" 
              stroke="#888"
              tick={{ fill: '#888', fontSize: 10 }}
            />
            <YAxis 
              stroke="#888"
              tick={{ fill: '#888', fontSize: 10 }}
            />
            <Tooltip 
              contentStyle={{ backgroundColor: '#1a1a1a', border: '1px solid #333' }}
              labelStyle={{ color: '#fff' }}
            />
            <Legend />
            <ReferenceLine
              y={0.06}
              stroke="#f59e0b"
              strokeDasharray="6 6"
              label={{ value: "Loss Threshold (0.06)", fill: "#f59e0b", fontSize: 10 }}
            />
            <Line 
              type="monotone" 
              dataKey="avg_loss" 
              stroke="#ef4444" 
              name="Avg Loss"
              dot={false}
              strokeWidth={2}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
