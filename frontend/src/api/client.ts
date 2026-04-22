import { useEffect, useRef, useCallback } from "react";
import { useTrainingStore } from "../store/training";

type MessageHandler = (data: any) => void;

export function useWebSocket(url?: string) {
  const wsRef = useRef<WebSocket | null>(null);
  const handlersRef = useRef<Map<string, MessageHandler[]>>(new Map());
  const reconnectTimeoutRef = useRef<ReturnType<typeof setTimeout>>();
  
const store = useTrainingStore();
  
  // Create stable versions of store actions to avoid re-renders
  const setConnected = useRef(store.setConnected);
  const setConnectionStatus = useRef(store.setConnectionStatus);
  const addLog = useRef(store.addLog);
  const setTraining = useRef(store.setTraining);
  const setMetrics = useRef(store.setMetrics);
  const addMetricsHistory = useRef(store.addMetricsHistory);
  const setCubeState = useRef(store.setCubeState);
  const setScrambleMoves = useRef(store.setScrambleMoves);
  const setCubeStep = useRef(store.setCubeStep);

  useEffect(() => {
    setConnected.current = store.setConnected;
    setConnectionStatus.current = store.setConnectionStatus;
    addLog.current = store.addLog;
    setTraining.current = store.setTraining;
    setMetrics.current = store.setMetrics;
    addMetricsHistory.current = store.addMetricsHistory;
    setCubeState.current = store.setCubeState;
    setScrambleMoves.current = store.setScrambleMoves;
    setCubeStep.current = store.setCubeStep;
  }, [store]);

  const connect = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.CONNECTING || wsRef.current?.readyState === WebSocket.OPEN) {
      return;
    }

    const { port } = window.location;
    const apiHost = port === "3001" ? "localhost:8000" : window.location.host;
    const wsUrl = url || `ws://${apiHost}/ws`;
    
    console.log("Connecting to:", wsUrl);
    const ws = new WebSocket(wsUrl);
    wsRef.current = ws;
    
    setConnectionStatus.current("connecting");
    addLog.current("system", "Connecting to server...");
    
    ws.onopen = () => {
      setConnected.current(true);
      setConnectionStatus.current("connected");
      addLog.current("system", "Connected to server");
      ws.send(JSON.stringify({ type: "get_status", payload: {} }));
    };
    
    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        const { type, payload } = data;
        
        switch (type) {
          case "connected":
            addLog.current("system", "Connected to Python server");
            break;
          case "disconnected":
            addLog.current("warning", "Lost connection to Python server");
            break;
          case "training_started":
            setTraining.current(true);
            addLog.current("training", "Training started");
            break;
          case "training_stopped":
            setTraining.current(false);
            addLog.current("training", "Training stopped");
            break;
          case "status":
            setTraining.current(payload.status === "training");
            if (payload.metrics) {
              setMetrics.current(payload.metrics);
            }
            break;
          case "metrics_update":
            setMetrics.current(payload);
            addMetricsHistory.current(payload);
            break;
          case "adi_progress":
            addLog.current(
              "training",
              `AVI ${payload.adi_step}/${payload.adi_steps_total} - loss: ${payload.avg_loss.toFixed(4)} (depth: ${payload.scramble_depth})`
            );
            break;
          case "episode_complete":
            const solvedStr = payload.solved ? "✓ Solved" : "✗ Failed";
            addLog.current(
              "episode",
              `Episode ${payload.episode}: ${solvedStr} (reward: ${payload.reward.toFixed(2)}, steps: ${payload.steps})`
            );
            break;
          case "checkpoint_saved":
            addLog.current("checkpoint", `Checkpoint saved: ${payload.path}`);
            break;
          case "scramble_result":
            setCubeState.current(payload.state);
            setScrambleMoves.current(payload.moves);
            addLog.current("scramble", `Scrambled with moves: ${payload.moves.join(" ")}`);
            break;
          case "cube_step":
            setCubeStep.current(payload);
            if (payload.solved) {
              addLog.current("episode", `✅ Featured episode solved in ${payload.step} steps!`);
            }
            break;
          case "solve_result":
            addLog.current("solve", payload.solved ? "Solved!" : "Could not solve");
            break;
          case "error":
            addLog.current("error", payload.message || "Unknown error");
            break;
          default:
            break;
        }

        const handlers = handlersRef.current.get(type) || [];
        handlers.forEach(handler => handler(payload));
      } catch (e) {
        console.error("Failed to parse message:", e);
      }
    };
    
    ws.onclose = () => {
      setConnected.current(false);
      setConnectionStatus.current("disconnected");
      addLog.current("system", "Disconnected from server");
      
      reconnectTimeoutRef.current = setTimeout(connect, 3000);
    };
    
    ws.onerror = (error) => {
      setConnectionStatus.current("error");
      addLog.current("error", "Connection error check console");
      console.error("WebSocket error:", error);
    };
  }, [url]); // No 'store' dependency to prevent loop
  
  const send = useCallback((type: string, payload: any = {}) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ type, payload }));
    }
  }, []);
  
  const on = useCallback((event: string, handler: MessageHandler) => {
    if (!handlersRef.current.has(event)) {
      handlersRef.current.set(event, []);
    }
    handlersRef.current.get(event)!.push(handler);
    
    return () => {
      const handlers = handlersRef.current.get(event);
      if (handlers) {
        const index = handlers.indexOf(handler);
        if (index > -1) handlers.splice(index, 1);
      }
    };
  }, []);
  
  useEffect(() => {
    connect();
    
    return () => {
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current);
      }
      wsRef.current?.close();
    };
  }, [connect]);
  
  return { send, on, isConnected: store.isConnected };
}
