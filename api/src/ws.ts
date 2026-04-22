import { PythonClient } from "./lib/python-client";
import { ServerWebSocket } from "bun";
import type { TrainingMetrics, EpisodeResult, ScrambleResult, CubeStepResult } from "./types";

interface WsData {
  clientId: string;
}

export function createWebSocketHandler(pythonClient: PythonClient) {
  const clients = new Set<ServerWebSocket<WsData>>();
  
  // Forward Python server messages to all connected frontend clients
  pythonClient.on("metrics_update", (data: TrainingMetrics) => {
    broadcast(clients, { type: "metrics_update", payload: data });
  });

  pythonClient.on("episode_complete", (data: EpisodeResult) => {
    broadcast(clients, { type: "episode_complete", payload: data });
  });

  pythonClient.on("checkpoint_saved", (data: { path: string }) => {
    broadcast(clients, { type: "checkpoint_saved", payload: data });
  });

  pythonClient.on("training_started", (data: TrainingMetrics) => {
    broadcast(clients, { type: "training_started", payload: data });
  });

  pythonClient.on("training_stopped", (data: TrainingMetrics) => {
    broadcast(clients, { type: "training_stopped", payload: data });
  });

  pythonClient.on("scramble_result", (data: ScrambleResult) => {
    broadcast(clients, { type: "scramble_result", payload: data });
  });

  pythonClient.on("cube_step", (data: CubeStepResult) => {
    broadcast(clients, { type: "cube_step", payload: data });
  });

  pythonClient.on("adi_progress", (data: any) => {
    broadcast(clients, { type: "adi_progress", payload: data });
  });

  pythonClient.on("solve_result", (data: any) => {
    broadcast(clients, { type: "solve_result", payload: data });
  });

  pythonClient.on("status", (data: any) => {
    broadcast(clients, { type: "status", payload: data });
  });

  pythonClient.on("error", (data: any) => {
    broadcast(clients, { type: "error", payload: data });
  });

  return {
    open(ws: ServerWebSocket<WsData>) {
      const clientId = Math.random().toString(36).slice(2, 9);
      ws.data = { clientId };
      clients.add(ws);
      console.log(`Frontend client connected: ${clientId}`);
    },
    
    message(ws: ServerWebSocket<WsData>, message: string | Buffer) {
      try {
        const data = JSON.parse(message.toString());
        handleClientMessage(data, pythonClient, ws);
      } catch (e) {
        ws.send(JSON.stringify({ type: "error", payload: { message: "Invalid JSON" } }));
      }
    },
    
    close(ws: ServerWebSocket<WsData>) {
      clients.delete(ws);
      console.log(`Frontend client disconnected: ${ws.data.clientId}`);
    },
  };
}

function handleClientMessage(
  data: { type: string; payload: any },
  pythonClient: PythonClient,
  ws: ServerWebSocket<WsData>
) {
  const { type, payload } = data;
  
  switch (type) {
    case "ping":
      ws.send(JSON.stringify({ type: "pong", payload: {} }));
      break;
    case "start_training":
      pythonClient.send("start_training", payload);
      break;
    case "stop_training":
      pythonClient.send("stop_training", {});
      break;
    case "get_status":
      pythonClient.send("get_status", {});
      break;
    case "update_config":
      pythonClient.send("update_config", payload);
      break;
    case "scramble":
      pythonClient.send("scramble", payload);
      break;
    case "solve":
      pythonClient.send("solve", {});
      break;
    default:
      ws.send(JSON.stringify({ type: "error", payload: { message: `Unknown type: ${type}` } }));
  }
}

function broadcast(clients: Set<ServerWebSocket<WsData>>, message: any) {
  const data = JSON.stringify(message);
  for (const client of clients) {
    try {
      client.send(data);
    } catch (e) {
      // Client disconnected
    }
  }
}
