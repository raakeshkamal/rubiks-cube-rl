import { Hono } from "hono";
import { serve } from "bun";
import { createRouter } from "./router";
import { createWebSocketHandler } from "./ws";
import { PythonClient } from "./lib/python-client";

// Python training server connection
const PYTHON_HOST = process.env.PYTHON_HOST || "localhost";
const PYTHON_PORT = parseInt(process.env.PYTHON_PORT || "8001");

// API server port
const API_PORT = parseInt(process.env.API_PORT || "8000");

async function main() {
  const pythonClient = new PythonClient(PYTHON_HOST, PYTHON_PORT);

  // Connect to Python server (non-blocking)
  pythonClient.connect().catch((err) => {
    console.warn(`Could not connect to Python server: ${err.message}`);
    console.warn("Will retry automatically...");
  });

  // Create HTTP router
  const app = createRouter(pythonClient);

  // Create WebSocket handler
  const wsHandler = createWebSocketHandler(pythonClient);

  // Start server
  serve({
    port: API_PORT,
    fetch(req, server) {
      // WebSocket upgrade
      const url = new URL(req.url);
      if (url.pathname === "/ws") {
        const upgraded = server.upgrade(req, {
          data: { clientId: "" } as { clientId: string },
        });
        if (!upgraded) {
          return new Response("WebSocket upgrade failed", { status: 500 });
        }
        return;
      }

      // HTTP routes
      return app.fetch(req);
    },
    websocket: wsHandler,
  });

  console.log(`🚀 API server running on http://localhost:${API_PORT}`);
  console.log(`📡 WebSocket available at ws://localhost:${API_PORT}/ws`);
  console.log(`🐍 Connecting to Python server at ws://${PYTHON_HOST}:${PYTHON_PORT}`);
}

main().catch(console.error);
