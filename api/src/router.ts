import { Hono } from "hono";
import { cors } from "hono/cors";
import { PythonClient } from "./lib/python-client";

export function createRouter(pythonClient: PythonClient) {
  const app = new Hono();

  // CORS
  app.use("/*", cors());

  // Health check
  app.get("/api/health", (c) => c.json({ status: "ok" }));

  // Training endpoints
  app.post("/api/training/start", async (c) => {
    const body = await c.req.json();
    await pythonClient.send("start_training", { config: body });
    return c.json({ message: "Training started" });
  });

  app.post("/api/training/stop", async (c) => {
    await pythonClient.send("stop_training", {});
    return c.json({ message: "Training stopped" });
  });

  app.get("/api/training/status", async (c) => {
    await pythonClient.send("get_status", {});
    return c.json({ message: "Status requested" });
  });

  app.post("/api/training/config", async (c) => {
    const body = await c.req.json();
    await pythonClient.send("update_config", body);
    return c.json({ message: "Config updated" });
  });

  // Cube endpoints
  app.get("/api/cube/scramble", async (c) => {
    const depth = parseInt(c.req.query("depth") || "10");
    await pythonClient.send("scramble", { depth });
    return c.json({ message: "Scramble requested" });
  });

  app.post("/api/cube/solve", async (c) => {
    await pythonClient.send("solve", {});
    return c.json({ message: "Solve requested" });
  });

  // Move list
  app.get("/api/moves", (c) => {
    return c.json({
      moves: [
        "U", "U'", "D", "D'",
        "L", "L'", "R", "R'",
        "F", "F'", "B", "B'"
      ]
    });
  });

  return app;
}
