import asyncio
import json
import os
import glob
import websockets
from typing import Optional

from rl.trainer import Trainer, TrainingConfig
from cube.simulator import Cube3State, CubeSimulator


class TrainingServer:
    """WebSocket server for communication with Bun API."""
    
    def __init__(self, host: str = "localhost", port: int = 8001):
        self.host = host
        self.port = port
        self.trainer: Optional[Trainer] = None
        self.clients = set()
        self.last_scramble_face_state: Optional[list[int]] = None
        self.last_scramble_sticker_state = None
    
    async def handler(self, websocket):
        """Handle incoming WebSocket connections."""
        try:
            # The connection is only fully established here
            self.clients.add(websocket)
            print(f"New WebSocket session established. Total clients: {len(self.clients)}")
            
            async for message in websocket:
                try:
                    data = json.loads(message)
                    await self._handle_message(data, websocket)
                except json.JSONDecodeError:
                    await websocket.send(json.dumps({
                        "type": "error",
                        "payload": {"message": "Invalid JSON"}
                    }))
        except websockets.exceptions.ConnectionClosed:
            pass
        finally:
            if websocket in self.clients:
                self.clients.remove(websocket)
            print(f"WebSocket session closed. Total clients: {len(self.clients)}")
    
    async def _handle_message(self, data: dict, websocket):
        """Route messages to handlers."""
        msg_type = data.get("type")
        payload = data.get("payload", {})
        
        handlers = {
            "ping": self._handle_ping,
            "start_training": self._handle_start,
            "stop_training": self._handle_stop,
            "get_status": self._handle_status,
            "update_config": self._handle_config_update,
            "scramble": self._handle_scramble,
            "solve": self._handle_solve,
        }
        
        handler = handlers.get(msg_type)
        if handler:
            await handler(payload, websocket)
        else:
            await websocket.send(json.dumps({
                "type": "error",
                "payload": {"message": f"Unknown message type: {msg_type}"}
            }))
    
    async def _handle_ping(self, payload: dict, websocket):
        await websocket.send(json.dumps({"type": "pong", "payload": {}}))
    
    async def _handle_start(self, payload: dict, websocket):
        if self.trainer and self.trainer.is_running:
            await websocket.send(json.dumps({
                "type": "error",
                "payload": {"message": "Training already running"}
            }))
            return
        
        # Create trainer with config from payload
        config = TrainingConfig(**payload.get("config", {}))
        self.trainer = Trainer(config)
        
        resume = payload.get("resume", False)
        if resume:
            checkpoint_dirs = [
                os.path.join(config.artifact_root, "checkpoints"),
                "checkpoints",
                "backend/checkpoints",
            ]
            checkpoint_dir = next((path for path in checkpoint_dirs if os.path.exists(path)), None)

            if checkpoint_dir:
                checkpoints = glob.glob(os.path.join(checkpoint_dir, "epoch_*.pt")) or glob.glob(
                    os.path.join(checkpoint_dir, "episode_*.pt")
                )
                if checkpoints:
                    import re

                    def get_ep_num(f):
                        m = re.search(r'(?:epoch|episode)_(\d+)\.pt', f)
                        return int(m.group(1)) if m else -1

                    latest_checkpoint = max(checkpoints, key=get_ep_num)
                    print(f"Resuming from checkpoint: {latest_checkpoint}")
                    try:
                        self.trainer.load_checkpoint(latest_checkpoint)
                    except Exception as e:
                        print(f"Failed to load checkpoint {latest_checkpoint}: {e}")
                else:
                    print(f"No checkpoints found in {checkpoint_dir}")

        await websocket.send(json.dumps({
            "type": "training_started",
            "payload": self.trainer.get_metrics_dict()
        }))
        
        # Run training loop
        async def on_episode(data):
            await self._broadcast({
                "type": "episode_complete",
                "payload": data
            })
        
        async def on_checkpoint(path):
            await self._broadcast({
                "type": "checkpoint_saved",
                "payload": {"path": path}
            })
        
        # Run training in background
        asyncio.create_task(self._run_training_loop(on_episode, on_checkpoint))
    
    async def _run_training_loop(self, on_episode, on_checkpoint):
        """Run training epochs until stopped."""
        async def on_eval_step_callback(state, move, step_num, solved, dist):
            await self._broadcast({
                "type": "cube_step",
                "payload": {
                    "state": state.tolist(),
                    "move": move,
                    "step": int(step_num),
                    "solved": bool(solved),
                    "distance": int(dist),
                }
            })

        def on_adi_progress_callback(data):
            asyncio.ensure_future(self._broadcast({
                "type": "adi_progress",
                "payload": data,
            }))

        while self.trainer and not self.trainer.should_stop:
            metrics = await self.trainer.run_epoch(
                on_episode=lambda d: asyncio.ensure_future(on_episode(d)),
                on_checkpoint=lambda p: asyncio.ensure_future(on_checkpoint(p)),
                on_adi_progress=on_adi_progress_callback,
                on_eval_step=on_eval_step_callback,
            )
            # Send metrics update
            await self._broadcast({
                "type": "metrics_update",
                "payload": self.trainer.get_metrics_dict()
            })

            await asyncio.sleep(0.01)
    
    async def _handle_stop(self, payload: dict, websocket):
        if self.trainer:
            print("Stopping trainer...")
            self.trainer.stop()
            metrics = self.trainer.get_metrics_dict()
            # self.trainer = None # Keep for metrics access but it will stop
            await self._broadcast({
                "type": "training_stopped",
                "payload": metrics
            })
        else:
            await websocket.send(json.dumps({
                "type": "error",
                "payload": {"message": "No training in progress"}
            }))
    
    async def _handle_status(self, payload: dict, websocket):
        if self.trainer:
            status = "training" if not self.trainer.should_stop else "idle"
            metrics = self.trainer.get_metrics_dict()
        else:
            status = "idle"
            metrics = {}
        
        await websocket.send(json.dumps({
            "type": "status",
            "payload": {
                "status": status,
                "metrics": metrics,
                "config": self.trainer.config.__dict__ if self.trainer else {}
            }
        }))
    
    async def _handle_config_update(self, payload: dict, websocket):
        if self.trainer:
            for key, value in payload.items():
                if hasattr(self.trainer.config, key):
                    setattr(self.trainer.config, key, value)
            await websocket.send(json.dumps({
                "type": "config_updated",
                "payload": self.trainer.config.__dict__
            }))
    
    async def _handle_scramble(self, payload: dict, websocket):
        depth = payload.get("depth", 10)
        cube = CubeSimulator()
        state, moves = cube.scramble(depth)
        self.last_scramble_face_state = state.tolist()
        self.last_scramble_sticker_state = cube.sticker_state.copy()
        
        await websocket.send(json.dumps({
            "type": "scramble_result",
            "payload": {
                "state": state.tolist(),
                "moves": moves,
                "depth": depth,
            }
        }))
    
    async def _handle_solve(self, payload: dict, websocket):
        trainer = self.trainer or Trainer(TrainingConfig())
        self.trainer = trainer

        if self.last_scramble_sticker_state is None:
            cube = CubeSimulator()
            state, _ = cube.scramble(payload.get("depth", 5))
            self.last_scramble_face_state = state.tolist()
            self.last_scramble_sticker_state = cube.sticker_state.copy()

        result = trainer.agent.solve(Cube3State(self.last_scramble_sticker_state.copy()), trainer.config.max_search_nodes, 1)

        if result.path_states:
            solved_face_state = [idx // 9 for idx in range(54)]
            for step_idx, state in enumerate(result.path_states[1:], start=1):
                await self._broadcast({
                    "type": "cube_step",
                    "payload": {
                        "state": state.tolist(),
                        "move": result.moves[step_idx - 1],
                        "step": step_idx,
                        "solved": step_idx == len(result.moves) and result.solved,
                        "distance": int(sum(int(state[idx] != solved_face_state[idx]) for idx in range(len(state)))),
                    }
                })

        await websocket.send(json.dumps({
            "type": "solve_result",
            "payload": {
                "solved": result.solved,
                "moves": result.moves,
                "cost": result.cost,
                "nodes_generated": result.nodes_generated,
            }
        }))
    
    async def _broadcast(self, message: dict):
        """Send message to all connected clients."""
        data = json.dumps(message)
        for client in self.clients:
            try:
                await client.send(data)
            except websockets.exceptions.ConnectionClosed:
                pass
    
    async def start(self):
        """Start the WebSocket server."""
        print(f"Starting training server on ws://{self.host}:{self.port}")
        # Using a higher-level serve that handles handshake failures better
        async with websockets.serve(
            self.handler, 
            self.host, 
            self.port,
        ):
            await asyncio.Future()  # Run forever
