# 🧊 Rubik's Cube RL Trainer

Deep Q-Network training system for solving the Rubik's Cube with real-time visualization.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    FRONTEND (Bun + Vite + React)                            │
│  Cube Visualizer  │  Training Dashboard  │  Metrics Charts  │  Logs       │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │ WebSocket
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    API GATEWAY (Bun + Hono)                                 │
│  REST API  │  WebSocket Handler  │  IPC Bridge to Python                  │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │ IPC
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    RL TRAINING (Python + uv + PyTorch CUDA)                  │
│  PyTorch CNN  │  NumPy Cube Simulator  │  DQN Agent  │  Training Loop     │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Tech Stack

### Frontend
- **Bun** - Package manager & runtime
- **Vite** - Build tool
- **React 19** - UI framework
- **Zustand** - State management
- **Recharts** - Charts
- **Hono** - HTTP framework

### Backend (RL Training)
- **Python 3.11+** - Training runtime
- **uv** - Package manager
- **PyTorch** - Neural network with CUDA (NVIDIA GPU)
- **NumPy** - Cube simulation
- **WebSockets** - IPC communication

## Quick Start

### Cloud Training (Recommended)

Train on 16GB NVIDIA GPUs via **[Vast.ai](VAST_AI_GUIDE.md)** for best performance.

### Local Prerequisites
- Bun installed
- Python 3.11+ with uv installed
- macOS (MPS fallback) or Linux (CUDA)

### 1. Install Dependencies

```bash
# Install Bun dependencies
bun install

# Install Python dependencies
cd backend && uv sync
```

### 2. Start Services

**Option A — All-in-one (Cloud / Vast.ai)**
```bash
bun run start:cloud
```

**Option B — Separate terminals**

Terminal 1 — Training Server (Python)
```bash
cd backend && uv run python src/main.py
```

Terminal 2 — API Server (Bun)
```bash
bun dev:api
```

Terminal 3 — Frontend (Vite)
```bash
bun dev:frontend
```

Or run API + frontend together:
```bash
bun dev
```

### 3. Open Browser

Navigate to `http://localhost:3000`

## Features

- **Real-time Training**: Watch the agent learn in real-time
- **CUDA Acceleration**: Uses NVIDIA GPU for faster training
- **Mixed Precision (AMP)**: Optional fp16 training for 1.5-2× speedup on modern GPUs
- **Interactive Controls**: Start/stop training, adjust hyperparameters
- **Live Metrics**: Charts showing solve rate, reward, loss
- **Cube Visualization**: 2D unfolded cube state display
- **Training Logs**: Detailed training progress logging

## Configuration

Training hyperparameters can be adjusted in the frontend dashboard or by modifying `TrainingConfig` in `backend/src/rl/trainer.py`.

Key parameters:
- `scramble_depth`: Initial scramble difficulty (starts easy, increases via curriculum)
- `hidden_size`: Neural network width (default 5000, matching DeepCubeA)
- `learning_rate`: Adam optimizer learning rate
- `batch_size`: Training batch size (default 10,000 for GPU)
- `train_epochs_per_update`: ADI iterations per update (default 100)
- `loss_thresh`: Curriculum threshold (default 0.05)

Environment variables for cloud tuning:
- `HEURISTIC_BATCH_SIZE` — heuristic inference batch size during search (default 32768)
- `USE_AMP=1` — enable automatic mixed precision (fp16) training
- `USE_TORCH_COMPILE=1` — enable `torch.compile()` for extra speedup

## Project Structure

```
rubiks-cube-rl/
├── api/                        # Bun API server
│   ├── src/
│   │   ├── index.ts           # Entry point
│   │   ├── router.ts          # Hono REST routes
│   │   ├── ws.ts              # WebSocket handler
│   │   ├── types.ts           # TypeScript types
│   │   └── lib/
│   │       └── python-client.ts  # IPC to Python
│   ├── package.json
│   └── tsconfig.json
├── backend/                    # Python RL training
│   ├── src/
│   │   ├── main.py            # Entry point
│   │   ├── cube/
│   │   │   └── simulator.py   # NumPy cube simulator
│   │   ├── rl/
│   │   │   ├── network.py     # PyTorch CNN (CUDA + MPS support)
│   │   │   ├── agent.py       # DQN agent with AMP
│   │   │   └── trainer.py     # Training loop
│   │   └── server/
│   │       └── ipc.py         # WebSocket IPC server
│   ├── pyproject.toml
│   └── uv.lock
├── frontend/                   # React frontend
│   ├── src/
│   │   ├── main.tsx
│   │   ├── App.tsx
│   │   ├── App.css
│   │   ├── api/
│   │   │   └── client.ts      # WebSocket hook
│   │   ├── components/
│   │   │   ├── CubeVisualizer.tsx
│   │   │   ├── TrainingDashboard.tsx
│   │   │   ├── MetricsChart.tsx
│   │   │   └── TrainingLogs.tsx
│   │   ├── store/
│   │   │   └── training.ts    # Zustand store
│   │   └── types/
│   │       └── api.ts
│   ├── index.html
│   ├── package.json
│   ├── tsconfig.json
│   └── vite.config.ts
├── package.json                # Bun workspace root
└── README.md
```

## License

MIT
