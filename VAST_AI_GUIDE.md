# Deploying to Vast.ai

To deploy this project to the cloud and take advantage of much stronger desktop GPUs (like the RTX 4080), the easiest path is to use **Vast.ai** and SSH tunneling.

## 1. Setup the GPU Instance

The **RTX 4080 (16GB)** is highly recommended. It offers excellent PyTorch calculation rates per dollar. The **RTX A4000 (16GB)** is a strong secondary option.

You can rent a machine from Vast.ai by using the command line tool or their Web UI. Choose an **Ubuntu PyTorch CUDA** base image with SSH access.

## 2. Deploy Code to Instance

Connect via SSH, and copy the repository to the instance (or `git clone` if it's on GitHub).

### Sync from local machine

Set your instance details as environment variables, then run the sync script:

```bash
export VAST_HOST="ssh8.vast.ai"
export VAST_PORT="12188"
export VAST_USER="root"
export TARGET_DIR="~/rubiks-cube-rl"

./scripts/sync_to_vast.sh
```

This uses `rsync` to copy everything except `node_modules`, `.venv`, `checkpoints`, and `.git`.

### Provision the instance

Once synced (or cloned), SSH into the instance and run:

```bash
cd ~/rubiks-cube-rl
./scripts/vast_setup.sh
```

This installs Bun, `uv`, Python dependencies, and verifies CUDA is accessible.

## 3. Start Training

The easiest way is to use the unified cloud startup script:

```bash
cd ~/rubiks-cube-rl
bun run start:cloud
```

This starts all three services in the background:
- Python training server (`ws://127.0.0.1:8001`)
- Bun API gateway (`http://localhost:8000`)
- Vite frontend (`http://localhost:3000`)

Press `Ctrl+C` to stop all services cleanly.

### Manual startup (alternative)

If you prefer separate terminals:

```bash
# Terminal 1
cd backend && uv run python src/main.py

# Terminal 2
bun dev:api

# Terminal 3
bun dev:frontend
```

## 4. Remote Observation Setup

We want the API and frontend to think they are running locally so we don't need to struggle with CORS issues, websocket configuration across proxies, or exposing ports transparently to the internet.

From your **local laptop**, open an SSH tunnel:

```bash
ssh -L 3000:localhost:3000 -L 8000:localhost:8000 -p $VAST_PORT $VAST_USER@$VAST_HOST
```

Leave this SSH connection open while you observe the project.

Now, navigate to `http://localhost:3000` on your laptop's browser. You will see the training dashboard pulling live WebSocket data from the Vast.ai GPU instance.

## 5. Cloud Tuning

These environment variables let you tune performance for your specific GPU without touching code:

| Variable | Default | Description |
|---|---|---|
| `HEURISTIC_BATCH_SIZE` | `32768` | Batch size for neural network inference during A* search. Increase if you have extra VRAM; decrease if you hit OOM. |
| `USE_AMP` | `0` | Set to `1` to enable **Automatic Mixed Precision** (fp16). Gives ~1.5-2× training speedup and ~40% VRAM savings on Ampere/Ada GPUs. |
| `USE_TORCH_COMPILE` | `0` | Set to `1` to enable `torch.compile()` for extra inference/training speedup. May be unstable on some PyTorch versions. |
| `TRAINING_HOST` | `127.0.0.1` | Host for the Python training WebSocket server. |
| `TRAINING_PORT` | `8001` | Port for the Python training WebSocket server. |
| `API_PORT` | `8000` | Port for the Bun API gateway. |

Example with all optimizations enabled:

```bash
export HEURISTIC_BATCH_SIZE=65536
export USE_AMP=1
export USE_TORCH_COMPILE=1
bun run start:cloud
```

## 6. Resuming Training

Checkpoints are saved automatically to `saved_models/cube3/checkpoints/`. When you click **Start Training** in the UI with **Resume** enabled, the latest checkpoint is loaded automatically.

To copy checkpoints back to your local machine:

```bash
rsync -avz -e "ssh -p $VAST_PORT" \
    "$VAST_USER@$VAST_HOST:~/rubiks-cube-rl/saved_models/" ./saved_models/
```
