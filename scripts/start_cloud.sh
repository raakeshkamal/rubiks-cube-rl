#!/bin/bash
# Unified cloud startup script for Vast.ai / NVIDIA GPU instances
# Starts: Python training server -> Bun API -> Vite frontend
set -e

echo "🚀 Rubik's Cube RL — Cloud Startup"
echo "==================================="

# Verify CUDA
if command -v nvidia-smi &> /dev/null; then
    echo "✅ CUDA detected:"
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
else
    echo "⚠️  nvidia-smi not found. CUDA may not be available."
fi

# Environment defaults
export TRAINING_HOST="${TRAINING_HOST:-127.0.0.1}"
export TRAINING_PORT="${TRAINING_PORT:-8001}"
export API_PORT="${API_PORT:-8000}"

# Trap to kill child processes on exit
cleanup() {
    echo ""
    echo "🛑 Shutting down services..."
    jobs -p | xargs -r kill 2>/dev/null || true
    wait
    echo "✅ All services stopped."
}
trap cleanup SIGINT SIGTERM EXIT

# Start Python training server
echo ""
echo "🐍 Starting Python training server on ws://$TRAINING_HOST:$TRAINING_PORT..."
cd backend
uv run python src/main.py &
TRAINING_PID=$!
cd ..

# Wait for Python server to be ready
for i in {1..30}; do
    if python3 -c "import socket; s=socket.socket(); s.settimeout(1); s.connect(('$TRAINING_HOST',$TRAINING_PORT)); s.close()" 2>/dev/null; then
        echo "✅ Python server ready."
        break
    fi
    sleep 1
    if [ $i -eq 30 ]; then
        echo "⚠️  Python server did not start in time. Check logs above."
        exit 1
    fi
done

# Start Bun API gateway
echo ""
echo "🚀 Starting Bun API gateway on http://localhost:$API_PORT..."
cd api
bun run dev &
API_PID=$!
cd ..

# Start Vite frontend
echo ""
echo "⚛️  Starting Vite frontend on http://localhost:3000..."
cd frontend
bun run dev &
FRONTEND_PID=$!
cd ..

echo ""
echo "==================================="
echo "All services are running!"
echo ""
echo "  Frontend:   http://localhost:3000"
echo "  API:        http://localhost:$API_PORT"
echo "  Training:   ws://$TRAINING_HOST:$TRAINING_PORT"
echo ""
echo "Press Ctrl+C to stop all services."
echo "==================================="

# Wait for all background jobs
wait
