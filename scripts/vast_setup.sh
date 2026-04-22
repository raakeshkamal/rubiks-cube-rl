#!/bin/bash
# fast-deploy script for Vast.ai Ubuntu instances
# Run this on your vast.ai instance to setup everything
set -e

echo "🚀 Installing dependencies..."
sudo apt-get update && sudo apt-get install -y curl git unzip

echo "🚀 Installing Bun..."
curl -fsSL https://bun.sh/install | bash
export BUN_INSTALL="$HOME/.bun"
export PATH="$BUN_INSTALL/bin:$PATH"

echo "🚀 Installing uv..."
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.cargo/env

echo "🚀 Checking out repo & installing deps..."
# Assumes repository has been cloned via SCP or git clone into ~/rubiks-cube-rl
if [ ! -d "backend" ]; then
    echo "⚠️  Please run this script from the rubiks-cube-rl directory."
    exit 1
fi

bun install
cd backend
export UV_PYTHON_DOWNLOADS=1
uv sync

echo "🚀 Verifying CUDA..."
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi
    python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"
else
    echo "⚠️  nvidia-smi not found. Ensure you selected a CUDA-enabled image on Vast.ai."
    exit 1
fi

echo "✅ Setup complete! You can now run the backend with:"
echo "cd backend && uv run python src/main.py"
echo ""
echo "Or start everything at once with:"
echo "bun run start:cloud"
