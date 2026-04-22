#!/bin/bash
# Syncs the necessary codebase to your Vast.ai instance
# We use 'rsync' as it is a smarter version of 'scp' that can exclude massive folders like node_modules

VAST_HOST="${VAST_HOST:-ssh8.vast.ai}"
VAST_PORT="${VAST_PORT:-12188}"
VAST_USER="${VAST_USER:-root}"
TARGET_DIR="${TARGET_DIR:-~/rubiks-cube-rl}"

echo "🚀 Syncing code to Vast.ai instance ($VAST_USER@$VAST_HOST:$VAST_PORT)..."

# Ensure the target directory exists
ssh -p "$VAST_PORT" "$VAST_USER@$VAST_HOST" "mkdir -p $TARGET_DIR"

# Rsync copies all files while ignoring big dependencies/checkpoints
rsync -avz -e "ssh -p $VAST_PORT" \
    --exclude 'node_modules' \
    --exclude '.venv' \
    --exclude '.git' \
    --exclude '__pycache__' \
    --exclude 'checkpoints' \
    --exclude '*.pt' \
    --exclude '.DS_Store' \
    ./ "$VAST_USER@$VAST_HOST:$TARGET_DIR/"

echo "✅ Sync complete!"
echo ""
echo "To connect and start training, run:"
echo "ssh -L 3000:localhost:3000 -L 8000:localhost:8000 -p $VAST_PORT $VAST_USER@$VAST_HOST"
echo "cd $TARGET_DIR && ./scripts/vast_setup.sh"
