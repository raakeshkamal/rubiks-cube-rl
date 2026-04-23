#!/usr/bin/env bash
# Sync checkpoints from Vast.ai instance to local machine every 10 minutes

REMOTE_HOST="root@ssh2.vast.ai"
REMOTE_PORT="12288"
REMOTE_DIR="~/rubiks-cube-rl/backend/saved_models/"
LOCAL_DIR="/Users/raakeshkamal/Documents/rubiks-cube-rl/backend/saved_models/"
SSH_OPTS="-p ${REMOTE_PORT} -o ConnectTimeout=30 -o StrictHostKeyChecking=no"

mkdir -p "${LOCAL_DIR}"

echo "$(date '+%Y-%m-%d %H:%M:%S') — Starting checkpoint sync loop"
echo "Remote: ${REMOTE_HOST}:${REMOTE_PORT}:${REMOTE_DIR}"
echo "Local:  ${LOCAL_DIR}"
echo "Press Ctrl+C to stop"
echo ""

while true; do
    echo "$(date '+%Y-%m-%d %H:%M:%S') — Syncing..."

    rsync -avz --delete \
        -e "ssh ${SSH_OPTS}" \
        "${REMOTE_HOST}:${REMOTE_DIR}" \
        "${LOCAL_DIR}"

    status=$?
    if [ $status -eq 0 ]; then
        echo "$(date '+%Y-%m-%d %H:%M:%S') — OK"
    else
        echo "$(date '+%Y-%m-%d %H:%M:%S') — FAILED (exit code: $status, will retry in 10min)"
    fi

    echo "$(date '+%Y-%m-%d %H:%M:%S') — Next sync in 10 minutes."
    echo ""
    sleep 600
done
