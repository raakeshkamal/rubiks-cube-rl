"""Main entry point for the RL training server."""
import asyncio
import sys
import os
import logging

# Set up logging to be less noisy with handshake errors
logging.basicConfig(level=logging.INFO)
logging.getLogger('websockets').setLevel(logging.ERROR)

# Add src directory to path
sys.path.insert(0, os.path.dirname(__file__))

from server.ipc import TrainingServer

async def main():
    host = os.environ.get("TRAINING_HOST", "127.0.0.1")
    port = int(os.environ.get("TRAINING_PORT", "8001"))
    
    server = TrainingServer(host=host, port=port)
    await server.start()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
