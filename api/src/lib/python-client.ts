// Use common global WebSocket
// import { WebSocket } from "ws";

type MessageHandler = (message: any) => void;

export class PythonClient {
  private ws: any | null = null;
  private handlers: Map<string, MessageHandler[]> = new Map();
  private reconnectAttempts = 0;
  private maxReconnectAttempts = 10;
  private reconnectDelay = 2000;

  constructor(private host: string = "localhost", private port: number = 8001) {}

  connect(): Promise<void> {
    return new Promise((resolve, reject) => {
      const url = `ws://${this.host}:${this.port}`;
      console.log(`Connecting to Python server at ${url}`);

      try {
        // Use global WebSocket (Native in Bun)
        this.ws = new WebSocket(url);

        this.ws.onopen = () => {
          console.log("✅ Successfully established WebSocket connection with Python");
          this.reconnectAttempts = 0;
          this.emit("connected", {});
          resolve();
        };

        this.ws.onmessage = (event: MessageEvent) => {
          try {
            const message = JSON.parse(event.data.toString());
            this.emit(message.type, message.payload);
          } catch (e) {
            console.error("Failed to parse message from Python:", e);
          }
        };

        this.ws.onclose = (event: CloseEvent) => {
          console.log(`Disconnected from Python server (Code: ${event.code})`);
          this.emit("disconnected", {});
          this.attemptReconnect();
        };

        this.ws.onerror = (error: any) => {
          console.error("WebSocket connection error");
          // Don't reject if we are already connected, handled by onclose
          if (this.reconnectAttempts === 0) reject(error);
        };
      } catch (err) {
        reject(err);
      }
    });
  }

  private attemptReconnect() {
    if (this.reconnectAttempts < this.maxReconnectAttempts) {
      this.reconnectAttempts++;
      console.log(`📡 Reconnecting... (${this.reconnectAttempts}/${this.maxReconnectAttempts}) in ${this.reconnectDelay}ms`);
      setTimeout(() => {
        this.connect().catch(() => {});
      }, this.reconnectDelay);
    }
  }

  send(type: string, payload: any): void {
    if (!this.ws || this.ws.readyState !== 1 /* OPEN */) {
      console.warn(`Cannot send '${type}': Python server not ready (ReadyState: ${this.ws?.readyState})`);
      return;
    }
    this.ws.send(JSON.stringify({ type, payload }));
  }

  on(event: string, handler: MessageHandler): () => void {
    if (!this.handlers.has(event)) {
      this.handlers.set(event, []);
    }
    this.handlers.get(event)!.push(handler);
    
    // Return unsubscribe function
    return () => {
      const handlers = this.handlers.get(event);
      if (handlers) {
        const index = handlers.indexOf(handler);
        if (index > -1) handlers.splice(index, 1);
      }
    };
  }

  private emit(event: string, payload: any): void {
    const handlers = this.handlers.get(event) || [];
    handlers.forEach(handler => handler(payload));
  }

  disconnect(): void {
    if (this.ws) {
      this.ws.close();
      this.ws = null;
    }
  }
}
