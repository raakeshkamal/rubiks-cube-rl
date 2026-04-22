import React from 'react'
import ReactDOM from 'react-dom/client'
import App from './App.tsx'

console.log("🚀 Rubik's Cube RL Trainer starting...");

try {
  const root = document.getElementById('root');
  if (!root) throw new Error("Could not find root element");
  
  ReactDOM.createRoot(root).render(
    <React.StrictMode>
      <App />
    </React.StrictMode>
  );
  console.log("✅ Root rendered successfully");
} catch (err) {
  console.error("💥 FAILED TO RENDER:", err);
  document.body.innerHTML = `<div style="color: red; padding: 20px;"><h1>Render Crash</h1><pre>${err instanceof Error ? err.stack : String(err)}</pre></div>`;
}
