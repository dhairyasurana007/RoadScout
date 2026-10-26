import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    strictPort: true,
    // Send API calls to the FastAPI backend (uvicorn on port 8000).
    proxy: { "/api": "http://localhost:8000" },
  },
});
