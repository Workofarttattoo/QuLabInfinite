import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";

export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    // 5173 — avoid clash with Grafana (docker-compose maps grafana to host :3000)
    port: 5173,
    proxy: {
      "/mcp": {
        target: "http://127.0.0.1:8102",
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/mcp/, ""),
      },
    },
  },
  preview: {
    port: 5173,
    proxy: {
      "/mcp": {
        target: "http://127.0.0.1:8102",
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/mcp/, ""),
      },
    },
  },
});
