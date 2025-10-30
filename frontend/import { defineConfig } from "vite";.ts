import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// keep it ultra-light; no extra plugins
export default defineConfig({
  plugins: [react()],
  server: { port: 5173 },
});