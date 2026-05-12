import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import { resolve } from "path";
import { fileURLToPath } from "url";

const __dirname = fileURLToPath(new URL(".", import.meta.url));

export default defineConfig({
  plugins: [react()],
  // Browser IIFE must not reference Node's `process` (some deps do); otherwise the
  // bundle throws before `NyckChatMd` is defined and the UI falls back to plain text.
  define: {
    "process.env.NODE_ENV": JSON.stringify("production"),
  },
  build: {
    lib: {
      entry: resolve(__dirname, "src/bridge.jsx"),
      name: "NyckChatMd",
      formats: ["iife"],
      fileName: () => "assistant-md.js",
    },
    outDir: "build",
    emptyOutDir: true,
  },
});
