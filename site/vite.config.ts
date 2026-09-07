import { resolve } from "node:path";
import { defineConfig } from "vite";

export default defineConfig({
  root: resolve(import.meta.dirname),
  base: "/regressio/",
  build: {
    outDir: resolve(import.meta.dirname, "../site-dist"),
    emptyOutDir: true,
    sourcemap: true,
  },
  worker: {
    format: "es",
  },
  server: {
    host: "127.0.0.1",
  },
});
