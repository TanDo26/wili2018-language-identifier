import { defineConfig, loadEnv } from 'vite';
import react from '@vitejs/plugin-react';

// Dev proxy: the browser calls /api/... on the Vite origin and Vite forwards it
// to the backend (FastAPI on :8000) or a Prism mock (e.g. http://localhost:4010).
// Override with:  VITE_API_TARGET=http://localhost:4010 npm run dev
export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), '');
  const target = env.VITE_API_TARGET || 'http://localhost:8000';
  return {
    plugins: [react()],
    server: {
      proxy: {
        '/api': { target, changeOrigin: true },
        '/health': { target, changeOrigin: true },
      },
    },
  };
});
