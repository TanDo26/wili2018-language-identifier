import { defineConfig, loadEnv } from 'vite';
import react from '@vitejs/plugin-react';

// Console dev server runs on :3001 so the end-user app can keep :3000 (see docker-compose.yml).
// /api is proxied to FastAPI (or a Prism mock) for the registry views added later:
//   VITE_API_TARGET=http://localhost:4010 npm run dev
export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), '');
  const target = env.VITE_API_TARGET || 'http://localhost:8000';
  return {
    plugins: [react()],
    server: {
      port: 3001,
      proxy: {
        '/api': { target, changeOrigin: true },
        '/health': { target, changeOrigin: true },
      },
    },
    test: {
      environment: 'jsdom',
      globals: true,
    },
  };
});
