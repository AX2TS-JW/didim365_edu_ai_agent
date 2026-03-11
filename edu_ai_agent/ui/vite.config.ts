import { defineConfig, loadEnv } from "vite";
import react from "@vitejs/plugin-react";
import tsconfigPaths from 'vite-tsconfig-paths'
import path from "path";
import svgr from 'vite-plugin-svgr';

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), '');
  return {
    resolve: {
      alias: {
        '@': path.resolve(__dirname, './src')
      }
    },

    server: {
      proxy: {
        "/api": {
          target: env.VITE_API_BASE_URL || 'http://localhost:8000',
          changeOrigin: true,
          rewrite: (path) => path.replace(/^\/api/, ""),
          // secure: false,
        },
      }
    },

    plugins: [
      svgr({
        svgrOptions: {
          icon: true,
          replaceAttrValues: {
            '#737373': '{props.color}',
            'white': '{props.color}'
          }
        },

        include: '**/*.svg?react'
      }),
      tsconfigPaths(),
      react(),
    ],
  };
});
