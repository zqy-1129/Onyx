import { defineConfig, type ProxyOptions } from 'vite'
import react from '@vitejs/plugin-react'

// 后端地址：开发时用 Vite 代理，前端代码里不出现主机名（也就不需要 CORS）。
// 用 ONYX_API 覆盖，例如后端不在默认端口时：ONYX_API=http://127.0.0.1:9000 npm run dev
const API_TARGET = process.env.ONYX_API ?? 'http://127.0.0.1:8787'

const apiProxy: ProxyOptions = {
  target: API_TARGET,
  changeOrigin: true,
}

export default defineConfig({
  plugins: [react()],
  server: { port: 5173, proxy: { '/api': apiProxy } },
  build: { outDir: 'dist', sourcemap: true },
  test: { environment: 'jsdom', globals: true },
} as never)
