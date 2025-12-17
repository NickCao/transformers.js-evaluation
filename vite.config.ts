import { defineConfig } from 'vite';

export default defineConfig({
    // Base path for GitHub Pages - set via env or defaults to '/'
    base: process.env.VITE_BASE || '/',
    server: {
        port: 3000,
    },
    preview: {
        port: 3000,
    },
    optimizeDeps: {
        exclude: ['@huggingface/transformers'],
    },
    build: {
        target: 'esnext',
    },
});
