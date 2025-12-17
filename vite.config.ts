import { defineConfig } from 'vite';

export default defineConfig({
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
