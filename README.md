# Transformers.js Benchmark

A browser-based benchmark tool for measuring GPU/CPU performance of large language model inference using [Transformers.js](https://huggingface.co/docs/transformers.js).

## Features

- **Real-time LLM inference benchmarking** in your browser
- **WebGPU support** for GPU-accelerated inference
- **WebAssembly fallback** for CPU inference
- **Key metrics tracking:**
  - **TTFT (Time to First Token)** - Measures initial latency
  - **ITL (Inter Token Latency)** - Average time between tokens
  - **Throughput (tokens/sec)** - Overall generation speed
- **Live metrics updates** - Watch performance in real-time
- **Export results** as JSON or CSV
- **Speedometer-inspired UI** - Clean, modern design

## Requirements

- [Bun](https://bun.sh/) runtime
- Modern browser with WebGPU support (Chrome 113+, Edge 113+) for GPU benchmarks
- Any modern browser for WASM/CPU benchmarks

## Quick Start

```bash
# Install dependencies
bun install

# Start the development server
bun run dev

# Open http://localhost:3000 in your browser
```

## Usage

1. **Select a model** from the dropdown menu
2. **Choose device** - WebGPU (GPU) or WASM (CPU)
3. **Set max tokens** to generate
4. Click **Start Benchmark**
5. View your results including TTFT, ITL, and throughput

## Available Models

| Model | Size | Description |
|-------|------|-------------|
| Granite 4.0 350M | ~350M params | IBM's Granite model, optimized for web |

## Metrics Explained

### TTFT (Time to First Token)
The time from when you submit a prompt until the first token is generated. This includes:
- Tokenization of input
- Model warm-up (if first run)
- Prefill/prompt processing

Lower TTFT means faster initial response.

### ITL (Inter Token Latency)
The average time between generating consecutive tokens. This measures the steady-state generation speed.

Lower ITL means faster streaming output.

### Throughput (tokens/sec)
The overall rate of token generation, calculated as total tokens divided by total time.

Higher throughput means better overall performance.

## Deployment

### GitHub Pages

This project includes a GitHub Actions workflow for automatic deployment to GitHub Pages.

1. Push your code to the `main` branch
2. Go to your repository Settings → Pages
3. Set Source to "GitHub Actions"
4. The workflow will automatically build and deploy on push

The site will be available at `https://<username>.github.io/<repo-name>/`

### Manual Build

```bash
# Build for production
bun run build

# Preview the build locally
bun run preview

# The built files are in the ./dist directory
```

## Technical Details

- Built with [Transformers.js](https://huggingface.co/docs/transformers.js) v3
- Uses ONNX Runtime Web for inference
- Models are automatically downloaded and cached in the browser
- Bundled with [Vite](https://vitejs.dev/)

## Browser Compatibility

| Browser | WebGPU | WASM |
|---------|--------|------|
| Chrome 113+ | ✅ | ✅ |
| Edge 113+ | ✅ | ✅ |
| Firefox | ❌ | ✅ |
| Safari 18+ | ✅ | ✅ |

## License

MIT
