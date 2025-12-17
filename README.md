# Transformers.js Benchmark

A browser-based benchmark tool for measuring GPU/CPU performance of large language model inference using [Transformers.js](https://huggingface.co/docs/transformers.js).

![Benchmark Screenshot](https://browserbench.org/Speedometer3.1/)

## Features

- **Real-time LLM inference benchmarking** in your browser
- **WebGPU support** for GPU-accelerated inference
- **WebAssembly fallback** for CPU inference
- **Key metrics tracking:**
  - **TTFT (Time to First Token)** - Measures initial latency
  - **ITL (Inter Token Latency)** - Average time between tokens
  - **Throughput (tokens/sec)** - Overall generation speed
- **Multiple model support** - Test different model sizes
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
| Llama 3.2 1B Instruct | ~1B params | Good balance of quality and speed |
| Qwen2.5 0.5B Instruct | ~0.5B params | Fast, lightweight |
| SmolLM2 135M Instruct | ~135M params | Very fast, for testing |
| SmolLM2 360M Instruct | ~360M params | Fast with better quality |

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

## Technical Details

- Built with [Transformers.js](https://huggingface.co/docs/transformers.js) v3
- Uses ONNX Runtime Web for inference
- Models are automatically downloaded and cached in the browser
- Supports quantized models (q4f16 for WebGPU, q4 for WASM)

## Browser Compatibility

| Browser | WebGPU | WASM |
|---------|--------|------|
| Chrome 113+ | ✅ | ✅ |
| Edge 113+ | ✅ | ✅ |
| Firefox | ❌ | ✅ |
| Safari 18+ | ✅ | ✅ |

## License

MIT
