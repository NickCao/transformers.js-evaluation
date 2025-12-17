/**
 * Transformers.js Benchmark
 * Measures GPU/CPU performance for LLM inference using transformers.js
 */

import { pipeline, type TextGenerationPipeline } from '@huggingface/transformers';

// Types
interface BenchmarkResults {
    model: string;
    device: string;
    maxTokens: number;
    metrics: {
        ttft: number;
        itl: number;
        tokensGenerated: number;
        totalTimeSeconds: number;
        tokensPerSecond: number;
    };
    tokenTimestamps: number[];
    timestamp: string;
    userAgent: string;
}

interface MetricsUpdate {
    ttft?: number | null;
    itl?: number | null;
    tokens?: number | null;
    totalTime?: number | null;
}

// Benchmark state
let isRunning = false;
let generator: TextGenerationPipeline | null = null;
let currentModel: string | null = null;
let results: BenchmarkResults | null = null;

// DOM Elements
const elements = {
    scoreValue: document.getElementById('score-value')!,
    scoreUnit: document.getElementById('score-unit')!,
    scoreRingProgress: document.getElementById('score-ring-progress') as SVGCircleElement,
    ttftValue: document.getElementById('ttft-value')!,
    itlValue: document.getElementById('itl-value')!,
    tokensValue: document.getElementById('tokens-value')!,
    totalTimeValue: document.getElementById('total-time-value')!,
    modelSelect: document.getElementById('model-select') as HTMLSelectElement,
    deviceSelect: document.getElementById('device-select') as HTMLSelectElement,
    tokensSelect: document.getElementById('tokens-select') as HTMLSelectElement,
    startBtn: document.getElementById('start-btn') as HTMLButtonElement,
    statusText: document.getElementById('status-text')!,
    progressFill: document.getElementById('progress-fill')!,
    outputSection: document.getElementById('output-section')!,
    outputBox: document.getElementById('output-box')!,
    detailsSection: document.getElementById('details-section')!,
    detailsTbody: document.getElementById('details-tbody')!,
    copyJsonBtn: document.getElementById('copy-json-btn') as HTMLButtonElement,
    downloadJsonBtn: document.getElementById('download-json-btn') as HTMLButtonElement,
    downloadCsvBtn: document.getElementById('download-csv-btn') as HTMLButtonElement,
};

// Navigation
document.querySelectorAll('.nav-btn').forEach(btn => {
    btn.addEventListener('click', () => {
        document.querySelectorAll('.nav-btn').forEach(b => b.classList.remove('active'));
        document.querySelectorAll('.view').forEach(v => v.classList.remove('active'));
        btn.classList.add('active');
        const view = (btn as HTMLElement).dataset.view;
        document.getElementById(`${view}-view`)?.classList.add('active');
    });
});

// Test prompt for benchmarking
const BENCHMARK_PROMPT = [
    { role: 'user', content: 'Write a short story about a robot learning to paint.' }
];

// Update status
function updateStatus(text: string, progress: number | null = null) {
    elements.statusText.textContent = text;
    if (progress !== null) {
        (elements.progressFill as HTMLElement).style.width = `${progress}%`;
    }
}

// Update score display with animation
function updateScore(score: number, animate = true) {
    if (animate) {
        elements.scoreValue.classList.add('animating');
        setTimeout(() => elements.scoreValue.classList.remove('animating'), 300);
    }
    elements.scoreValue.textContent = score.toFixed(1);
    
    // Update ring progress (normalize to 0-100 range, assuming max ~50 tokens/sec)
    const normalizedScore = Math.min(score / 50, 1);
    const circumference = 2 * Math.PI * 90; // r=90
    const offset = circumference * (1 - normalizedScore);
    elements.scoreRingProgress.style.strokeDashoffset = String(offset);
}

// Update metrics display
function updateMetrics(metrics: MetricsUpdate) {
    if (metrics.ttft != null) {
        elements.ttftValue.textContent = metrics.ttft.toFixed(1);
    }
    if (metrics.itl != null) {
        elements.itlValue.textContent = metrics.itl.toFixed(2);
    }
    if (metrics.tokens != null) {
        elements.tokensValue.textContent = String(metrics.tokens);
    }
    if (metrics.totalTime != null) {
        elements.totalTimeValue.textContent = metrics.totalTime.toFixed(2);
    }
}

// Reset display
function resetDisplay() {
    elements.scoreValue.textContent = '—';
    elements.ttftValue.textContent = '—';
    elements.itlValue.textContent = '—';
    elements.tokensValue.textContent = '—';
    elements.totalTimeValue.textContent = '—';
    elements.scoreRingProgress.style.strokeDashoffset = '565.48';
    elements.outputSection.classList.remove('visible');
    elements.detailsSection.classList.remove('visible');
    elements.outputBox.textContent = '';
    (elements.progressFill as HTMLElement).style.width = '0%';
}

// Load model
async function loadModel(modelId: string, device: string): Promise<boolean> {
    updateStatus(`Loading model: ${modelId}...`, 10);
    
    try {
        generator = await pipeline('text-generation', modelId, {
            dtype: 'fp16',
            device: device as 'webgpu' | 'wasm',
            progress_callback: (progress: any) => {
                if (progress.status === 'downloading') {
                    const pct = progress.progress || 0;
                    updateStatus(`Downloading: ${progress.file} (${pct.toFixed(1)}%)`, 10 + pct * 0.3);
                } else if (progress.status === 'loading') {
                    updateStatus(`Loading model into memory...`, 45);
                }
            }
        }) as TextGenerationPipeline;
        currentModel = modelId;
        updateStatus('Model loaded successfully', 50);
        return true;
    } catch (error: any) {
        console.error('Failed to load model:', error);
        updateStatus(`Error loading model: ${error.message}`, 0);
        return false;
    }
}

// Run benchmark
async function runBenchmark() {
    if (isRunning) return;
    
    isRunning = true;
    elements.startBtn.disabled = true;
    elements.startBtn.classList.add('running');
    elements.startBtn.querySelector('.btn-text')!.textContent = 'Running...';
    
    resetDisplay();
    
    const modelId = elements.modelSelect.value;
    const device = elements.deviceSelect.value;
    const maxTokens = parseInt(elements.tokensSelect.value);
    
    // Check WebGPU support
    if (device === 'webgpu' && !navigator.gpu) {
        updateStatus('WebGPU not supported in this browser. Please use WASM instead.', 0);
        resetBenchmarkState();
        return;
    }
    
    try {
        // Load model if needed
        if (!generator || currentModel !== modelId) {
            const loaded = await loadModel(modelId, device);
            if (!loaded) {
                resetBenchmarkState();
                return;
            }
        }
        
        updateStatus('Starting benchmark...', 55);
        
        // Timing variables
        const tokenTimestamps: number[] = [];
        let generatedText = '';
        let firstTokenTime: number | null = null;
        const startTime = performance.now();
        
        // Show output section
        elements.outputSection.classList.add('visible');
        elements.outputBox.textContent = '';
        
        // Run generation with streaming
        updateStatus('Generating tokens...', 60);
        
        const output = await generator(BENCHMARK_PROMPT, {
            max_new_tokens: maxTokens,
            do_sample: true,
            temperature: 0.7,
            top_p: 0.9,
            callback_function: (beams: any) => {
                // This is called for each token
                const now = performance.now();
                
                if (tokenTimestamps.length === 0) {
                    firstTokenTime = now;
                }
                tokenTimestamps.push(now);
                
                // Update progress
                const progress = 60 + (tokenTimestamps.length / maxTokens) * 35;
                updateStatus(`Generating: ${tokenTimestamps.length}/${maxTokens} tokens`, progress);
                
                // Get the current output text
                if (beams && beams[0] && beams[0].output_token_ids && generator) {
                    const currentText = generator.tokenizer.decode(beams[0].output_token_ids, { skip_special_tokens: true });
                    if (currentText.length > generatedText.length) {
                        generatedText = currentText;
                        elements.outputBox.textContent = generatedText;
                        elements.outputBox.scrollTop = elements.outputBox.scrollHeight;
                    }
                }
            }
        });
        
        const endTime = performance.now();
        
        // Extract final text
        if (output && output[0] && output[0].generated_text) {
            const finalText = output[0].generated_text;
            // Handle chat format
            if (Array.isArray(finalText)) {
                const assistantMsg = finalText.find((m: any) => m.role === 'assistant');
                generatedText = assistantMsg ? assistantMsg.content : '';
            } else {
                generatedText = finalText as string;
            }
            elements.outputBox.textContent = generatedText;
        }
        
        // Calculate metrics
        const totalTime = (endTime - startTime) / 1000; // seconds
        const ttft = firstTokenTime ? (firstTokenTime - startTime) : 0; // ms
        const numTokens = tokenTimestamps.length;
        
        // Calculate ITL (inter-token latency)
        let itl = 0;
        if (tokenTimestamps.length > 1) {
            const latencies: number[] = [];
            for (let i = 1; i < tokenTimestamps.length; i++) {
                latencies.push(tokenTimestamps[i] - tokenTimestamps[i - 1]);
            }
            itl = latencies.reduce((a, b) => a + b, 0) / latencies.length;
        }
        
        // Calculate throughput (tokens per second)
        const tokensPerSecond = numTokens / totalTime;
        
        // Store results
        results = {
            model: modelId,
            device: device,
            maxTokens: maxTokens,
            metrics: {
                ttft: ttft,
                itl: itl,
                tokensGenerated: numTokens,
                totalTimeSeconds: totalTime,
                tokensPerSecond: tokensPerSecond,
            },
            tokenTimestamps: tokenTimestamps,
            timestamp: new Date().toISOString(),
            userAgent: navigator.userAgent,
        };
        
        // Update display
        updateScore(tokensPerSecond);
        updateMetrics({
            ttft: ttft,
            itl: itl,
            tokens: numTokens,
            totalTime: totalTime,
        });
        
        // Show details section
        showDetailedResults(results);
        
        updateStatus('Benchmark complete!', 100);
        
    } catch (error: any) {
        console.error('Benchmark error:', error);
        updateStatus(`Error: ${error.message}`, 0);
    }
    
    resetBenchmarkState();
}

// Reset benchmark state
function resetBenchmarkState() {
    isRunning = false;
    elements.startBtn.disabled = false;
    elements.startBtn.classList.remove('running');
    elements.startBtn.querySelector('.btn-text')!.textContent = 'Start Benchmark';
}

// Get latencies from token timestamps
function getLatencies(): number[] {
    if (!results || results.tokenTimestamps.length < 2) return [0];
    const latencies: number[] = [];
    for (let i = 1; i < results.tokenTimestamps.length; i++) {
        latencies.push(results.tokenTimestamps[i] - results.tokenTimestamps[i - 1]);
    }
    return latencies;
}

// Show detailed results
function showDetailedResults(res: BenchmarkResults) {
    elements.detailsSection.classList.add('visible');
    
    const latencies = getLatencies();
    const rows = [
        ['Model', res.model, 'The LLM model used for benchmark'],
        ['Device', res.device.toUpperCase(), 'Inference backend (WebGPU/WASM)'],
        ['TTFT', `${res.metrics.ttft.toFixed(2)} ms`, 'Time to first token'],
        ['ITL (avg)', `${res.metrics.itl.toFixed(2)} ms`, 'Average inter-token latency'],
        ['ITL (min)', `${Math.min(...latencies).toFixed(2)} ms`, 'Minimum inter-token latency'],
        ['ITL (max)', `${Math.max(...latencies).toFixed(2)} ms`, 'Maximum inter-token latency'],
        ['Tokens Generated', String(res.metrics.tokensGenerated), 'Total tokens produced'],
        ['Total Time', `${res.metrics.totalTimeSeconds.toFixed(3)} s`, 'End-to-end generation time'],
        ['Throughput', `${res.metrics.tokensPerSecond.toFixed(2)} tok/s`, 'Tokens per second'],
    ];
    
    elements.detailsTbody.innerHTML = rows.map(([metric, value, desc]) => `
        <tr>
            <td>${metric}</td>
            <td>${value}</td>
            <td>${desc}</td>
        </tr>
    `).join('');
}

// Export functions
function exportJSON(): string | null {
    if (!results) return null;
    return JSON.stringify(results, null, 2);
}

function exportCSV(): string | null {
    if (!results) return null;
    const headers = ['Metric', 'Value'];
    const rows = [
        ['Model', results.model],
        ['Device', results.device],
        ['TTFT (ms)', results.metrics.ttft.toFixed(2)],
        ['ITL (ms)', results.metrics.itl.toFixed(2)],
        ['Tokens Generated', String(results.metrics.tokensGenerated)],
        ['Total Time (s)', results.metrics.totalTimeSeconds.toFixed(3)],
        ['Throughput (tok/s)', results.metrics.tokensPerSecond.toFixed(2)],
        ['Timestamp', results.timestamp],
    ];
    return [headers, ...rows].map(row => row.join(',')).join('\n');
}

// Event listeners
elements.startBtn.addEventListener('click', runBenchmark);

elements.copyJsonBtn.addEventListener('click', async () => {
    const json = exportJSON();
    if (json) {
        await navigator.clipboard.writeText(json);
        elements.copyJsonBtn.textContent = 'Copied!';
        setTimeout(() => elements.copyJsonBtn.textContent = 'Copy JSON', 2000);
    }
});

elements.downloadJsonBtn.addEventListener('click', () => {
    const json = exportJSON();
    if (json) {
        const blob = new Blob([json], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `benchmark-results-${Date.now()}.json`;
        a.click();
        URL.revokeObjectURL(url);
    }
});

elements.downloadCsvBtn.addEventListener('click', () => {
    const csv = exportCSV();
    if (csv) {
        const blob = new Blob([csv], { type: 'text/csv' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `benchmark-results-${Date.now()}.csv`;
        a.click();
        URL.revokeObjectURL(url);
    }
});

// Device change handler - reset model when device changes
elements.deviceSelect.addEventListener('change', () => {
    generator = null;
    currentModel = null;
    updateStatus('Device changed. Model will be reloaded on next benchmark.');
});

// Initialize
updateStatus('Ready to benchmark. Select a model and click Start.');
