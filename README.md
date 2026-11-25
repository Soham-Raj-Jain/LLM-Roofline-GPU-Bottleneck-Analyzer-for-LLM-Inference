# LLM Roofline: GPU Bottleneck Analyzer for LLM Inference

A hardware-aware performance analyzer that determines whether Large Language Model inference is limited by **GPU compute**, **memory bandwidth**, or **GPU under-utilization**.

The system profiles an LLM workload, models its arithmetic intensity using a Roofline-style approach, benchmarks the underlying GPU with CUDA, and produces an automatic diagnosis with optimization recommendations.

---

## Overview

LLM inference performance depends on several interacting factors:

- Model size
- Batch size
- Context length
- Prefill vs decode
- Numerical precision
- GPU compute throughput
- GPU memory bandwidth
- KV-cache traffic

Because of this, simply observing that an LLM is "slow" does not tell us **why** it is slow.

This project attempts to answer:

> **Given an LLM, GPU, and inference configuration, what hardware resource is actually limiting performance?**

The analyzer separates inference into:

- **Prefill** — processing the initial prompt
- **Decode** — autoregressive token-by-token generation

and classifies each phase independently.

---

# Key Features

### CUDA GPU Characterization

The notebook includes custom CUDA C++ microbenchmarks that measure:

- Practical GPU memory bandwidth
- CUDA-core arithmetic throughput
- FP16 GEMM / Tensor Core performance

These values are used to construct a measured hardware Roofline rather than relying only on theoretical GPU specifications.

---

### Roofline-Based Bottleneck Analysis

For each workload, the system estimates:

```text
Arithmetic Intensity = FLOPs / Bytes Moved
```

and compares it with the GPU's measured machine balance:

```text
Machine Balance =
Measured Compute Throughput / Measured Memory Bandwidth
```

A normalized bottleneck score is then calculated:

```text
-1                                      +1
Memory -------------------------------- Compute
                   0
                Balanced
```

Interpretation:

| Score | Diagnosis |
|---|---|
| Near `-1` | Strongly memory-dominated |
| Near `0` | Balanced |
| Near `+1` | Strongly compute-dominated |

---

## More Than Just "Memory vs Compute"

Low arithmetic intensity does not necessarily mean the GPU's full memory bandwidth is being used.

The analyzer therefore distinguishes between:

```text
MEMORY-DOMINATED

MEMORY-BANDWIDTH SATURATED

MEMORY-DOMINATED + UNDERUTILIZED/LATENCY-LIMITED

COMPUTE-DOMINATED

COMPUTE SATURATED

COMPUTE-DOMINATED + UNDERUTILIZED/SHAPE-LIMITED
```

This makes the diagnosis more useful than simply labeling a workload as memory-bound or compute-bound.

---

# Models Used

The main controlled benchmark family is **Qwen3**:

```text
Qwen3-0.6B
Qwen3-1.7B
Qwen3-4B
```

Using multiple model sizes from the same architecture allows us to study scaling effects without changing the overall Transformer design.

The project also includes cross-model validation using other decoder-only Transformer models such as:

```text
TinyLlama
Phi-family models
Other compatible Hugging Face causal LMs
```

Qwen is used for experimentation and validation.

The analyzer itself is designed to support **common dense decoder-only Hugging Face Transformer models**.

---

# Experimental Variables

The benchmark suite varies:

### Model Size

```text
0.6B
1.7B
4B
```

### Batch Size

```text
1
2
4
8
```

### Context Length

```text
128
512
1024
2048
4096
```

### Inference Phase

```text
Prefill
Decode
```

### Precision

```text
FP16
INT8
INT4
```

This allows the project to study **when an LLM transitions between memory-dominated and compute-dominated execution**.

---

# System Architecture

```text
                        LLM Model
                            +
                    Inference Configuration
                            +
                           GPU
                            │
                            ▼
                ┌─────────────────────┐
                │ Hardware Profiler   │
                │                     │
                │ CUDA Bandwidth      │
                │ FP16 Compute Roof   │
                └─────────┬───────────┘
                          │
                          ▼
                ┌─────────────────────┐
                │ LLM Benchmark       │
                │                     │
                │ Prefill             │
                │ Decode              │
                │ Latency             │
                │ Throughput          │
                └─────────┬───────────┘
                          │
                          ▼
                ┌─────────────────────┐
                │ Analytical Model    │
                │                     │
                │ FLOPs               │
                │ Memory Traffic      │
                │ Arithmetic Intensity│
                └─────────┬───────────┘
                          │
                          ▼
                ┌─────────────────────┐
                │ Bottleneck Engine   │
                │                     │
                │ Memory              │
                │ Compute             │
                │ Under-utilization   │
                └─────────┬───────────┘
                          │
                          ▼
                ┌─────────────────────┐
                │ Recommendation      │
                │ Engine              │
                └─────────┬───────────┘
                          │
                          ▼
                Optimization Validation
```

---

# Example Diagnostic

A typical analysis may produce:

```text
================================================================================
LLM GPU DIAGNOSTIC
================================================================================

GPU:        Tesla T4
Model:      Qwen/Qwen3-0.6B
Precision:  FP16
Batch:      1
Context:    1024

--------------------------------------------------------------------------------
PREFILL
--------------------------------------------------------------------------------

Latency:               98.287 ms
Throughput:            10418 tokens/s
Arithmetic intensity:  1024 FLOP/byte
Bottleneck score:      +0.981

Diagnosis:
COMPUTE-DOMINATED


--------------------------------------------------------------------------------
DECODE
--------------------------------------------------------------------------------

Latency:               55.813 ms
Throughput:            17.9 tokens/s
Arithmetic intensity:  ~1 FLOP/byte
Bottleneck score:      -1.000

Diagnosis:
MEMORY-DOMINATED + UNDERUTILIZED/LATENCY-LIMITED
```

The same model can therefore exhibit completely different bottlenecks during different stages of inference.

---

# Why Prefill and Decode Behave Differently

One of the main findings investigated by this project is the difference between **prefill** and **decode**.

### Decode

During autoregressive generation, the model repeatedly accesses large weight tensors while generating relatively little computation per token.

Conceptually:

```text
Load weights
     ↓
small amount of computation
     ↓
generate one token
     ↓
repeat
```

This often produces very low arithmetic intensity.

---

### Prefill

During prompt processing, the same model weights are reused across many tokens:

```text
Load weights
     ↓
process hundreds/thousands of tokens
     ↓
large amount of computation
```

This dramatically increases arithmetic intensity.

As context length or batch size increases, workloads can move toward the compute side of the Roofline.

---

# Quantization Validation

The analyzer does not stop at identifying a bottleneck.

It also recommends possible optimizations.

For memory-dominated workloads, one recommendation is:

```text
FP16
 ↓
INT8
 ↓
INT4
```

The notebook benchmarks all three and measures:

- Model memory footprint
- Prefill latency
- Decode latency
- Tokens per second
- Speedup relative to FP16

This allows us to test whether the optimization predicted by the analyzer actually improves performance.

Importantly, the project does **not assume quantization must be faster**.

If quantization reduces memory usage but introduces enough kernel overhead to increase latency, the result is reported as such.

This allows the system to distinguish between:

```text
Theoretical bottleneck
```

and

```text
Real end-to-end performance
```

---

# Optimization Recommendation Engine

Depending on the diagnosis, the system may recommend:

### Memory-Dominated Workloads

```text
INT8 / INT4 weight quantization
KV-cache optimization
Larger batch sizes
Reduced memory movement
Kernel fusion
Optimized inference kernels
```

### Compute-Dominated Workloads

```text
FP16 / BF16 Tensor Core execution
Optimized GEMM kernels
Optimized attention kernels
Kernel fusion
torch.compile
```

### Underutilized Workloads

```text
Increase batch size
Increase workload parallelism
CUDA Graphs
Reduce kernel-launch overhead
Compilation
```

Recommendations are treated as **hypotheses** and should be validated experimentally.

---

# Arbitrary Hugging Face Model Analysis

The final notebook includes a generic analyzer.

Example:

```python
ARBITRARY_MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

ARBITRARY_BATCH = 1
ARBITRARY_CONTEXT = 2048
ARBITRARY_PRECISION = "fp16"
```

The system automatically attempts to determine:

```text
Parameter count
Number of layers
Hidden dimension
Attention heads
KV heads
Head dimension
Model memory footprint
```

and then performs the complete analysis pipeline.

---

# Supported Architectures

The current analytical model is intended primarily for:

> **Dense decoder-only Transformer language models**

Common Hugging Face causal-LM architectures can often be analyzed automatically.

Architectures requiring additional modeling include:

```text
Mixture-of-Experts models
Encoder-decoder models
State-space models
Multimodal models
Recurrent architectures
Unusual attention mechanisms
```

The notebook explicitly warns about architectures for which the simple FLOP/memory model may not be accurate.

---

# Tech Stack

```text
Python
PyTorch
Hugging Face Transformers
CUDA C++
NVIDIA CUDA Toolkit
bitsandbytes
NumPy
Pandas
Matplotlib
Google Colab
```

---

# GPU Platform

The primary experimental GPU is:

```text
NVIDIA Tesla T4
16 GB VRAM
```

running through Google Colab.

The design is hardware-normalized, so the same notebook can also be run on GPUs such as:

```text
NVIDIA L4
NVIDIA A100
Other CUDA-capable NVIDIA GPUs
```

A different GPU produces a different machine balance and therefore potentially a different bottleneck classification for the same LLM workload.

---

# Running the Project

## 1. Open the Notebook

Upload:

```text
LLM_Roofline_Final_Colab.ipynb
```

to Google Colab.

---

## 2. Enable GPU

Go to:

```text
Runtime
→ Change runtime type
→ GPU
```

A Tesla T4 is sufficient for the main experiments.

---

## 3. Verify CUDA

The first cells automatically check:

```bash
nvidia-smi
nvcc --version
```

and verify that PyTorch can access the GPU.

---

## 4. Run the CUDA Microbenchmarks

The notebook compiles custom CUDA code using:

```bash
nvcc
```

to measure practical GPU memory bandwidth.

It then benchmarks FP16 matrix multiplication to estimate the practical Tensor Core compute ceiling.

---

## 5. Run Qwen Scaling Experiments

Start with:

```python
RUN_MODE = "quick"
```

to verify the pipeline.

Then use:

```python
RUN_MODE = "full"
```

for the full benchmark matrix.

---

## 6. Run Quantization Validation

The notebook evaluates:

```text
FP16
INT8
INT4
```

on the same workload and reports speedup or slowdown relative to FP16.

---

## 7. Analyze Another Model

At the end of the notebook, change:

```python
ARBITRARY_MODEL_ID
ARBITRARY_BATCH
ARBITRARY_CONTEXT
ARBITRARY_PRECISION
```

and run the analyzer.

---

# Generated Outputs

The notebook exports several datasets:

```text
llm_roofline_results.csv

hardware_summary.json

quantization_validation_results.csv

cross_model_validation.csv

arbitrary_model_report.csv
```

These can be used for further analysis, visualization, or a future dashboard.

---

# Visualizations

The notebook generates:

### Roofline Plot

Shows workload arithmetic intensity relative to the measured memory and compute roofs.

### Bottleneck Score Plot

Visualizes the transition:

```text
Memory ←──────── Balanced ────────→ Compute
```

### Batch × Context Heatmaps

Shows how increasing context length and batch size changes the bottleneck.

### Quantization Comparison

Compares:

```text
FP16 vs INT8 vs INT4
```

for both latency and model memory footprint.

---

# Research Questions

The project explores several systems questions:

### RQ1

**When does LLM inference transition from memory-dominated to compute-dominated execution?**

### RQ2

**How do batch size and context length affect arithmetic intensity?**

### RQ3

**Why do prefill and decode exhibit different GPU bottlenecks?**

### RQ4

**How does model size affect GPU utilization?**

### RQ5

**Can a hardware-normalized Roofline metric predict useful optimization strategies?**

### RQ6

**Does reducing model memory traffic through quantization improve workloads diagnosed as memory-dominated?**

---

# Main Project Hypothesis

> **LLM inference bottlenecks can be estimated from the interaction between workload arithmetic intensity and GPU machine balance, and this diagnosis can be used to recommend hardware-aware performance optimizations.**

The project validates this hypothesis through direct runtime benchmarking rather than relying only on theoretical estimates.

---

# Final Goal

The long-term goal is to make the analyzer behave like an:

## LLM Performance Doctor

A user provides:

```text
Model
GPU
Batch size
Context length
Precision
```

and receives:

```text
Performance statistics
        +
Prefill diagnosis
        +
Decode diagnosis
        +
Arithmetic intensity
        +
GPU compute utilization
        +
GPU bandwidth utilization
        +
Bottleneck score
        +
Optimization recommendations
```

without manually analyzing CUDA performance characteristics.

---

# Limitations

The current version has several intentional limitations:

- FLOP and memory-traffic values are analytical estimates.
- Quantized models introduce additional metadata and kernel overhead not fully represented by simple bytes-per-parameter calculations.
- GPU utilization values are derived from modeled work and measured runtime rather than complete hardware performance-counter traces.
- The generic architecture analyzer focuses on dense decoder-only Transformers.
- Hosted Colab environments may restrict access to some low-level GPU performance counters.

These limitations are useful directions for future work.

---

# Future Work

Potential extensions include:

```text
NVIDIA Nsight Compute integration
CUPTI hardware-counter profiling
KV-cache quantization experiments
FlashAttention benchmarking
CUDA Graphs for decode
torch.compile comparisons
Multi-GPU analysis
More GPU architectures
MoE-aware FLOP modeling
Automatic optimization ranking
Web dashboard / Streamlit interface
Learned bottleneck predictor
```

A future learned predictor could use benchmark data to estimate:

```text
P(memory-bound)
P(compute-bound)
P(underutilized)
```

before running the complete profiling pipeline.

---

# Project Summary

```text
LLM
 +
GPU
 +
Workload
 │
 ▼
CUDA Hardware Characterization
 │
 ▼
Prefill + Decode Benchmarking
 │
 ▼
FLOPs / Memory Traffic Estimation
 │
 ▼
Roofline Analysis
 │
 ▼
Bottleneck Diagnosis
 │
 ▼
Optimization Recommendation
 │
 ▼
Experimental Validation
```

The result is a practical framework for understanding **why LLM inference is slow on a given GPU and what optimization is most likely to help**.

---

## Authors

Built as a four-person GPU systems / LLM inference performance project.

Contributions include:

- CUDA and GPU characterization
- LLM benchmarking
- Roofline and bottleneck modeling
- Quantization and optimization validation
- Cross-model analysis
- Visualization and reporting

---

## License

This project is intended for educational and research purposes.

Add your preferred open-source license before public distribution.
