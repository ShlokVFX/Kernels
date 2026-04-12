# Generic Datacenter Kernels

A collection of high-performance CUDA kernels for datacenter workloads using Triton and GLUON.

## Current Implementations

### Flash Attention Forward (B200)
- **File**: `flash_attn_fwd_tcgen05_tma_causal.py`
- **Target**: NVIDIA Blackwell (B200)
- **Features**:
  - FP16 & BF16 support
  - Causal & non-causal attention
  - Multi-Head Attention (MHA)
  - Head dimensions: 64, 128
  - TMA (Tensor Memory Accelerator) for async data movement
  - TCGEN05 MMA operations

### Modal Runner
- **File**: `flash_attn_modal_runner.py`
- **Purpose**: Runner script for flash attention implementations

## Requirements

- PyTorch (CUDA-enabled)
- Triton (with GLUON support)
- NVIDIA B200 GPU (or compatible architecture)

## Building & Running

```bash
python flash_attn_modal_runner.py
```

## Supported Configurations

- **Data Types**: BF16, FP16
- **Attention Types**: Causal, Non-causal
- **Batch Sizes**: Flexible (tested with 1-2)
- **Head Dimensions**: 64, 128

## Not Yet Implemented

- GQA, MQA (Grouped/Multi-Query Attention)
- Varlen sequences
- Sliding window attention
- Split-KV optimization
- Head dimensions: 96, 192, 256

## Benchmarks

Includes correctness testing against cuDNN and performance benchmarking on B200.

---

*Personal experiments in high-performance kernel development*
