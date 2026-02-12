# Fast LLM Inference System

High-performance LLM inference system targeting 300 tokens/sec with minimal RAM usage.

## Goals
- 300+ tokens/second generation speed
- Minimize RAM usage for large context windows
- Support for 16B-24B parameter models on 32GB RAM

## Planned Features
- Speculative decoding with small draft models
- KV cache quantization (INT4/INT8)
- Flash Attention integration
- Paged attention for memory efficiency
- Prefix caching for repeated prompts

## Models
Download models separately (not included in repo due to size):
- Phi-3-mini-4k-instruct-q4.gguf (draft model)
- Additional models as needed

## Status
🚧 Work in progress
