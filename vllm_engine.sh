#!/bin/bash

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False
NCCL_TIMEOUT=1800
NCCL_ASYNC_ERROR_HANDLING=1
NCCL_P2P_LEVEL=NVL
TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=1200
VLLM_WORKER_MULTIPROC_METHOD=spawn

# --- Llama 3.1 70B Instruct ---
# vllm serve /path/to/model/llama3_3-70b-instruct \
#     --trust-remote-code \
#     --tensor-parallel-size 8 \
#     --max-num-batched-tokens 8192 \
#     --max-num-seqs 512 \
#     --gpu-memory-utilization 0.9 \
#     --api-key token-abc123 \
#     --chat-template vllm_chat_template_llama3.1_json.jinja \
#     --host 0.0.0.0 \
#     --port 8000 \
#     --kv-cache-dtype auto

# # --- Qwen 2.5 72B Instruct ---
# vllm serve /path/to/model/qwen2_5-72b-instruct \
#     --trust-remote-code \
#     --tensor-parallel-size 8 \
#     --max-num-batched-tokens 8192 \
#     --max-model-len 4096 \
#     --max-num-seqs 512 \
#     --gpu-memory-utilization 0.9 \
#     --api-key token-abc123 \
#     --host 0.0.0.0 \
#     --port 8000 \
#     --kv-cache-dtype auto

