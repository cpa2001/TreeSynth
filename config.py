import os

# Common configuration parameters
DEFAULT_CONFIG = {
    "max_depth": 4,
    "num_samples_per_node": 10,
    "max_attribute_count": 50,
    "max_sample_infinite_attribute": 50,
    "infinite_path_samples": 10,
    # "infinite_path_samples": 5,
    "max_workers": 64,
    "max_concurrent_requests": 64,
    "max_retries": 5,
}

# Backend-specific configurations
BACKEND_CONFIGS = {
    "vllm": {
        "api_base": "vllm-api-base",
        "api_key": "vllm-api-key",
        "model_name": "/path/to/model/qwen2_5-72b-instruct",
    },
    "azure": {
        "api_key": "azure-api-key",
        "model_name": "gpt-4o",
        "azure_endpoint": "https://azure-endpoint.openai.azure.com/",
        "api_version": "2024-10-21",
    },
    "openai": {
        "api_key": "openai-api-key",
        "model_name": "gpt-4o",
    }
}

# API pool configurations
VLLM_API_POOL = [
    {
        "api_base": "vllm-api-base",
        "api_key": "vllm-api-key",
        "model_name": "/path/to/model/qwen2_5-72b-instruct",
    },
]

OPENAI_API_POOL = [
    {
        "endpoint": "https://azure-endpoint.openai.azure.com/",
        "key": "azure-api-key1",
        "version": "2024-10-21",
        "model": "gpt-4o",
    },
    {
        "endpoint": "https://azure-endpoint.openai.azure.com/",
        "key": "azure-api-key2",
        "version": "2024-10-21",
        "model": "gpt-4o",
    }
]