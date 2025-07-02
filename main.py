import os
import json
import logging
import datetime
import signal
import sys
import concurrent.futures
import asyncio
import shutil
import random
import torch
import numpy as np
from config import DEFAULT_CONFIG, BACKEND_CONFIGS, VLLM_API_POOL, OPENAI_API_POOL
from generator_code_async import LLMInference, APIPool, TreeNode, inject_runtime_to_tree

def set_seed(seed: int = 42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def signal_handler(signum, frame):
    logger = logging.getLogger()
    logger.info(f"Received signal {signum}, terminating.")
    sys.exit(1)

def setup_logger(log_file="generation.log"):
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    if logger.hasHandlers():
        logger.handlers.clear()

    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    fh = logging.FileHandler(log_file, mode="w", encoding="utf-8")
    fh.setLevel(logging.DEBUG)

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    ch.setFormatter(formatter)
    fh.setFormatter(formatter)

    logger.addHandler(ch)
    logger.addHandler(fh)
    return logger

async def run_generation_process(
    log_file,
    output_file,
    tree_structure_file,
    result_file,
    backend="vllm",
    use_api_pool=False,
    config_override=None
):
    logger = logging.getLogger()
    logger.info(f"Starting data generation process with backend: {backend}")
    
    config = DEFAULT_CONFIG.copy()
    
    if backend in BACKEND_CONFIGS:
        config.update(BACKEND_CONFIGS[backend])
    
    if config_override:
        config.update(config_override)
    
    api_pool = None
    if use_api_pool:
        if backend == "vllm":
            logger.info("Using vLLM API pool")
            api_pool = APIPool(VLLM_API_POOL)
        elif backend == "openai":
            logger.info("Using OpenAI API pool")
            api_pool = APIPool(OPENAI_API_POOL)

    llm_engine = LLMInference(
        backend=backend,
        api_pool=api_pool,
        config=config,
        logger=logger,
        max_retries=config.get("max_retries", 5),
        max_workers=config.get("max_workers", 64),
        max_concurrent_requests=config.get("max_concurrent_requests", 64)
    )

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    logger.info(f"Initializing {backend} Inference Engine...")
    threadpool_executor = concurrent.futures.ThreadPoolExecutor(max_workers=config.get("max_workers", 64))

    root_node = None
    if os.path.exists(output_file):
        logger.info(f"Detected existing {output_file}, attempting to load previous tree...")
        try:
            with open(output_file, "r", encoding="utf-8") as f:
                tree_data = json.load(f)
            root_node = TreeNode.from_dict(tree_data, parent=None)
            inject_runtime_to_tree(
                root_node,
                llm_engine,
                threadpool_executor,
                config.get("max_depth", 4),
                config.get("max_sample_infinite_attribute", 50),
                config.get("max_attribute_count", 50)
            )
            logger.info("Successfully loaded existing tree. Will continue expansion.")
        except Exception as e:
            logger.error(f"Failed to load {output_file} due to {e}. Will start fresh.")
            root_node = None

    if root_node is None:
        root_node = TreeNode(
            llm_engine=llm_engine,
            threadpool_executor=threadpool_executor,
            tree_structure_file=tree_structure_file,
            depth=0,
            parent=None,
            dimension=None,  
            attribute_value=None,
            max_depth=config.get("max_depth", 4),
            num_samples_per_node=config.get("num_samples_per_node", 10),
            infinite_threshold=config.get("max_sample_infinite_attribute", 50),
            max_attribute_count=config.get("max_attribute_count", 50),
        )

    await root_node.expand_nodes_async(output_file=output_file, result_file=result_file)

    root_node.save_tree_structure(tree_structure_file)

    if not root_node.samples:
        logger.error("Root node has no samples after expansion.")
        return None

    logger.info("Data generation completed successfully.")
    return root_node

if __name__ == "__main__":
    set_seed(42)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join("output", timestamp)
    os.makedirs(output_dir, exist_ok=True)

    log_file = os.path.join(output_dir, "generation.log")
    output_file = os.path.join(output_dir, "output.json")
    tree_structure_file = os.path.join(output_dir, "tree_structure.txt")
    result_file = os.path.join(output_dir, "result.jsonl")

    load_from = None
    if load_from is not None and os.path.exists(load_from):
        print(f"[main] Found old output.json => copying from {load_from} to {output_file}")
        shutil.copyfile(load_from, output_file)
    else:
        print(f"[main] No old file found, will start fresh.")

    setup_logger(log_file)
    
    # Run with vLLM backend
    #asyncio.run(
    #    run_generation_process(
    #        log_file=log_file,
    #        output_file=output_file,
    #        tree_structure_file=tree_structure_file,
    #        result_file=result_file,
    #        backend="vllm",
    #        use_api_pool=False
    #    )
    #)
    
    # Run with OpenAI backend
    asyncio.run(
        run_generation_process(
            log_file=log_file,
            output_file=output_file,
            tree_structure_file=tree_structure_file,
            result_file=result_file,
            backend="openai",
            use_api_pool=False
        )
    )