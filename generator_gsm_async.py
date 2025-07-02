import json
import random
import logging
from collections import deque
from tqdm import tqdm
import os
import torch
import numpy as np
import sys
from openai import OpenAI, AzureOpenAI
from PrettyPrint import PrettyPrintTree
import re
import asyncio,threading
import httpx
import random
import concurrent.futures
import time
from config import *

class APIPool:
    """Generic API pool that works for both vLLM and OpenAI backends"""
    
    def __init__(self, api_configs, requests_per_minute=1000):
        self.configs = api_configs
        self.index = 0
        self.lock = threading.Lock()
        self.error_counts = {i: 0 for i in range(len(api_configs))}
        
        self.requests_per_minute = requests_per_minute
        self.min_delay = 60.0 / requests_per_minute
        self.last_request_time = {i: 0 for i in range(len(self.configs))}
        
    def get_next_config(self):
        with self.lock:
            best_index = self.index
            min_errors = float('inf')
            
            for i in range(len(self.configs)):
                idx = (self.index + i) % len(self.configs)
                if self.error_counts[idx] < min_errors:
                    min_errors = self.error_counts[idx]
                    best_index = idx
            
            current_time = time.time()
            time_since_last = current_time - self.last_request_time[best_index]
            if time_since_last < self.min_delay:
                time.sleep(self.min_delay - time_since_last)
            
            self.index = (best_index + 1) % len(self.configs)
            self.last_request_time[best_index] = time.time()
            
            return self.configs[best_index], best_index
    
    def report_error(self, index):
        with self.lock:
            self.error_counts[index] += 1
    
    def report_success(self, index):
        with self.lock:
            self.error_counts[index] = 0
    
class LLMInference:
    """Unified inference class for both vLLM and OpenAI backends"""
    
    def __init__(
        self,
        backend="vllm",
        api_pool=None,
        config=None,
        logger=None,
        max_retries=5,
        max_workers=20,
        max_concurrent_requests=64,
    ):
        self.logger = logger or logging.getLogger()
        self.backend = backend.lower().strip()
        self.max_retries = max_retries
        self.threadpool_executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=max_workers
        )
        self.api_pool = api_pool
        self.semaphore = asyncio.Semaphore(max_concurrent_requests)
        self.config = config or {}

        if self.backend == "vllm":
            self.model_name = self.config.get("model_name")
            if api_pool is None:
                self.logger.info("Using vLLM backend with single endpoint.")
                self.client = OpenAI(
                    base_url=self.config.get("api_base"), 
                    api_key=self.config.get("api_key")
                )
            else:
                self.logger.info("Using vLLM backend with API pool.")
                # Client will be created per request from the pool

        elif self.backend == "openai":
            self.model_name = self.config.get("model_name", "gpt-4o")
            if api_pool is None:
                self.logger.info("Using OpenAI/Azure backend with single endpoint.")
                self.client = AzureOpenAI(
                    azure_endpoint=self.config.get("azure_endpoint"),
                    api_key=self.config.get("api_key"),
                    api_version=self.config.get("api_version")
                )
            else:
                self.logger.info("Using OpenAI/Azure backend with API pool.")
                # Client will be created per request from the pool
        else:
            raise ValueError(f"backend must be 'vllm' or 'openai', got {self.backend}.")
    
    async def generate_per_prompt_async(self, prompt, max_tokens=1024, temperature=0.7):
        assert isinstance(prompt, str), "Prompt must be a string."
        retries = 0
        ans = ""

        backoff = 1.0
        while retries < self.max_retries:
            retries += 1

            async with self.semaphore:
                try:
                    if self.api_pool:
                        # Select from pool based on backend
                        if self.backend == "vllm":
                            conf, conf_idx = self.api_pool.get_next_config()
                            local_client = OpenAI(
                                base_url=conf["api_base"],
                                api_key=conf["api_key"]
                            )
                            model_name = conf["model_name"]
                        else:  # openai
                            conf, conf_idx = self.api_pool.get_next_config()
                            local_client = AzureOpenAI(
                                azure_endpoint=conf["endpoint"],
                                api_key=conf["key"],
                                api_version=conf["version"]
                            )
                            model_name = conf["model"]
                    else:
                        local_client = self.client
                        model_name = self.model_name

                    if self.logger:
                        self.logger.info(
                            f"[Async LLM Input] model={model_name}, tokens={max_tokens}, temp={temperature}, Prompt:\n{prompt}"
                        )

                    completion = await asyncio.to_thread(
                        local_client.chat.completions.create,
                        model=model_name,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=temperature,
                        max_tokens=max_tokens,
                    )
                    
                    ans = completion.choices[0].message.content.strip()
                    
                    if self.logger:
                        self.logger.info(f"[Async LLM API Output]:\n{ans}")
                    
                    # Report success if using pool
                    if self.api_pool:
                        self.api_pool.report_success(conf_idx)
                        
                    break  # success => break

                except httpx.HTTPStatusError as http_err:
                    # Report error if using pool
                    if self.api_pool:
                        self.api_pool.report_error(conf_idx)
                        
                    if http_err.response.status_code == 429:
                        self.logger.error(
                            f"[429] Too Many Requests => attempt {retries}/{self.max_retries}, backoff={backoff}s"
                        )
                        await asyncio.sleep(backoff)
                        backoff *= 2
                    else:
                        self.logger.error(f"HTTPStatusError: {http_err}, attempt {retries}.")
                        await asyncio.sleep(backoff)
                        backoff *= 2

                except Exception as e:
                    # Report error if using pool
                    if self.api_pool:
                        self.api_pool.report_error(conf_idx)
                        
                    self.logger.error(f"Async LLM API error: {e}, attempt {retries}.")
                    await asyncio.sleep(backoff)
                    backoff *= 2

        if ans == "":
            self.logger.error(f"[Failed] No response after {self.max_retries} attempts.")
        return ans

    async def generate_batch_async(self, prompts, max_tokens=1024, temperature=0.7):
        """
        Asynchronously generate completions for multiple prompts.
        """
        if isinstance(prompts, str):
            # If only one prompt, directly call single prompt
            return await self.generate_per_prompt_async(prompts, max_tokens, temperature)

        tasks = []
        for p in prompts:
            tasks.append(
                asyncio.create_task(
                    self.generate_per_prompt_async(p, max_tokens, temperature)
                )
            )
        # Execute concurrently
        results = await asyncio.gather(*tasks)
        return results

def parse_json_candidates(response, logger=None, debug=False):

    if not response or not isinstance(response, str):
        if logger:
            logger.error("Invalid response for JSON parsing.")
        return []

    if response.strip().lower() == '"infinite"':
        if logger:
            logger.info("Detected 'infinite' response.")
        return "infinite"
    # return "infinite"

    results = []

    def extract_balanced(text, open_char, close_char):
        stack = []
        start = None
        candidates = []
        for i, char in enumerate(text):
            if char == open_char:
                if not stack:
                    start = i
                stack.append(char)
            elif char == close_char and stack:
                stack.pop()
                if not stack:
                    candidates.append(text[start : i + 1])
                    start = None
        return candidates

    obj_candidates = extract_balanced(response, "{", "}")
    arr_candidates = extract_balanced(response, "[", "]")
    candidates = obj_candidates + arr_candidates

    for json_str in candidates:
        cleaned_str = json_str.strip().rstrip(",").rstrip(".").strip()
        try:
            parsed = json.loads(cleaned_str)
            if isinstance(parsed, (dict, list)):
                results.append(parsed)
        except json.JSONDecodeError:
            lines = response.splitlines()
            results = [line.strip() for line in lines if line.strip()]
            if debug and logger:
                logger.debug(f"JSONDecodeError: {cleaned_str}")

    if not results and logger:
        logger.warning("No valid JSON found.")
    else:
        if logger and debug:
            logger.debug(f"Parsed JSON candidates: {results}")

    return results


def parse_attributes_from_str(response: str, logger=None, debug=False) -> list:

    if logger and debug:
        logger.debug(f"Step3 raw response:\n{response}")

    lines = response.splitlines()
    lines = [l.strip() for l in lines if l.strip()]

    if logger and debug:
        logger.debug(f"Parsed lines: {lines}")

    if len(lines) == 1 and lines[0].lower() == "null":
        if logger and debug:
            logger.debug(
                "LLM indicates existing attributes fully cover the dimension (null)."
            )
        return ["null"]

    attr_values = [x.strip('"') for x in lines]
    return attr_values


class TreeNode:

    def __init__(
        self,
        depth,
        llm_engine=None,
        parent=None,
        dimension=None,
        attribute_value=None,
        max_depth=5,
        num_samples_per_node=10,
        infinite_threshold=10,
        max_attribute_count=20,
        threadpool_executor=None,
        tree_structure_file="tree_structure.txt",
    ):

        if parent is None:
            assert depth == 0, "Root node must have depth=0 if no parent."
        else:
            assert depth == parent.depth + 1, "Child node must have parent's depth+1."
        self.depth = depth

        self.dimension = dimension
        self.attribute_value = attribute_value
        self.parent = parent
        self.children = []
        self.samples = None
        self.dimensions = []
        self.max_depth = max_depth
        self.num_samples_per_node = num_samples_per_node
        self.max_attribute_count = max_attribute_count
        self.infinite_threshold = infinite_threshold
        self.tree_structure_file = tree_structure_file

        assert llm_engine is not None, "LLM engine must be provided."
        self.llm_engine = llm_engine
        assert threadpool_executor is not None, "Threadpool executor must be provided."
        self.threadpool_executor = threadpool_executor

        self.logger = getattr(self.llm_engine, "logger", None)

    def logging(self, msg, level="info"):
        if self.logger:
            getattr(self.logger, level)(msg)

    def is_leaf(self):
        return len(self.children) == 0

    def is_root(self):
        return self.parent is None

    def is_infinite(self, attribute_values):
        return len(attribute_values) > self.infinite_threshold

    def __str__(self):
        dimension = self.dimension if self.dimension else "root"
        attribute_value = self.attribute_value if self.attribute_value else "None"

        return f"dimension: {dimension}\n" f"attribute_value: {attribute_value}"

    def to_dict(self):
        return {
            "attribute_value": self.attribute_value,
            "dimension": self.dimension,
            "samples": self.samples,
            "children": [child.to_dict() for child in self.children],
        }
    
    def count_infinite_nodes_in_path(self):
        count = 0
        current = self
        while current:
            if isinstance(current.attribute_value, (list, set)):
                count += 1
            current = current.parent
        return count

    def retrieve_parents(self):

        parents = []
        current = self
        while current:
            parents.append(current)
            current = current.parent
        return parents

    def retrieve_dimension_values(self):

        parents = self.retrieve_parents()
        parents = parents[:-1]
        parents.reverse()

        dimensions = []
        for parent in parents:
            dim = parent.dimension
            value = parent.attribute_value
            
            if isinstance(value, (list, set)):
                value = random.choice(list(value))

            assert (dim is not None) and (
                value is not None
            ), "Dimension and attribute_value must not be None."
            dimensions.append(
                {
                    "dimension": dim,
                    "attribute_value": value,
                }
            )
        return dimensions

    def retrieve_parent_dimensions(self):

        attribute_values = self.retrieve_dimension_values()
        dimensions = [d["dimension"] for d in attribute_values]

        return dimensions

    def retrieve_root(self):

        current = self
        while current.parent:
            current = current.parent
        return current

    def save_tree_structure(self, output_file):

        root = self.retrieve_root()

        pt = PrettyPrintTree(
            lambda x: x.children,
            lambda x: f"""dim: {x.dimension if x.dimension else "root"}\nattr: {x.attribute_value}\nchild_count:({len(x.children)})""",
            orientation=PrettyPrintTree.Horizontal
        )
        tree_as_str = pt(root, return_instead_of_print=True)

        ansi_escape = re.compile(r"(?:\x1B[@-_][0-?]*[ -/]*[@-~])")
        tree_as_str = ansi_escape.sub("", tree_as_str)

        with open(output_file, "w", encoding="utf-8") as f:
            f.write(tree_as_str)

        self.logging(
            f"Tree structure saved (full tree) to {output_file}.", level="info"
        )

    def format_dim_prompt(self):
        """generate prompt for selecting dimension and classifying"""

        assert self.samples is not None, "Samples must be generated first."
        samples = ""
        for i, s in enumerate(self.samples, 1):
            samples += f"""{i}. {s}\n"""
        samples = samples.strip()

        dimensions = self.retrieve_parent_dimensions()

        prompt = f"""As an analysis expert, your task is to examine the following questions to identify the SINGLE most significant dimension that characterizes the question space and differentiates these questions.
Questions:
{samples}

Dimension Requirements:
1. Core Dimension Identification: Identify exactly ONE core dimension that best distinguishes these questions.
2. Excluded Dimensions: {', '.join(dimensions)}
3. Unique Categorization: Each question MUST be categorized into exactly ONE attribute value.
4. Mutually Exclusive Values: Attribute values must be mutually exclusive.
5. Clarity in Values: Avoid ambiguous attribute values, such as "others".
6. Independent Values: Each attribute must be a single distinct value - NO combined values like "attribute1_and_attribute2" or "attribute1/attribute2"! Each attribute must be a single distinct value - NO combined values like "attribute1_and_attribute2" or "attribute1/attribute2"! Each attribute must be a single distinct value - NO combined values like "attribute1_and_attribute2" or "attribute1/attribute2"!

Organize your responses in the following format without any extra text or explanations:
{{
"dimension": "dimension_name",
"attributes": {{
    "attribute1": [list of sample indices],
    "attribute2": [list of sample indices],
    ...
}}
}}
"""
        return prompt

    def select_dimension_and_classify(self):

        parent_dimensions = self.retrieve_parent_dimensions()

        prompt = self.format_dim_prompt()
        response = self.llm_engine.generate_batch(prompt, max_tokens=1024, temperature=1.0)
        candidates = parse_json_candidates(response, logger=self.logger, debug=True)

        self.logging(f"Parsed candidates: {candidates}", level="debug")

        valid = True
        new_dim = None
        for c in candidates:
            if isinstance(c, dict) and ("dimension" in c) and ("attributes" in c):
                dim = c["dimension"].strip()
                attr_map = c["attributes"]

                all_indices = set()
                for cat, arr in attr_map.items():
                    if not isinstance(arr, list) or any(
                        not (isinstance(x, int) and 1 <= x <= len(self.samples))
                        for x in arr
                    ):
                        valid = False
                        self.logging(
                            f"Invalid attribute '{cat}': {arr}.", level="debug"
                        )
                        break
                    all_indices.update(arr)

                if valid and all_indices == set(range(1, len(self.samples) + 1)):
                    if dim not in parent_dimensions:
                        new_dim = c
                        break
                    else:
                        self.logging(
                            f"Dimension '{dim}' is already used.", level="debug"
                        )
        if new_dim is None:
            self.logging(
                "No valid dimension classification found. Generate again.",
                level="warning",
            )
            return self.select_dimension_and_classify()

        return new_dim

    def format_expand_prompt(self, dimension, attribute_values):

        prompt = f"""As an analysis expert, your task is to supplement the potential attribute values for a specified dimension in order to comprehensively model the entire space of questions.

Dimension: {dimension}
Exiting attributes values: {json.dumps(attribute_values, indent=2)}

Requirements for New Attribute Values:
1. Clarity: Avoid ambiguous values, such as "others".
2. Mutual Exclusivity: Ensure that attribute values do not overlap.
3. Completeness: Ensure that all possible attribute values fully cover the dimension.
4. GRADE LEVEL: Keep all values within elementary and middle school students' understanding! Keep all values within elementary and middle school students' understanding! Keep all values within elementary and middle school students' understanding! 
5. SIMPLICITY: Use basic, straightforward terms that young students can understand! Use basic, straightforward terms that young students can understand! Use basic, straightforward terms that young students can understand! 

Organize your responses in the following format without any extra text or explanations:
- If the existing attribute values completely cover the entire dimension, only output "null". For example,
null
- If the number of potential attribute values is more than 10, first output 10 potential new attribute values, and end your output with "infinite" in a new line. For example,
attribute value 1
attribute value 2
...
attribute value 10
infinite
- Otherwise, output all the potential new attribute values, and end your output with "complete" in a new line. For example,
attribute value 1
attribute value 2
...
attribute value n
complete

"""

        return prompt

    def format_gen_prompt(self):

        if self.is_root():
            prompt = """As a math expert, you are tasked to generate 10 GSM8K-style math word problems suitable for a bright middle school student.

Each question should meet the following criteria:
1. Format: Write problems as real-world word problems that require mathematical reasoning to solve.
2. Step Count: Require between 2 and 8 steps to solve.
3. Operations: Utilize basic arithmetic operations: addition (+), subtraction (-), multiplication (*), and division (/).
4. Complexity: Vary in context and complexity, but REMAIN ACCESSIBLE TO MIDDLE SCHOOL STUDENTS!
5. Clarity: Provide clear, concise questions that encourage step-by-step calculations to reach the final answer.
6. Language: Use natural, conversational language to describe situations while keeping problems clear and unambiguous.
7. Diversity: Ensure that the questions are diverse and distinct from one another from all potential perspectives.

Organize your responses in the following format without any extra text or explanations:
Question 1: text
Question 2: text
...
Question 10: text
"""

        else:
            attributes = self.retrieve_dimension_values()
            attributes_json = json.dumps(attributes, indent=2, ensure_ascii=False)
            prompt = f"""As a math expert, you are tasked to generate 10 GSM8K-style math word problems suitable for a bright middle school student.

Each question should meet the following criteria:
1. Format: Write problems as real-world word problems that require mathematical reasoning to solve.
2. Step Count: Require between 2 and 8 steps to solve.
3. Operations: Utilize basic arithmetic operations: addition (+), subtraction (-), multiplication (*), and division (/).
4. Complexity: Vary in context and complexity, but REMAIN ACCESSIBLE TO MIDDLE SCHOOL STUDENTS!
5. Clarity: Provide clear, concise questions that encourage step-by-step calculations to reach the final answer.
6. Language: Use natural, conversational language to describe situations while keeping problems clear and unambiguous.
7. Diversity: Ensure that the questions are diverse and distinct from one another from all potential perspectives.
8. Attributes: Each problem should be associated with all these attributes: {attributes_json}

Organize your responses in the following format without any extra text or explanations:
Question 1: text
Question 2: text
...
Question 10: text
"""

        return prompt

    async def generate_samples_async(self):
        async def generate_subsamples():
            single_prompt = self.format_gen_prompt()

            responses = await self.llm_engine.generate_batch_async([single_prompt])

            all_samples = []
            pattern = r"Question\s+(\d+)\s*:\s*(.*?)(?=\s*Question\s+\d+:|$)"

            for idx, raw_text in enumerate(responses, start=1):
                self.logging(f"[Prompt {idx}] raw response = {raw_text}", level="debug")
                matches = re.findall(pattern, raw_text, flags=re.DOTALL)
                for qnum, qtext in matches:
                    all_samples.append(qtext.strip())
            return all_samples

        all_samples = await generate_subsamples()
        self.samples = all_samples
        return self.samples


    
    async def select_dimension_and_classify_async(self, max_attempts=5):
        for attempt in range(max_attempts):
            parent_dimensions = self.retrieve_parent_dimensions()
            prompt = self.format_dim_prompt()

            responses = await self.llm_engine.generate_batch_async(
                prompt, max_tokens=1024, temperature=1.0
            )
            response = responses[0] if isinstance(responses, list) else responses

            candidates = parse_json_candidates(response, logger=self.logger, debug=True)
            self.logging(
                f"[Attempt {attempt+1}/{max_attempts}] Parsed dimension candidates: {candidates}",
                level="debug"
            )

            found_dim = None
            for c in candidates:
                valid = True
                if isinstance(c, dict) and ("dimension" in c) and ("attributes" in c):
                    dim = c["dimension"].strip()
                    attr_map = c["attributes"]

                    all_indices = set()
                    for cat, arr in attr_map.items():
                        if not isinstance(arr, list) or any(
                            not (isinstance(x, int) and 1 <= x <= len(self.samples))
                            for x in arr
                        ):
                            valid = False
                            self.logging(f"Invalid attribute '{cat}': {arr}.", level="debug")
                            break
                        all_indices.update(arr)

                    if valid:
                        if all_indices == set(range(1, len(self.samples) + 1)) \
                        and dim not in parent_dimensions:
                            found_dim = c
                            break

            if found_dim is not None:
                return found_dim
            else:
                self.logging(
                    f"No valid dimension classification found in attempt {attempt+1}, will retry...",
                    level="warning"
                )

        self.logging(
            f"Failed to classify dimension after {max_attempts} attempts => skip node expansion.",
            level="error"
        )
        return None


    async def expand_dimension_async(self, dimension, attribute_values):
        attempts = 0
        while True:
            prompt = self.format_expand_prompt(dimension, attribute_values)
            responses = await self.llm_engine.generate_batch_async(prompt, max_tokens=1024, temperature=0.7)
            raw_res = responses[0] if isinstance(responses, list) else responses

            candidates = parse_attributes_from_str(raw_res, logger=self.logger, debug=True)
            if (not candidates) or (candidates[-1] not in ["null", "infinite", "complete"]):
                attempts += 1
                self.logging(f"Attempt {attempts}: invalid response => retrying", level="warning")
                continue

            elif candidates[-1] == "infinite":
                attribute_values += candidates[:-1]
                if len(attribute_values) > self.max_attribute_count:
                    break
                else:
                    self.logging("Insufficient attributes for infinite => continue refilling", level="warning")
                    continue
            elif candidates[-1] == "null":
                self.logging(
                    f"No valid expansion info found for dimension '{dimension}', use original attributes.",
                    level="warning",
                )
                break
            elif candidates[-1] == "complete":
                attribute_values += candidates[:-1]
                break
            else:
                self.logging(f"Unexpected last candidate: {candidates[-1]}", level="warning")
                continue

        return attribute_values

    async def expand_nodes_async(self, output_file=None, result_file=None):
        self.save_tree_structure(self.tree_structure_file)

        children = await self._expand_single_node_async(output_file, result_file)
        queue = deque(children)

        level = 0
        while queue:
            level_size = len(queue)
            tasks = []
            self.logging(f"[BFS] Start processing level={level} with {level_size} nodes", level="info")

            for _ in range(level_size):
                node = queue.popleft()
                tasks.append(asyncio.create_task(
                    node._expand_single_node_async(output_file, result_file)
                ))
            results = await asyncio.gather(*tasks)

            # BFS: collect next level
            for child_list in results:
                for c in child_list:
                    queue.append(c)
            self.save_tree_structure(self.tree_structure_file)
            level += 1

        self.save_tree_structure(self.tree_structure_file)

    async def _expand_single_node_async(self, output_file, result_file):
        config = getattr(self.llm_engine, "config", {})
        infinite_path_samples = config.get("infinite_path_samples", 3)
        
        if self.depth >= self.max_depth:
            self.logging(f"[Leaf@MaxDepth] depth={self.depth}, stop expansion.", level="info")
            
            infinite_count = self.count_infinite_nodes_in_path()
            total_samples = max(1, infinite_path_samples ** infinite_count)
            self.logging(f"Path has {infinite_count} infinite nodes, generating {total_samples} sample sets", level="info")
            
            all_samples = []
            for i in range(total_samples):
                samples = await self.generate_samples_async()
                all_samples.extend(samples)
                if i > 0:
                    self.logging(f"Generated sample set {i+1}/{total_samples} for infinite path", level="info")
            
            self.samples = all_samples
            
            if result_file:
                with open(result_file, "a", encoding="utf-8") as f:
                    for q in all_samples:
                        line = {"question": q}
                        f.write(json.dumps(line, ensure_ascii=False) + "\n")
            if output_file:
                tree_dict = self.retrieve_root().to_dict()
                with open(output_file, "w", encoding="utf-8") as f:
                    json.dump(tree_dict, f, ensure_ascii=False, indent=2)
            return []

        samples = await self.generate_samples_async()

        dim_dict = await self.select_dimension_and_classify_async(max_attempts=5)
        if dim_dict is None:
            self.logging("Dimension classification failed => treat this node as leaf.", "warning")
            if result_file:
                with open(result_file, "a", encoding="utf-8") as f:
                    for q in samples:
                        line = {"question": q}
                        f.write(json.dumps(line, ensure_ascii=False) + "\n")
            return []

        dimension = dim_dict["dimension"]
        attribute_list = list(dim_dict["attributes"].keys())
        expanded_list = await self.expand_dimension_async(dimension, attribute_list)
        for attr in expanded_list:
            if attr not in dim_dict["attributes"]:
                dim_dict["attributes"][attr] = []

        all_attrs = list(dim_dict["attributes"].keys())

        if self.is_infinite(all_attrs):
            child = type(self)(
                depth=self.depth + 1,
                llm_engine=self.llm_engine,
                dimension=dimension,
                attribute_value=all_attrs,
                parent=self,
                max_depth=self.max_depth,
                num_samples_per_node=self.num_samples_per_node,
                infinite_threshold=self.infinite_threshold,
                max_attribute_count=self.max_attribute_count,
                threadpool_executor=self.threadpool_executor,
                tree_structure_file=self.tree_structure_file,
            )
            self.children = [child]
        else:
            self.children = []
            for attr in all_attrs:
                c = type(self)(
                    depth=self.depth + 1,
                    llm_engine=self.llm_engine,
                    dimension=dimension,
                    attribute_value=attr,
                    parent=self,
                    max_depth=self.max_depth,
                    num_samples_per_node=self.num_samples_per_node,
                    infinite_threshold=self.infinite_threshold,
                    max_attribute_count=self.max_attribute_count,
                    threadpool_executor=self.threadpool_executor,
                    tree_structure_file=self.tree_structure_file,
                )
                self.children.append(c)

        if not self.children:
            self.logging(f"[Leaf] Node dimension={dimension} => leaf node with no children, writing samples.", level="info")
            
            infinite_count = self.count_infinite_nodes_in_path()
            total_samples = max(0, infinite_path_samples ** infinite_count - 1)
            
            if total_samples > 0:
                additional_samples = []
                for i in range(total_samples):
                    extra_samples = await self.generate_samples_async()
                    additional_samples.extend(extra_samples)
                    self.logging(f"Generated additional sample set {i+1}/{total_samples} for infinite path (leaf node)", level="info")
                samples.extend(additional_samples)
            
            if result_file:
                with open(result_file, "a", encoding="utf-8") as f:
                    for q in samples:
                        line = {"question": q}
                        f.write(json.dumps(line, ensure_ascii=False) + "\n")

        if output_file:
            tree_dict = self.retrieve_root().to_dict()
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(tree_dict, f, ensure_ascii=False, indent=2)

        return self.children
    
    @staticmethod
    def from_dict(d: dict, parent=None):
        node = TreeNode(
            depth=0,
            llm_engine=None,
            threadpool_executor=None,
            dimension=d.get("dimension"),
            attribute_value=d.get("attribute_value"),
            max_depth=5,
            num_samples_per_node=10,
            infinite_threshold=50,
            max_attribute_count=50,
            parent=parent,
            tree_structure_file="tree_structure.txt",
        )
        node.samples = d.get("samples", [])

        for child_dict in d.get("children", []):
            child_node = TreeNode.from_dict(child_dict, parent=node)
            node.children.append(child_node)

        if parent:
            node.depth = parent.depth + 1

        return node

def inject_runtime_to_tree(node: TreeNode,
                           llm_engine,
                           threadpool_executor,
                           max_depth,
                           infinite_threshold,
                           max_attribute_count):

    node.llm_engine = llm_engine
    node.threadpool_executor = threadpool_executor
    node.max_depth = max_depth
    node.infinite_threshold = infinite_threshold
    node.max_attribute_count = max_attribute_count

    for child in node.children:
        child.parent = node
        child.depth = node.depth + 1
        inject_runtime_to_tree(child,
                               llm_engine,
                               threadpool_executor,
                               max_depth,
                               infinite_threshold,
                               max_attribute_count)



class TeeToFile:
    def __init__(self, original_stream, file_path, mode="w", encoding="utf-8"):
        self.original_stream = original_stream
        self.file = open(file_path, mode=mode, encoding=encoding, buffering=1)

    def write(self, data):
        self.original_stream.write(data)
        self.original_stream.flush()
        self.file.write(data)
        self.file.flush()

    def flush(self):
        self.original_stream.flush()
        self.file.flush()

    def isatty(self):
        return True

    def close(self):
        self.file.close()


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
