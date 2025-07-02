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
from typing import List, Dict, Union
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

        elif self.backend == "azure":
            self.model_name = self.config.get("model_name", "gpt-4o")
            if api_pool is None:
                self.logger.info("Using Azure backend with single endpoint.")
                self.client = AzureOpenAI(
                    azure_endpoint=self.config.get("azure_endpoint"),
                    api_key=self.config.get("api_key"),
                    api_version=self.config.get("api_version")
                )
            else:
                self.logger.info("Using Azure backend with API pool.")
                # Client will be created per request from the pool
        elif self.backend == "openai":
            self.model_name = self.config.get("model_name", "gpt-4o")
            if api_pool is None:
                self.logger.info("Using OpenAI backend with single endpoint.")
                self.client = OpenAI(
                    api_key=self.config.get("api_key")
                )
        else:
            raise ValueError(f"backend must be 'vllm', 'azure' or 'openai', got {self.backend}.")
    
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
                        elif self.backend == "azure":
                            conf, conf_idx = self.api_pool.get_next_config()
                            local_client = AzureOpenAI(
                                azure_endpoint=conf["endpoint"],
                                api_key=conf["key"],
                                api_version=conf["version"]
                            )
                            model_name = conf["model"]
                        else:
                            conf, conf_idx = self.api_pool.get_next_config()
                            local_client = OpenAI(
                                api_key=conf["api_key"]
                            )
                            model_name = conf["model_name"]
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
        if isinstance(prompts, str):
            return await self.generate_per_prompt_async(prompts, max_tokens, temperature)

        tasks = [asyncio.create_task(self.generate_per_prompt_async(p, max_tokens, temperature)) for p in prompts]
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


def parse_samples(raw_text: str, logger=None):
    pattern = r"""Instance\s*(\d+)\s*:\s*
SCENARIO:\s*(.*?)\s*(?=\nENTITIES:)
ENTITIES:\s*(.*?)\s*(?=\nKEY INFORMATION:)
KEY INFORMATION:\s*(.*?)\s*(?=\nSTORY SECOND SENTENCE:)
STORY SECOND SENTENCE:\s*(.*?)\s*(?=\nQUESTION:)
QUESTION:\s*(.*?)\s*(?=\nCORRECT ANSWER)
CORRECT ANSWER.*?:\s*(.*?)\s*(?=\nCOUNTERFACTUAL ANSWER)
COUNTERFACTUAL ANSWER.*?:\s*(.*?)\s*(?=(?:\n{1,}[-\s]*Instance\s+\d+\s*:)|$)"""
    ins_key = ["scenario", "entities", "key information", "story second sentence", "question", "correct", "counterfactual"]
    all_instances = []
    all_stories = []
    matches = re.findall(pattern, raw_text)
    for qtuple in matches:
        try:
            qdict = {}
            for i, s in enumerate(qtuple[1:]):
                qdict[ins_key[i]] = s.strip()
            logger.info(json.dumps(qdict, indent=2, ensure_ascii=False))
            all_instances.append(qdict)
            all_stories.append(qdict["key information"] + " " + qdict["story second sentence"])
        except:
            qtexts = [s for s in qtuple[1:]]
            logger.info(f"WRONG: {qtuple[0]}__{qtexts}__")
    return all_stories, all_instances

def construct_sft_data(e, logger=None) -> Dict:
    try:
        num = random.randint(0, 1)
        data = {
            "scenario": e["scenario"],
            "story": e["key information"] + " " + e["story second sentence"],
            "question": e["question"],
            "choices": {
                "text": [e["correct"], e["counterfactual"]] if num==0 else [e["counterfactual"], e["correct"]],
                "label": ["A", "B"]
            },
            "answerKey": "A" if num==0 else "B"
        }
        return data
    except:
        if logger:
            logger.warning("Fail to construct sft data.")
        return {}


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
        self.instances = []
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
            if isinstance(self.attribute_value, (list, set)):
                count +=1
            current = current.parent
        return count

    def retrieve_parents(self):
        """retrieve all parents of the current node
        start from the current node to the root
        """
        parents = []
        current = self
        while current:
            parents.append(current)
            current = current.parent
        return parents

    def retrieve_dimension_attributes(self):

        parents = self.retrieve_parents()
        parents = parents[:-1]  # dispose of root
        parents.reverse()       # top -> self

        dim_attrs = []
        for parent in parents:
            dim = parent.dimension
            value = parent.attribute_value
            if isinstance(value, list):
                value = random.choice(value)

            assert (dim is not None) and (value is not None), "Dimension and attribute_value must not be None." 

            dim_attrs.append(
                {
                    "dimension": dim,
                    "attribute_value": value,
                }
            )

        return dim_attrs

    def retrieve_parent_dimensions(self):
        dim_attrs = self.retrieve_dimension_attributes()
        return [d["dimension"] for d in dim_attrs]

    def retrieve_root(self):

        current = self
        while current.parent:
            current = current.parent
        return current

    def save_tree_structure(self, tree_file):

        root = self.retrieve_root()

        pt = PrettyPrintTree(
            lambda x: x.children,
            lambda x: f"""dim: {x.dimension if x.dimension else "root"}\nattr: {x.attribute_value}\nchild_count:({len(x.children)})""",
            orientation=PrettyPrintTree.Horizontal
        )
        tree_as_str = pt(root, return_instead_of_print=True)

        ansi_escape = re.compile(r"(?:\x1B[@-_][0-?]*[ -/]*[@-~])")
        tree_as_str = ansi_escape.sub("", tree_as_str)

        with open(tree_file, "w", encoding="utf-8") as f:
            f.write(tree_as_str)

        self.logging(
            f"Tree structure saved (full tree) to {tree_file}.", level="info"
        )
    
    def format_gen_prompt(self):
        if self.is_root():
            raise ValueError("Root node should not generate samples!") 

        else:
            attributes = self.retrieve_dimension_attributes()


            attributes_json = json.dumps(attributes, indent=2, ensure_ascii=False)
            
            prompt = f"""I want you to come up with 10 diverse short story instances. Each story should involve a Person X (or a group of people) who is NOT aware of a certain critical piece of KEY INFORMATION about an object, a person or an action (Object Z / Person Z / Action Z, Person Y). A scenario would be given to you which specifies the general reason for this unawareness and information asymmetry.

For each instance, your task is to instantiate the scenario with a two-sentence story. Follow these steps:
1. Decide how to instantiate the main entities in the story:
  - Person X (required): either a real, creative name followed by a simple descriptor indicating their role in the story or a group of people.
  - Object Z / Person Z / Action Z (required): This will be the subject of the KEY INFORMATION.
  - Person Y (optional): any additional character or group if needed for the story, but is not required.
2. Write the KEY INFORMATION about Object Z / Person Z / Action Z (and Person Y) that is unknown to person X (due to the general reason given in the scenario). Person X should not be able to observe this KEY INFORMATION through their actions in the story (either implicit or explicit actions). DO NOT use information which might be observed by person X through normal, careful observation (such as "expiration date", "leaking container", "smell", etc). This will be the first sentence in the story.
3. For the second sentence of the story, write a sentence about what person X will usually do regarding Object Z / Person Z / Action Z (and Person Y) in the scenario (ignoring the KEY INFORMATION). This sentence should describe what the character does using fine-grained actions. DO NOT include any descriptions which involve the emotions or thoughts of person X, just describe actions.
4. Write a question about what the next action of person X will likely be.
5. Write a correct answer to the question (given the fact that person X is not aware of the KEY INFORMATION). Make sure the story does not have any mention of this action.
6. Write a counterfactual (incorrect) answer to the question. This answer should be a likely answer to the question under the assumption that person X somehow has full access to the KEY INFORMATION after all (maybe only possible using "magic" or some omnipotent skill).

Important Reminders to Double-Check Before Generating Each Story Instance:
- Avoid stories about fantasy and magic, rather make them grounded in the real world.
- The fact that person X is unaware of the KEY INFORMATION should be a purely implicit deduction based on the commonsense logic of the scenario.
- Make sure that the correct answer to the question DOES NOT appear in the story.
- Ensure the KEY INFORMATION is NOT a regular occurrence or common practice that can be assumed to be true by default, or likely to be noticed through normal observation (e.g., a bottle that is leaking)
- DO NOT make KEY INFORMATION so minor that it does not affect the action even if person X had been aware of it.
- DO NOT use phrases which make the hidden nature of the KEY INFORMATION obvious. That is, DO NOT use phrases like "actually", "in fact", "secret", "hidden", etc.
- Maintain Diversity: Make sure each story instance is DISTINCT from the others in all aspects, except for the shared "SCENARIO" across the 10 instances.
- Keywords Variation: Avoid repeating the same phrases or keywords across different instances.

Additionally, all stories should be grounded in the real-world **scenario** below, which describes a naturally existing tpye of information asymmetry or unawareness:
SCENARIO: {attributes_json}

Now, **strictly organize your responses in the following format**. Separate each instance using **only a blank line** (no extra dividers or explanations).

Instance 1:
SCENARIO: <the scenario provided above>
ENTITIES: <entities, Person X = ..., Object Z / Person Z / Action Z = ..., (optional) Person Y = ...>
KEY INFORMATION: <key information, a sentence>
STORY SECOND SENTENCE: <story second sentence, a sentence>
QUESTION: <question, a sentence>
CORRECT ANSWER (Person X doesn’t know the KEY INFORMATION): <correct answer, a verb phrase with no more than 15 words>
COUNTERFACTUAL ANSWER (assume Person X actually knows the KEY INFORMATION): <counterfactual answer, a verb phrase with no more than 15 words, similar in length to CORRECT ANSWER>

Instance 2:
SCENARIO: <the scenario provided above>
ENTITIES: <entities, Person X = ..., Object Z / Person Z / Action Z = ..., (optional) Person Y = ...>
KEY INFORMATION: <key information, a sentence>
STORY SECOND SENTENCE: <story second sentence, a sentence>
QUESTION: <question, a sentence>
CORRECT ANSWER (Person X doesn’t know the KEY INFORMATION): <correct answer, a verb phrase with no more than 15 words>
COUNTERFACTUAL ANSWER (assume Person X actually knows the KEY INFORMATION): <counterfactual answer, a verb phrase with no more than 15 words, similar in length to CORRECT ANSWER>

...

Instance 10:
SCENARIO: <the scenario provided above>
ENTITIES: <entities, Person X = ..., Object Z / Person Z / Action Z = ..., (optional) Person Y = ...>
KEY INFORMATION: <key information, a sentence>
STORY SECOND SENTENCE: <story second sentence, a sentence>
QUESTION: <question, a sentence>
CORRECT ANSWER (Person X doesn’t know the KEY INFORMATION): <correct answer, a verb phrase with no more than 15 words>
COUNTERFACTUAL ANSWER (assume Person X actually knows the KEY INFORMATION): <counterfactual answer, a verb phrase with no more than 15 words, similar in length to CORRECT ANSWER>
"""
        # Each story instance must include all of the following **ATTRIBUTES**:
        #{layer1_attribute}
        return prompt

    async def generate_samples_async(self):
        async def generate_subsamples():
            single_prompt = self.format_gen_prompt()

            responses = await self.llm_engine.generate_batch_async([single_prompt], max_tokens=4096, temperature=0.7)
            all_samples, all_instances = [], []
            for idx, raw_text in enumerate(responses, start=1):
                self.logging(f"[Prompt {idx}] raw response = {raw_text}", level="debug")
                samples, instances = parse_samples(raw_text, logger=self.logger)
                all_samples.extend(samples)
                all_instances.extend(instances)
            return all_samples, all_instances

        all_samples, all_instances = await generate_subsamples()
        return all_samples, all_instances
        #self.samples = all_samples
        #self.instances = all_instances
        #return self.samples, self.instances



    def format_dim_prompt(self):
        """generate prompt for selecting dimension and classifying"""

        assert self.samples is not None, "Samples must be generated first."
        samples = ""
        for i, s in enumerate(self.samples, 1):
            samples += f"""{i}. {s}\n"""
        samples = samples.strip()

        parent_dimensions = self.retrieve_parent_dimensions()

        prompt = f"""Below are some stories that take place in real-world scenarios where unawareness and information asymmetry with various underlying reasons naturally exists. As an expert equipped with rich commonsense and extensive knowledge, your task is to examine the following stories to identify the SINGLE most significant dimension that characterizes the story space and differentiates these stories.
Stories:
{samples}

Dimension Requirements:
1. Core Dimension Identification: Identify exactly ONE core dimension that best distinguishes these stories.
2. Excluded Dimensions: {', '.join(parent_dimensions)}
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
                        if all_indices == set(range(1, len(self.samples) + 1)) and dim not in parent_dimensions:
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

    def format_expand_prompt(self, dimension, attribute_values):

        prompt = f"""As an analysis expert, your task is to supplement the potential attribute values for a specified dimension in order to comprehensively model the entire space of stories. Note that these stories take place in real-world scenarios where information asymmetry naturally exists, with various underlying causes.

Dimension: {dimension}
Existing attributes values: {json.dumps(attribute_values, indent=2)}

Requirements for New Attribute Values:
1. Clarity: Avoid ambiguous values, such as "others".
2. Mutual Exclusivity: Ensure that attribute values do not overlap with each other or with the existing values.
3. Completeness: Ensure that all possible attribute values fully cover the dimension.
4. Harmfulness and Unethicality: Avoid 

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


    async def _handling_leaf_node_async(self, output_file, result_file, samples_off_shelf=[], instances_off_shelf=[]):

        config = getattr(self.llm_engine, "config", {})
        infinite_path_samples = config.get("infinite_path_samples", 3)

        # Count infinite nodes in path and calculate total sample sets needed
        infinite_count = self.count_infinite_nodes_in_path()
        total_samples = max(1, infinite_path_samples ** infinite_count)
        if samples_off_shelf and instances_off_shelf:
            total_samples = total_samples - 1
        self.logging(f"Path has {infinite_count} infinite nodes, already have {1 if samples_off_shelf else 0} sample set, start generating {total_samples} sample sets", level="info")

        all_samples, all_instances = samples_off_shelf, instances_off_shelf
        for i in range(total_samples):
            # Each generation gets a fresh random selection of attributes from infinite nodes
            samples, instances = await self.generate_samples_async()
            all_samples.extend(samples)
            all_instances.extend(instances)
            if i > 0:
                self.logging(f"Generated sample set {i+1}/{total_samples} for infinite path", level="info")
        
        self.samples, self.instances = all_samples, all_instances

        if result_file:
            with open(result_file, "a", encoding="utf-8") as f:
                for inst in all_instances:
                    line = construct_sft_data(inst, self.logger)
                    f.write(json.dumps(line, ensure_ascii=False) + "\n")

        if output_file:
            tree_dict = self.retrieve_root().to_dict()
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(tree_dict, f, ensure_ascii=False, indent=4)


    async def _expand_single_node_async(self, output_file, result_file):

        if self.is_root():
            self.logging("Root node detected. Apply predefined dimension-values.", level="info")

            dimension = "Type of Information Asymmetry"
            attribute_values = [
                "Hidden Defect", "Misleading Marketing", "Counterfeit", "Misleading Containers", 
                "Hidden Intent", "Undisclosed Status", "Incomplete Disclosure", "Biased Data", 
                "Selective Information Release", "Skewed Performance Metrics", "False Promises", "Withheld Expertise", 
                "Manipulated Feedback", "Inaccurate Accounting", "Opaque Pricing", "Concealed Conflicts of Interest"
            ]

            self.children = []
            for attr in attribute_values:
                child = type(self)(
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
                self.children.append(child)

            if output_file:
                tree_dict = self.retrieve_root().to_dict()
                with open(output_file, "w", encoding="utf-8") as f:
                    json.dump(tree_dict, f, ensure_ascii=False, indent=2)
                    
            return self.children

        if self.depth >= self.max_depth:
            self.logging(f"[Leaf@MaxDepth] depth={self.depth}, stop expansion.", level="info")
            await self._handling_leaf_node_async(output_file, result_file)
            return []

        samples, instances = await self.generate_samples_async() 
        self.samples = samples
        dim_dict = await self.select_dimension_and_classify_async(max_attempts=5)

        if dim_dict is None:
            self.logging("Dimension classification failed => treat this node as leaf.", "warning")
            await self._handling_leaf_node_async(output_file, result_file, samples, instances)
            return []

        dimension = dim_dict["dimension"]
        attribute_list = list(dim_dict["attributes"].keys())
        expanded_list = await self.expand_dimension_async(dimension, attribute_list)
        for attr in expanded_list:
            if attr not in dim_dict["attributes"]:
                dim_dict["attributes"][attr] = []

        all_attrs = list(dim_dict["attributes"].keys())

        if len(all_attrs) == 0:
            self.logging(f"[Leaf] Node dimension={dimension} => leaf node, writing samples.", level="info")
            await self._handling_leaf_node_async(output_file, result_file, samples, instances)
            return []

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

def inject_runtime_to_tree(
        node: TreeNode,
        llm_engine,
        threadpool_executor,
        max_depth,
        infinite_threshold,
        max_attribute_count
    ):

    node.llm_engine = llm_engine
    node.threadpool_executor = threadpool_executor
    node.max_depth = max_depth
    node.infinite_threshold = infinite_threshold
    node.max_attribute_count = max_attribute_count

    for child in node.children:
        child.parent = node
        child.depth = node.depth + 1
        inject_runtime_to_tree(
            child,
            llm_engine,
            threadpool_executor,
            max_depth,
            infinite_threshold,
            max_attribute_count
        )


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

