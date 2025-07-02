import json
import os
import re
from openai import OpenAI
import logging
import asyncio
#import concurrent.feature
import datetime
from collections import deque
from typing import List, Dict
import random
import aiofiles
import numpy as np

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


SAMPLE_PROMPT = """Your task is to create {n_samples} short stories where information asymmetry naturally exists. The goal is to generate training data that helps an LLM anticipate behavior based on mental states rather than observable clues. Below are some instructions to follow for each instance.
1. Write a two-sentence story.
    - Decide how to instantiate the main entities in the story:
        - Person X (required): either a real, creative name followed by a simple descriptor indicating their role in the story or a group of people.
        - Object Z / Person Z / Action Z (required): This will be the subject of the KEY INFORMATION.
        - Person Y (optional): any additional character or group if needed for the story, but is not required.
    - For the FIRST SENTENCE of the story, write the KEY INFORMATION about Object Z / Person Z / Action Z (and Person Y) that is unknown to Person X (due to the general reason given in the scenario). Person X should not be able to observe this KEY INFORMATION through their actions in the story (either implicit or explicit actions). DO NOT use information which might be observed by Person X through normal, careful observation (such as "expiration date", "leaking container", "smell", etc).
    - The SECOND SENTENCE of the story is about what Person X will usually do regarding Object Z / Person Z / Action Z (and Person Y) in the scenario (ignoring the KEY INFORMATION). This sentence should describe what the character does using fine-grained actions. DO NOT include any descriptions which involve the emotions or thoughts of Person X, just describe actions.

2. Generate Question & Answer Choices:
   - Write a question predicting what Person X will likely do next.  
   - Provide two verbal action choices:  
     - (A): Correct action to the question (given the fact that person X is not aware of the KEY INFORMATION). Make sure the story does not have any mention of this action.
     - (B): Counterfactual (incorrect) action to the question. This answer should be a likely answer to the question under the assumption that person X somehow has full access to the KEY INFORMATION after all (maybe only possible using "magic" or some omnipotent skill).
    - Each action should be a complete but concise verbal phrase, without adjectives or adverbs. Avoid making it too short or too detailed.
    - Ensure two choices are in the **same length**. 
3. Give the Final Answer: 
   - Provide a short chain-of-thought explaining why the correct answer is (A).  

### All stories MUST be **accociated with the ATTRIBUTES below**:
{attributes_json}

Now, organize your response ({n_samples} instances) in the following format. Separate each instance using **only a blank line** (no extra dividers or explanations).

Instance <N>:
[INPUT]
Given the following story, answer the question by giving the correct answer choice, (A) or (B).
Story: <the two-sentence story>
Question: <the question>
(A) <action choice when X is unaware of the key information>  
(B) <choice when X has full knowledge of the key information> 
What is the correct answer?

[ANSWER]
<the chain-of-thought>. So the answer is (A).
"""

CLASSIFICATION_PROMPT = """You are tasked with classifying the following story based on the criterion of **{dimension}**.

STORY:
{sample}

You must **select exactly one** category from the option list below that best fits the main characteristics of the story. If none is a perfect fit, select the closest one.

OPTION LIST:
{attributes}

Please strictly follow the output format below and do not output anything else.

Output in JSON format:
{{
    "category": "<selected category, using **exact text** from the list>",
    "explanation": "<one-sentence justification>"
}}
"""

class LLMInference:
    """Unified inference class for both vLLM and OpenAI backends"""

    def __init__(
        self,
        backend="vllm",
        logger=None,
        max_retries=5,
        max_workers=20,
        max_concurrent_requests=64,
    ):
        self.logger = logger or logging.getLogger()
        self.backend = backend.lower().strip()
        self.max_retries = max_retries
        self.semaphore = asyncio.Semaphore(max_concurrent_requests)

        if self.backend == "vllm":
            self.model_name = "model-name-to-vllm"
            self.client = OpenAI(
                base_url="url-to-vllm", 
                api_key="token-to-vllm"
            )
            self.logger.info("Using vLLM backend with single endpoint.")
        else:
            raise ValueError(f"Unsupported backend {self.backend}")
    
    async def generate_async(self, prompt, max_tokens=1024, temperature=0.7):
        assert isinstance(prompt, str), "Prompt must be a string."
        retries = 0
        ans = ""

        backoff = 1.0
        while retries < self.max_retries:
            retries += 1
            async with self.semaphore:
                try:
                    if self.logger:
                        self.logger.info(f"[Async LLM Input] model={self.model_name}, tokens={max_tokens}, temp={temperature}, Prompt:\n{prompt}")

                    completion = await asyncio.to_thread(
                        self.client.chat.completions.create,
                        model=self.model_name,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=temperature,
                        max_tokens=max_tokens,
                    )

                    ans = completion.choices[0].message.content.strip()

                    if self.logger:
                        self.logger.info(f"[Async LLM API Output]:\n{ans}")
                    
                    break

                except Exception as e:
                    self.logger.error(f"Async LLM API error: {e}, attempt {retries}.")
                    await asyncio.sleep(backoff)
                    backoff *= 2

        if ans == "":
            self.logger.error(f"[Failed] No response after {self.max_retries} attempts.")
        return ans


def parse_story(input: str, logger=None):
    """extract Story sentences from standard input"""
    pattern = r"\nStory:(.*?)(?=\nQuestion)"
    match = re.findall(pattern, input)
    if match:
        story = match[0].strip()
        if logger:
            logger.debug(f"Parsed story: {story}")
        return story
    else:
        if logger:
            logger.warning(f"Failed story parsing: {input}")
        return ""


def parse_attribute(response: str, logger=None):
    
    if not response or not isinstance(response, str):
        if logger:
            logger.warning(f"Invalid response for JSON parsing: {response}")
        return []

    def extract_balanced(text:str, left_delimiter='{', right_delimiter='}'):
        stack = []
        start = None
        candidates = []
        for i, char in enumerate(text):
            if char == left_delimiter:
                stack.append(char)
                if not stack:
                    start = i
            elif char == right_delimiter and stack:
                stack.pop()
                if not stack:
                    candidates.append(text[start: i+1])
                    start = None
        return candidates

    candidates = extract_balanced(response)     # extracting text within curly brace
    results = []
    for json_str in candidates:
        cleaned_str = json_str.strip().rstrip(',').rstrip('.').strip()
        try:
            parsed = json.loads(cleaned_str)
            if isinstance(parsed, dict):
                results.append(parsed)
        except json.JSONDecodeError:
            lines = response.splitlines()
            results = [line.strip() for line in lines if line.strip()]
            if logger:
                logger.debug(f"JSONDecodeError: {cleaned_str}")
    
    if not results:
        if logger:
            logger.warning("No valid JSON found.")
        return None
    if logger:
        logger.info(f"Parsed JSON candidates: {results}")
    return results[0]


def parse_sft_data(raw_text: str, logger=None) -> List[Dict]:

    pattern = r"""Instance\s*(\d+)\s*:\s*
\[INPUT\]\s*\n([\s\S]*?)(?=\n\[ANSWER\])
\[ANSWER\]\s*\n([\s\S]*?)(?=(?:\n{1,}[-\s]*Instance\s*\d+\s*:)|$)"""
    data_list = []
    matches = re.findall(pattern, raw_text, flags=re.DOTALL)
    for qtuple in matches:
        try:
            data = {
                "input": qtuple[1].strip(),
                "output": qtuple[2].strip()
            }
            data_list.append(data)
        except:
            qtexts = [s for s in qtuple[1:]]
            if logger:
                logger.warning(f"Failed sft data parsing:____{qtexts}___")

    return data_list


class TreeNode():
    def __init__(
        self,
        llm_engine,
        depth,
        sample_ids = [],
        samples = [],       # only for leaf nodes
        parent = None,
        dimension = None,
        attribute = None,
        ch_dimension = None,
        children = {},
        logger=None
    ):
        self.llm_engine = llm_engine
        self.depth = depth
        self.parent = parent
        self.sample_ids = sample_ids
        self.samples = samples
        self.dimension = dimension
        self.attribute = attribute
        self.children = children
        self.ch_dimension = ch_dimension
        self.logger = logger

    def to_dict(self):
        return {
            "depth": self.depth,
            "dimension": self.dimension,
            "attribute": self.attribute,
            "sample_ids": json.dumps(self.sample_ids, ensure_ascii=False),
            "samples": self.samples,
            "children": [child.to_dict() for child in list(self.children.values())],
        }
    def retrieve_parents(self):
        parents = []
        current = self
        while current:
            parents.append(current)
            current = current.parent
        return parents

    def retrieve_dimension_attributes(self):

        parents = self.retrieve_parents()
        parents = parents[:-1]
        parents.reverse()

        dim_attrs = []
        for cur in parents:
            dim = cur.dimension
            attribute = cur.attribute
            if isinstance(attribute, (list, set)):
                attribute = random.choice(list(attribute))
            assert (dim is not None) and (
                attribute is not None
            ), f"Dimension and attribute_value must not be None. {dim} {attribute}"

            dim_attrs.append(
                {"dimension": dim, "attribute": attribute}
            )
        return dim_attrs

    def retrieve_root(self):
        cur = self
        while cur.parent:
            cur = cur.parent
        return cur

    def count_infinite_node(self):
        num = 0
        cur = self
        while cur:
            if isinstance(cur.attribute, list) and len(cur.attribute) > 1:
                num += 1
            cur = cur.parent
        return num

    async def classify_per_sample(self, sample_id):
        prompt = CLASSIFICATION_PROMPT.format(
            sample=parse_story(original_dataset[sample_id]["input"]),
            dimension=self.ch_dimension,
            attributes=list(self.children.keys())
        )
        response = await self.llm_engine.generate_async(prompt, max_tokens=128, temperature=0.3)
        pydict = parse_attribute(response)

        try:
            self.children[pydict["category"].lower()].sample_ids.append(sample_id)
        except Exception as e:
            self.logger.warning(f"Fail to classify sample_id={sample_id} due to {e}\nRaw response: {response}, Parsed Dict: {pydict}")


    async def leaf_samples_curation(self):
        
        n_samples = 10 * 5**self.count_infinite_node()
        nn = len(self.sample_ids)
        if nn >= n_samples:
            curated_samples = [original_dataset[id] for id in random.sample(self.sample_ids, n_samples)]
            logger.info(f"Need {n_samples} data. Random sampled from {None} original data")
        else:
            curated_samples = [original_dataset[id] for id in self.sample_ids]
            logger.info(f"Need {n_samples} data. Original data only have {nn}. Now replenish {n_samples-nn} more.")
            for start_i in range(0, n_samples-nn, 10):
                attributes = self.retrieve_dimension_attributes()
                attributes_json = json.dumps(attributes, indent=2, ensure_ascii=False)
                prompt = SAMPLE_PROMPT.format(
                    n_samples=min(10, n_samples-nn-start_i), 
                    attributes_json=attributes_json
                )
                logger.info(f"Replenishing {start_i}~{start_i + min(10, n_samples-nn-start_i)}/{n_samples-nn} based on {attributes}")

                responses = await self.llm_engine.generate_async(prompt, max_tokens=4096, temperature=0.7)
                samples = parse_sft_data(responses, logger=self.logger)
                curated_samples.extend(samples)
        self.samples = curated_samples

        async with aiofiles.open(output_file, "w", encoding="utf-8") as f_out:
            await f_out.write(json.dumps(root.to_dict(), ensure_ascii=False, indent=4))

        async with aiofiles.open(result_file, "a", encoding="utf-8") as f_re:
            for sample in curated_samples:
                await f_re.write(json.dumps(sample, ensure_ascii=False) + "\n")

    @staticmethod
    def from_dict(d, llm_engine, depth, parent=None, logger=None):
        assert isinstance(d, dict), f"d type: {type(d)}"
        dimension = d.get("dimension", "")
        attribute = d.get("attribute_value", "")
        if isinstance(attribute, list):
            attribute = [attr.strip().lower() for attr in attribute]
        elif isinstance(attribute, str):
            attribute = attribute.strip().lower()
        children = d.get("children", [])

        node = TreeNode(
            llm_engine,
            depth,           
            parent=parent,
            dimension=dimension,
            attribute=attribute,
            ch_dimension=children[0].get("dimension" "") if children else None,
            children={},
            samples=[],
            sample_ids=[],
            logger=logger
        )

        for ch_d in children:
            if isinstance(ch_d, dict):
                child_node = TreeNode.from_dict(ch_d, llm_engine, depth+1, parent=node, logger=logger)
                if isinstance(child_node.attribute, list):
                    node.children["infinite"] = child_node
                elif child_node.attribute:
                    node.children[child_node.attribute] = child_node
        return node

    @staticmethod
    def from_ck_dict(d, llm_engine, depth, parent=None, logger=None):
        assert isinstance(d, dict), f"d type: {type(d)}"
        dimension = d.get("dimension", "")
        attribute = d.get("attribute", "")
        if depth == CHECKPOINT_DEPTH:
            sample_ids = []
        else:
            sample_ids = json.loads(d.get("sample_ids", "[]"))
            assert isinstance(sample_ids, list), f"sample_ids type {type(sample_ids)}"
            if len(sample_ids) > 0 :
                assert isinstance(sample_ids[0], int), f"sample_ids[0] type {type(sample_ids[0])}"
        samples = d.get("samples", [])
        if isinstance(attribute, list):
            attribute = [attr.strip().lower() for attr in attribute]
        elif isinstance(attribute, str):
            attribute = attribute.strip().lower()
        children = d.get("children", [])

        node = TreeNode(
            llm_engine,
            depth,         
            parent=parent,
            dimension=dimension,
            attribute=attribute,
            ch_dimension=children[0].get("dimension" "") if children else None,
            children={},
            samples=samples,
            sample_ids=sample_ids,
            logger=logger
        )

        for ch_d in children:
            if isinstance(ch_d, dict):
                child_node = TreeNode.from_ck_dict(ch_d, llm_engine, depth+1, parent=node, logger=logger)
                if isinstance(child_node.attribute, list):
                    node.children["infinite"] = child_node
                elif child_node.attribute:
                    node.children[child_node.attribute] = child_node
        return node

    def balanced_statistics(self):
        return len(self.sample_ids), len(self.samples)


async def classify_and_sampling_async(batch_size=100, leaf_batch_size=5):
    layer_q = deque([root, 0])
    while layer_q:
        cur_layer_todo = []
        depth = layer_q[-1]
        while not isinstance(layer_q[0], int):
            cur = layer_q.popleft()
            if len(cur.sample_ids) == 0:
                logger.info(f"--- Node [{depth}:{cur.dimension}:{"infinite" if isinstance(cur.attribute, list) else cur.attribute}] has 0 samples himself. Its {len(cur.children.values())} children[{cur.ch_dimension}] no longer need processing.")
                continue
            num_of_classified = 0           # Track how many samples have been processed
            for ch in list(cur.children.values()):
                layer_q.append(ch)          # same as above (bfs)
                num_of_classified += len(ch.sample_ids)
            if num_of_classified == 0:
                cur_layer_todo.append(cur)
                logger.info(f"+++ Node [{depth}:{cur.dimension}:{"infinite" if isinstance(cur.attribute, list) else cur.attribute}] has {len(cur.sample_ids)} samples himself. Its {len(cur.children.values())} children[{cur.ch_dimension}] have a total of {num_of_classified} classified samples")
            else:
                logger.info(f"--- Node [{depth}:{cur.dimension}:{"infinite" if isinstance(cur.attribute, list) else cur.attribute}] has {len(cur.sample_ids)} samples himself. Its {len(cur.children.values())} children[{cur.ch_dimension}] have a total of {num_of_classified} classified samples. No longer need processing.")
        
        tmp = layer_q.popleft()
        assert depth == tmp, f"{depth} != layer_q.popleft() {tmp}"
        if layer_q:
            layer_q.append(depth + 1)
        
        logger.info(f"Processing Depth-{depth}: {len(cur_layer_todo)} nodes...")
        if cur_layer_todo:                  
            task_queue = []
            for node in cur_layer_todo:
                if "infinite" in node.children:
                    node.children["infinite"].sample_ids = node.sample_ids
                else:
                    task_queue.extend([(node, sample_id) for sample_id in node.sample_ids])

            for i in range(0, len(task_queue), batch_size):
                batch_tasks = [
                    asyncio.create_task(task[0].classify_per_sample(task[1])) 
                    for task in task_queue[i:i+batch_size]
                ]
                await asyncio.gather(*batch_tasks)
                async with aiofiles.open(output_file, "w", encoding="utf-8") as f:
                    content = json.dumps(root.to_dict(), ensure_ascii=False, indent=4)
                    await f.write(content)
    
    logger.info(f"Successfully finish classification. Now start sampling for final datasets")

    bfs_queue = deque([root])
    leaf_queue = []
    while bfs_queue:
        node = bfs_queue.popleft()
        if not node.children:
            leaf_queue.append(node)
        else:
            for child_node in node.children.values():
                bfs_queue.append(child_node)

    for i in range(0, len(leaf_queue), leaf_batch_size):
        batch_tasks = [
            asyncio.create_task(leaf.leaf_samples_curation())
            for leaf in leaf_queue[i: i+leaf_batch_size]
        ]
        await asyncio.gather(*batch_tasks)

    logger.info(f"Successfully builed the final dataset!")
    return



if __name__ == "__main__":

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join("/workspace/v0/output/balanced_persona", timestamp)
    os.makedirs(output_dir, exist_ok=True)

    log_file = os.path.join(output_dir, "generation.log")
    output_file = os.path.join(output_dir, "output.json")
    result_file = os.path.join(output_dir, "balanced_results.jsonl")

    tree_file = "/workspace/v0/output/tree/structure/output.json"
    original_dataset_file = "/workspace/v0/output/persona/20250430_223946/output.jsonl"
    checkpoint_file = ""
    CHECKPOINT_DEPTH = None

    logger = setup_logger(log_file=log_file)
    llm_engine = LLMInference()

    original_dataset = {}
    with open(original_dataset_file, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            original_dataset[i] = json.loads(line)
    logger.info(f"Loaded {len(original_dataset)} high_temp samples from {tree_file}")

    if checkpoint_file and os.path.exists(checkpoint_file):
        with open(checkpoint_file, "r", encoding="utf-8") as f:
            tree_ck_dict = json.load(f)
        root = TreeNode.from_ck_dict(tree_ck_dict, llm_engine, 0, parent=None, logger=logger)
        logger.info(f"Successfully rebuilded decision tree from {checkpoint_file}.")

    else:
        with open(tree_file, "r", encoding="utf-8") as f:
            tree_dict = json.load(f)
        root = TreeNode.from_dict(tree_dict, llm_engine, 0, parent=None, logger=logger)
        root.sample_ids = list(original_dataset.keys())
        logger.info(f"Successfully builded decision tree from {tree_file}.")
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(root.to_dict(), f, ensure_ascii=False, indent=4)

    asyncio.run(classify_and_sampling_async(batch_size=100, leaf_batch_size=10))

    logger.info("Complete!")

