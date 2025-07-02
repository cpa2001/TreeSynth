import json
import os
import re
from openai import OpenAI
import logging
import asyncio
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

NORMAL_PROMPT = """You are given a short, two-sentence story illustrating information asymmetry, where Person X is unaware of a crucial fact about Y. Your task is to convert the story into an "information symmetry" version by adding a **subtle environmental or background clue** **only in the first sentence**. Importantly, do **not** describe Person X actively noticing or becoming aware of the clue; just insert a slight detail implying something might be wrong or unusual.

**Detailed Instructions**

1. **Original Story** (two sentences):
   - Sentence 1 reveals the hidden fact (which X originally does not know).
   - Sentence 2 describes Person X’s action, still unaware.

2. **Modified Story** (two sentences):
   - **Sentence 1**: Insert a minor clue that could lead X (or a reader) to infer the hidden fact **without** explicitly saying "X notices" or "X realizes." The clue should be subtle and not directly point to the hidden fact.
   - **Sentence 2**: Keep it almost the same as in the original story, unless trivial edits are needed for coherence. Avoid stating that X has already changed behavior. The point is that X **could** have inferred the fact from the background detail, but the text does not explicitly say so.

3. **Question & Choices**:
   - Use the same question from the original story or rephrase it slightly as "What does X do next?"
   - (A) The original uninformed action.
   - (B) The new informed action.
   - Each action should be a complete but concise verbal phrase, without adjectives or adverbs. Avoid making it too short or too detailed.

4. **Final Answer**:
   - Provide a short (1–2 sentences) reasoning that references the subtle background clue in the first sentence, leading X to choose (B).
   - End with: "So the answer is (B)."

Here is the original story, question and uninformed action:
Original Story: {story}
Original Question: {ques}
Original Action: {act}

Now, organize your response in the following format. Separate each instance using **only a blank line** (no extra dividers or explanations).

[INPUT]
Given the following story, answer the question by giving the correct answer choice, (A) or (B).
Original Story: <the original two-sentence story given to you which shows the old info asymmetry>
Modified Story: <the new two-sentence story where X has discovered the missing info>
Question: <the question>
(A) <old uninformed action>
(B) <new informed action>
What is the correct answer?

[ANSWER]
<brief chain-of-thought>. So the answer is (B).

### Key Reminders

- Do not say "X notices / sees / realizes / suspects." Instead, simply mention an observable detail in the environment or object. Let the user infer that X **could** realize it.
- Keep the second sentence almost the same.
- The inserted clue must be enough that (B) is justified.
- This ensures the final scenario still requires a bit of inference, rather than the story outright stating X’s awareness.
"""

class LLMInference:
    """Unified inference class for both vLLM and OpenAI backends"""

    def __init__(
        self,
        backend="vllm",
        logger=None,
        max_retries=5,
        max_concurrent_requests=64,
    ):
        self.logger = logger or logging.getLogger()
        self.backend = backend.lower().strip()
        self.max_retries = max_retries
        self.semaphore = asyncio.Semaphore(max_concurrent_requests)

        if self.backend == "vllm":
            self.model_name = "/workspace/models/llama3_3_70b"
            self.client = OpenAI(
                base_url="http://localhost:8070/v1", 
                api_key="token-abc123"
            )
            self.logger.info("Using vLLM backend with single endpoint.")
        elif self.backend == "openai":
            self.model_name = "meta-llama/Llama-3.3-70B-Instruct"
            self.client = OpenAI(
                base_url="https://api.deepinfra.com/v1/openai",
                api_key="****"
            )
            self.logger.info("Using OpenAI backend with single endpoint")
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


def parse_sft_data(raw_text: str):
    pattern = r"Modified Story:\s*(.+)(?=\n+\s*Question)\n+\s*Question:\s*(.+)(?=\n+\s*\(A\))\n+\s*\(A\)(.+)(?=\n+\s*\(B\))\n+\s*\(B\)(.+)(?=\n+)\n+\s*(.+)"
    matches = re.findall(pattern, raw_text, flags=re.DOTALL)
    if matches:
        qtuple = matches[0]
        try:
            data = {
                "story": qtuple[0].strip(),
                "question": qtuple[1].strip(),
                "a": qtuple[2].strip(),
                "b": qtuple[3].strip(),
                "cot": qtuple[4].strip()
            }
            prompt = (
                f"Given the following story, answer the question by giving the correct answer choice, (A) or (B).\n"
                f"Story: {data["story"]}\n"
                f"Question: {data["question"]}\n"
                f"(A) {data["a"]}\n"
                f"(B) {data["b"]}\n"
                f"What is the correct answer?"
            )
            sft_data = {
                "input": prompt,
                "output": data["cot"]
            }
            logger.info(f"Success extraction:\n{sft_data}")
            return sft_data
        except Exception as e:
            logger.warning(f"Failed extraction due to {e}")
    logger.info(f"Failed match. Raw_text:\n{raw_text}")
    return None


def get_prompt(sft_data):
    pi = r"\nStory:\s*(.*?)\n*Question:\s*(.*?)\n*\(A\)(.*?)\n*\(B\)(.*?)\n"
    po = r"the answer is\s*\((A|B)\)"
    try:
        mi, mo = re.findall(pi, sft_data["input"]), re.findall(po, sft_data["output"])
        if mi and mo:
            mi = mi[0]
            story, ques, a, b = mi[0].strip(), mi[1].strip(), mi[2].strip(), mi[3].strip()
            c = mo[0].strip()
            act = a if c == 'A' else b
            return NORMAL_PROMPT.format(story=story, ques=ques, act=act)
        else:
            logger.info(f"Fail to extract story, ques and act from {sft_data}")
            return None
    except Exception as e:
        logger.info(f"Failed match due to {e}. SFT_data: {sft_data}")
        return None


async def convert_to_normal(sft_data):
    prompt = get_prompt(sft_data)
    if prompt:
        response = await llm_engine.generate_async(prompt, temperature=0.7, max_tokens=512)
        if response:
            sft_data = parse_sft_data(response)
            if sft_data:
                async with aiofiles.open(result_file, "a", encoding="utf-8") as f:
                    await f.write(json.dumps(sft_data, ensure_ascii=False) + "\n")
            else:
                logger.warning(f"Failed parsing: {sft_data}")
        else:
            logger.warning(f"No Response for prompt:{prompt}")


async def sampling_async(batch_size=100):
    for start_i in range(0, len(origin_data), batch_size):
        batch_tasks = [
            asyncio.create_task(convert_to_normal(sft_data))
            for sft_data in origin_data[start_i: start_i+batch_size]
        ]
        await asyncio.gather(*batch_tasks)
    
    logger.info(f"Successfully converted the whole origin_data")
    return


if __name__ == "__main__":

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join("/workspace/v0/output/balanced_persona-normal", timestamp)
    os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.join(output_dir, "generation.log")
    result_file = os.path.join(output_dir, "output.jsonl")

    logger = setup_logger(log_file=log_file)
    llm_engine = LLMInference(
        logger=logger,
        backend="openai",
        max_concurrent_requests=96
    )

    ### Load the orginal dataset
    INPUT_FILE = "/workspace/v0/output/balanced_persona/balanced_results.jsonl"
    origin_data = []
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        for line in f:
            origin_data.append(json.loads(line))
    logger.info(f"Loading origin data {len(origin_data)}")

    ### Asyncio running
    asyncio.run(sampling_async(batch_size=200))

    logger.info("Complete!")
