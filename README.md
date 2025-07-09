<div align="center">

# 🌳 [TreeSynth: Synthesizing Diverse Data from Scratch via Tree-Guided Subspace Partitioning](https://arxiv.org/pdf/2503.17195)

</div>

<div align="center">
  <a href="https://arxiv.org/pdf/2503.17195"><img src="https://img.shields.io/badge/Paper-arXiv-red" alt="arXiv"></a>
  <!-- <a href="https://huggingface.co/datasets/proj-persona/PersonaHub"><img src="https://img.shields.io/badge/Dataset-%F0%9F%A4%97%20Hugging_Face-yellow" alt="Hugging Face"></a> -->
  <a href="https://creativecommons.org/licenses/by-nc-sa/4.0/"><img src="https://img.shields.io/badge/Data%20License-CC_BY_NC_SA_4.0-blue" alt="Data License"></a>
  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/Code%20License-MIT-blue" alt="Code License"></a>
</div>

<div align="center">
<img src="./image/cartoon_tree.jpg" width="100%">
</div>

## 🚀 News

- **[5/16/2025]** 🚀 We're excited to announce the release of our new paper and the initial public availability of its corresponding code.


## 💡 Introduction
We introduce **TREESYNTH**, a tree-guided subspace-based data synthesis approach inspired by decision trees. It constructs a spatial partitioning tree to recursively divide a task-specific full data space (i.e., root node) into numerous atomic subspaces (i.e., leaf nodes) with mutually exclusive and exhaustive attributes to ensure both distinctiveness and comprehensiveness before synthesizing samples within each atomic subspace.
<div align="center">
<img src="./image/Tree.jpg" width="90%">
</div>
This globally dividing-and-synthesizing method finally collects subspace samples into a comprehensive dataset, effectively circumventing repetition and space collapse to ensure the diversity of large-scale data synthesis. 

### 🔧 Algorithm Overview
The core TreeSynth algorithm follows a systematic tree-guided approach for data synthesis:

<div align="center">
<img src="./image/pseudo_code.png" width="80%">
</div>
<small><em>Pseudocode of the TreeSynth algorithm showing the tree construction and data synthesis process.</em></small>

<div align="center">
<img src="./image/TreeSynth.jpg" width="100%">
</div>
<small><em>Schematic illustrating TreeSynth’s advantage in spatial partitioning.</em></small>


## 📊 Performance
### 🌈 Excellent Data Diversity
TreeSynth exhibits substantially better data diversity and more comprehensive coverage across various tasks and models than both human-curated datasets and peer synthetic methods.
<div align="center">
<img src="./image/Diversity.jpg" width="100%">
</div>
<small><em> t-SNE visualization of LLaMA3.3-70B-Instruct-synthesized datasets for various methods across GSM8K, MATH, and Code Alpaca styles.</em></small>

### 🏆 Superior Downstream Performance
Models trained on TreeSynth data consistently outperform those trained on both human-crafted datasets and synthetic baselines across all the tasks, foundation and generation models
<div align="center">
<img src="./image/performance_GPT.jpg" width="100%">
</div>
<small><em>Model performance and data diversity comparison of various methods with GPT-4o-powered data synthesis.</em></small>

### 📈 Scalable Data Synthesis with Quality Preservation
With the global data spatial perspective guided by tree structure, TreeSynth effectively scales datasets while preserving data quality, suggesting great scalability wherein downstream performance consistently improves with increased data volume.

<div align="center">
<img src="./image/scaling_GPT.png" width="100%">
</div>
<small><em>Model performance trends across data scales for different methods powered by GPT-4o.</em></small>

<details><summary>Click to expand performance results of other models using TreeSynth</summary>

#### Performance results using TreeSynth with LLaMA 3.3 70B Instruct for data synthesize
<div align="center">
<img src="./image/performance_LLaMA.png" width="100%">
</div>
<small><em>Model performance and data diversity comparison of various methods with LLaMA3.3 70b Instruct-powered data synthesis.</em></small>

<div align="center">
<img src="./image/scaling_LLaMA.png" width="100%">
</div>
<small><em>Model performance trends across data scales for different methods powered by LLaMA3.3 70b Instruct.</em></small>

#### Performance results using TreeSynth with Qwen 2.5 72B Instruct for data synthesize
<div align="center">
<img src="./image/performance_Qwen.png" width="100%">
</div>
<small><em>Model performance and data diversity comparison of various methods with Qwen2.5 72b Instruct-powered data synthesis.</em></small>

<div align="center">
<img src="./image/scaling_Qwen.png" width="100%">
</div>
<small><em>Model performance trends across data scales for different methods powered by Qwen2.5 70b Instruct.</em></small>

</details>

## ⚡ Quick Start

### 🛠️ Prerequisites

- Python 3.8+
- API access to OpenAI/Azure or local model deployment via vLLM
- Required Python dependencies (see installation below)

### 📥 Installation

```bash
git clone https://github.com/cpa2001/TreeSynth.git
cd TreeSynth

# Install dependencies
pip install -r requirements.txt
```

### 📁 Project File Structure

```
TreeSynth/
├── config.py                    # Configuration: API settings, generation parameters
├── main.py                      # Main entry point: data generation workflow control
├── generator_math_async.py      # MATH style dataset generator
├── generator_code_async.py      # Code Alpaca style dataset generator
├── generator_gsm_async.py       # GSM8K style dataset generator
├── generator_mental_async.py    # SimpleToM style dataset generator
├── generator_mental_positive.py # Positive SimpleToM style dataset generator
├── balance_mental.py            # SimpleToM style data balancing tool
├── vllm_engine.sh               # vLLM local deployment script
├── vllm_chat_template_llama3.1_json.jinja  # vLLM chat template
└── image/                       # Documentation images directory
```

### ⚙️ Configuration

#### 1️⃣ Backend Setup

Edit `config.py` to configure your API backend:

**Option A: OpenAI API**
```python
BACKEND_CONFIGS = {
    "openai": {
        "api_key": "your-openai-api-key",
        "model_name": "gpt-4o",
    }
}
```

**Option B: Azure OpenAI**
```python
BACKEND_CONFIGS = {
    "azure": {
        "api_key": "your-azure-api-key",
        "model_name": "gpt-4o",
        "azure_endpoint": "https://your-endpoint.openai.azure.com/",
        "api_version": "2024-10-21",
    }
}
```

**Option C: vLLM (Local Deployment)**
```python
BACKEND_CONFIGS = {
    "vllm": {
        "api_base": "http://localhost:8000/v1",
        "api_key": "your-vllm-api-key",
        "model_name": "/path/to/model/qwen2_5-72b-instruct",
    }
}
```

For vLLM local deployment, use the provided script:
```bash
bash vllm_engine.sh
```

#### 2️⃣ Tree Generation Parameters

Configure core TreeSynth parameters in `DEFAULT_CONFIG` within `config.py`:

```python
DEFAULT_CONFIG = {
    "max_depth": 4,                          # Maximum tree depth
    "num_samples_per_node": 10,              # Samples generated per node
    "max_attribute_count": 50,               # Maximum attributes per node
    "max_sample_infinite_attribute": 50,     # Threshold for infinite attributes
    "infinite_path_samples": 10,             # Samples for infinite paths
    "max_workers": 64,                       # Concurrent workers
    "max_concurrent_requests": 64,           # Concurrent API requests
    "max_retries": 5,                        # Maximum retry attempts
}
```

#### 3️⃣ API Pool Configuration (Optional)

For high-throughput data generation, configure multiple API endpoints:

```python
# Multiple Azure endpoints for load balancing
OPENAI_API_POOL = [
    {
        "endpoint": "https://endpoint1.openai.azure.com/",
        "key": "azure-api-key1",
        "version": "2024-10-21",
        "model": "gpt-4o",
    },
    {
        "endpoint": "https://endpoint2.openai.azure.com/",
        "key": "azure-api-key2",
        "version": "2024-10-21",
        "model": "gpt-4o",
    }
]
```

### 🔄 Switching Data Generation Styles

TreeSynth supports multiple data styles. You need to manually modify the import statement in `main.py`:

#### 1️⃣ MATH Dataset Style
```python
# Modify import in main.py
from generator_math_async import LLMInference, APIPool, TreeNode, inject_runtime_to_tree
```

#### 2️⃣ Code Alpaca Style
```python
# Modify import in main.py
from generator_code_async import LLMInference, APIPool, TreeNode, inject_runtime_to_tree
```

#### 3️⃣ GSM8K Style
```python
# Modify import in main.py
from generator_gsm_async import LLMInference, APIPool, TreeNode, inject_runtime_to_tree
```

### ▶️ Running TreeSynth

#### Step-by-Step Guide:

1. **Configure API Keys**:
   - Edit `BACKEND_CONFIGS` in `config.py`
   - Replace API keys, endpoints with your actual configurations

2. **Select Backend**:
   - Edit lines 175-184 in `main.py`
   - Comment out unused backends, enable the one you want to use
   ```python
   # Use vLLM backend
   # asyncio.run(run_generation_process(..., backend="vllm", ...))
   
   # Use OpenAI backend  
   asyncio.run(run_generation_process(..., backend="openai", ...))
   ```

3. **Choose Data Style**:
   - Modify the import statement on line 13 of `main.py`:
   ```python
   # Math competition problems
   from generator_math_async import LLMInference, APIPool, TreeNode, inject_runtime_to_tree
   
   # Or code generation  
   # from generator_code_async import LLMInference, APIPool, TreeNode, inject_runtime_to_tree
   ```

4. **Run Generation**:

```bash
# Execute data generation pipeline
python main.py
```

#### Output Files:

The program will generate files in `output/timestamp/` directory:
- `generation.log` - Detailed generation logs
- `output.json` - Complete tree structure and intermediate results
- `tree_structure.txt` - Human-readable tree structure visualization
- `result.jsonl` - Final generated data samples (one JSON object per line)

### 🧩 Custom Data Domains

To create generators for new task domains, reference existing generator files:

1. **Copy Existing Generator**: Use `generator_math_async.py` as template
2. **Modify Prompt Templates**: Edit prompts in `format_dim_prompt()`, `format_expand_prompt` and `format_gen_prompt()`
3. **Adjust Parsing Logic**: Modify regex pattern matching
4. **Update Imports**: Import the new generator in `main.py`



## 🗺️ Overview of TreeSynth
TreeSynth consists of two key stages: data space partitioning and subspace data synthesis.

During the former phase, TreeSynth employs a spatial partitioning tree to recursively divide a task-specific whole data space (i.e., root node defined by textual descriptions) into numerous atomic subspaces (i.e., leaf nodes). These subspaces are characterized by mutually exclusive and exhaustive attribute values to ensure both distinctiveness and diversity.

In the subsequent subspace data synthesis phase, samples are generated within each subspace separately, before collecting them as a diverse and comprehensive dataset. By employing this globally divide-and-synthesize methodology, TreeSynth effectively prevents repetition and space collapse to ensure the diversity and completeness of large-scale data synthesis, successfully avoiding the drawbacks of previous methods.
<div align="center">
<img src="./image/Tree_example.jpg" width="100%">
</div>
<small><em>A spatial partitioning tree visualization of TreeSynth, exemplified through GSM8K-style data synthesis.</em></small>

## 📚 Citation
If you found our work useful, please consider starring and citing. Thank you!
```latex
@article{wang2025treesynth,
  title={TreeSynth: Synthesizing Diverse Data from Scratch via Tree-Guided Subspace Partitioning},
  author={Wang, Sheng and Chen, Pengan and Zhou, Jingqi and Li, Qintong and Dong, Jingwei and Gao, Jiahui and Xue, Boyang and Jiang, Jiyue and Kong, Lingpeng and Wu, Chuan},
  journal={arXiv preprint arXiv:2503.17195},
  year={2025}
}
```

[//]: # ()
[//]: # (## Star History)

[//]: # ()
[//]: # ([TreeSynth]&#40;https://github.com/cpa2001/TreeSynth&#41; 当前 star 数为 0，暂无增长趋势。)

