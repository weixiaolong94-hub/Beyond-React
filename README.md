<div align="center">

<h2>Beyond ReAct: A Planner-Centric Framework for Complex Tool-Augmented LLM Reasoning</h2>

</div>

<div align="center">
  
</div>

**AAAI 2026 • Poster**

This is the official repository for the paper "[Beyond ReAct: A Planner-Centric Framework for Complex Tool-Augmented LLM Reasoning](https://arxiv.org/abs/2511.10037)".
Our paper was accepted at the **AAAI 2026**.


<div align="center">
    <img src="./Features/paper.png" width="90%" height="auto" />
</div>


---

## 📂 Repository Structure

```text
├── ComplexTool-Plan/            # Large-scale benchmark and training data
│   ├── train_data/              # Data
│   ├── generation/              # Scripts for generating complex DAG workflows
│   └── evaluate/                # Graded evaluation sets (Easy & Hard splits)
├── RL_Hierarchical_Rewards/     # GRPO Reinforcement Learning logic
│   └── reward_function.py       # Multi-level reward (Syntax, Cycle, Connectivity, Fidelity)
├── StableToolBench/             # End-to-end execution evaluation environment
│   ├── toolbench/               # Core execution engine
│   └── solvable_queries/        # Standardized test cases for SOTA comparison
├── Scripts/                     # Utility scripts for training and inference
└── Features/                    # Documentation and technical illustrations
```

---

## 🚀 Key Features

### 1. 🏗️ Planner-Centric DAG Orchestration
Unlike the reactive "step-by-step" approach of ReAct, our framework **decouples planning from execution**. The dedicated Planner model generates a **Directed Acyclic Graph (DAG)** in a single forward pass, enabling:
*   **Global Optimization**: Resolving local bottlenecks by considering the entire workflow upfront.
*   **Maximum Parallelism**: Identifying independent tool calls that can be executed simultaneously to reduce latency.
*   **Location:** See `ComplexTool-Plan/generation` for the DAG generation logic.

### 2. 📊 ComplexTool-Plan: A Graded Benchmark
We introduce a large-scale, high-fidelity dataset featuring **4,500+ real-world APIs**. It serves as both a training ground and a rigorous evaluation standard:
*   **Multi-level Difficulty**: Includes **Easy** (parallel) and **Hard** (nested logic) queries to evaluate agentic boundaries.
*   **High Fidelity**: All query-DAG pairs are reverse-engineered to ensure strict logical alignment.
*   **Location:** Data available in `ComplexTool-Plan/train_data` and `ComplexTool-Plan/evaluate`.

### 3. ⚙️ Two-Stage RL Training (SFT + GRPO)
We elicit advanced reasoning capabilities through a systematic two-stage training methodology:
*   **Supervised Fine-Tuning (SFT)**: Provides a structured "cold start" for DAG syntax.
*   **GRPO with Hierarchical Rewards**: We use **Group Relative Policy Optimization** guided by a **Hierarchical Reward Function** that evaluates:
    *   *Level 1 & 2*: Structural integrity (Syntax, No-Cycle, Connectivity).
    *   *Level 3*: Logical fidelity (Edge-level F1 match and Perfect Match bonus).
*   **Location:** Reward logic implemented in `RL_Hierarchical_Rewards`.

### 4. ⚡ SOTA Performance & Efficiency
Beyond ReAct establishes a new performance frontier in tool-augmented reasoning:
*   **Superior Success Rate**: Outperforms GPT-4 (ReAct) on the **StableToolBench** by a significant margin.
*   **High Efficiency**: Reduces the average reasoning process to just **2.29 steps**, significantly faster than iterative parallel frameworks.
*   **Location:** Evaluation environment and test cases in `StableToolBench`.

### 5. 🛡️ Robust Scaling & Resilience
Our framework demonstrates a clear **scaling law** in planning robustness. As task complexity grows (from Easy to Hard), the DAG-optimized Planner exhibits a much more graceful performance decline compared to sequential models, proving its resilience in handling intricate, multi-tool workflows.

---



---

## 🚀 Getting Started

### 1. Environment Setup

We recommend using separate environments or a unified container for SFT and RL tasks.

#### **Supervised Fine-Tuning (via LLaMA-Factory)**
```bash
# Install core dependencies
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 vllm==0.6.5

# Setup LLaMA-Factory
git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory
pip install -e ".[metrics,vllm]"
cd ..
```

#### **Reinforcement Learning (via verl)**
```bash
# Setup verl for GRPO
git clone https://github.com/volcengine/verl.git
cd verl
pip install -e .

# Install execution backends (vLLM & Flash-Attention)
# Use USE_MEGATRON=0 for FSDP (Recommended for SLMs)
USE_MEGATRON=0 bash scripts/install_vllm_sglang_mcore.sh
cd ..
```

---

### 📚 ComplexTool-Plan Dataset Pipeline

The dataset generation follows a strict three-stage reverse-engineering pipeline located in `./ComplexTool-Plan/generation/`.

| Stage | Description | Command |
| :--- | :--- | :--- |
| **Step 1** | **Workflow Generation**: Author DAG plans from tool library. | `python generation/01_workflow.py --difficulty hard --workers 20` |
| **Step 2** | **Query Reverse-Engineering**: Generate intents from DAGs. | `python generation/02_reverse.py --difficulty hard --workers 20` |
| **Step 3** | **Validation**: Re-plan to filter high-fidelity pairs. | `python generation/03_replan.py --difficulty hard --workers 20` |

---

### 🛠️ Training Stage 1: Supervised Fine-Tuning

We perform full-parameter SFT on Qwen3 models to establish basic DAG syntax capabilities.

1.  **Data Preparation**: Move your generated `train.json` to `./LLaMA-Factory/data/`.
2.  **Register Dataset**: Add your dataset entry to `./LLaMA-Factory/data/dataset_info.json`:
    ```json
    "complextool_sft": { "file_name": "train.json" }
    ```
3.  **Launch Training**:
    ```bash
    llamafactory-cli train examples/train_full/qwen3_8b_full_sft.yaml
    ```

---

### ⚙️ Training Stage 2: RL with Hierarchical Rewards

We use the **GRPO** algorithm to refine the Planner's global awareness. The core innovation is our **Hierarchical Reward Function**, which terminates early on structural failures.

*   **Reward Logic**: Located in `./RL_Hierarchical_Rewards/reward_function.py`.
*   **Hierarchical Levels**: 
    1.  **Syntax & Cycle**: Critical structural checks (-10.0 penalty).
    2.  **Connectivity**: Semantic coherence check (-2.0 penalty).
    3.  **Fidelity**: Edge-level F1 score and Perfect Match bonus (+5.0).

*   **Run GRPO**:
    ```bash
    # Ensure your SFT checkpoint is ready as the actor's initial weights
    python -m verl.trainer.main_ppo \
        algorithm.type=grpo \
        model.partial_pretrain=path/to/sft_checkpoint \
        reward_model.reward_algo=hierarchical \
        trainer.n_gpus=8 \
        ... \
    ```

---

### 📊 Evaluation & Benchmarking

#### **Phase 1: Planning Accuracy (DAG-level)**
Run batch inference and calculate node/edge-level metrics on the `ComplexTool-Plan` evaluation set.
```bash
cd ./ComplexTool-Plan/evaluate

# 1. Batch Inference with vLLM
python inference.py --model_path ../../models/qwen3_8b_rl --query_path ./hard/evaluate.json

# 2. Results Merging & Cleaning
python data_process.py --output_path ./results/raw_output.json --evaluate_path ./hard/evaluate.json

# 3. Calculate Precision/Recall/F1/EM
python accuracy.py --input_file ./results/merged_final.json
```


#### **Phase 2: End-to-End Execution & Evaluation**
We evaluate the integrated system on **StableToolBench** to measure real-world performance. This phase involves deploying a virtual API server, running the parallel execution pipeline, and conducting a multi-dimensional analysis (Success, Preference, and Efficiency).

### 1. The Virtual API Environment
To ensure reproducibility and handle API instability, we use a virtual server that intercepts requests.

**Response Logic:**
1. **Cache Hit**: Returns pre-recorded responses from the local database.
2. **Real API**: If missing, calls the live RapidAPI/ToolBench server.
3. **LLM Simulation**: If the live API fails, `gpt-4-turbo-preview` generates a logically consistent response.

**Server Setup:**
*   **Data Preparation**: Download `toolenv` and `tool_response_cache` from the [ToolBench Data Release](https://github.com/OpenBMB/ToolBench).
*   **Configuration**: Update `StableToolBench/server/config.yml`:
    ```yaml
    api_key: "YOUR_OPENAI_KEY"
    model: ""
    toolbench_url: ""
    is_save: True  # Save new simulations to ./tool_response_new_cache
    port: 
    ```
*   **Launch Server**:
    ```bash
    cd StableToolBench/server
    python main.py
    ```

---

### 2. Inference Pipeline
Our framework decouples the **Planner** (Qwen-RL) from the **Executor** (GPT-4o/ToolLLaMA).

#### **Directory Requirements**
Ensure your data is organized as follows for the evaluation scripts to recognize:
```text
data/
└── toolenv/             # API definitions
output/
├── answer/              # Raw model outputs
├── model_predictions/   # Converted format for GPT evaluation
└── pass_rate_results/   # Final metric files
```

#### **Execution Parameters**
We provide a unified script `run_qa_pipeline_multithread.sh` with the following key arguments:
- **`backbone_model`**: `chatgpt_function` (Proprietary) or `toolllama_net` (Open-source/Ours).
- **`method`**:
    - `CoT@1`: Standard **ReAct** baseline.
    - `DFS_woFilter_w2`: **DFSDT** tree-search baseline.
    - `DAG_Plan`: **Our Beyond ReAct** (Planner-centric DAG).
- **`test_set`**: Covers 6 subsets (e.g., `G1_instruction` for I1-Inst, `G3_instruction` for I3-Inst).
- **`model_url`**: The endpoint for your trained Planner/Executor.

**Run Inference Example:**
```bash
# Evaluate Beyond ReAct (DTA-Llama) on I3-Inst subset
sh run_qa_pipeline_multithread.sh \
    toolllama_net \
    DAG_Plan \
    G3_instruction \
    http://YOUR_SERVER_IP:PORT/llama_parse_parallel
```

---

### 3. Comprehensive Performance Evaluation
Beyond simple success rates, we conduct a granular analysis of the agent's behavior.

#### ** Success & Preference Metrics**
We use the **StableToolEval** protocol (GPT-4 as a judge) to ensure objective assessment:
1.  **Standardization**: Convert raw JSON logs into the evaluatable format.
    ```bash
    sh run_convert_answer.sh ./output/answer/DAG_Plan_G3
    ```
2.  **Solvable Pass Rate (SoPR)**: Measures absolute task completion.
    ```bash
    sh run_pass_rate.sh ./output/model_predictions/DAG_Plan_G3 G3_instruction
    ```
3.  **Solvable Win Rate (SoWR)**: A head-to-head comparison against the GPT-3.5 ReAct baseline.
    ```bash
    sh run_preference.sh ./output/model_predictions/DAG_Plan_G3 G3_instruction
    ```

---
## Citation
If you find the project helpful, please cite:
*   [Beyond ReAct: A Planner-Centric Framework for Complex Tool-Augmented LLM Reasoning](https://arxiv.org/abs/2511.10037)
```bash
@article{wei2025beyond,
  title={Beyond ReAct: A Planner-Centric Framework for Complex Tool-Augmented LLM Reasoning},
  author={Wei, Xiaolong and Dong, Yuehu and Wang, Xingliang and Zhang, Xingyu and Zhao, Zhejun and Shen, Dongdong and Xia, Long and Yin, Dawei},
  journal={arXiv preprint arXiv:2511.10037},
  year={2025}
}
```
