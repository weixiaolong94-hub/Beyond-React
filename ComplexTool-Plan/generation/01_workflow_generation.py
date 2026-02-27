import json
import time
import os
import argparse
import random
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI, APIStatusError, APIConnectionError
from tqdm import tqdm
from utils import *

prompt_template = """
# Role
You are a top-tier system process architect with excellent creativity and logical analysis skills. Your task is to design a structurally sound and meaningful tool collaboration workflow that simulates a real-world task, based on a set of tools I provide.

# Input
I will provide you with a list of tools. Each tool is detailed with parameters such as `description`, `inputSchema`, `required`, and `name`. This is the sole basis for your design and analysis.

# Task
1.  **Deeply Understand Each Tool's Functionality**: All your designs must be strictly based on the tool definitions (especially `name`, `description`, and `inputSchema`). Never speculate on or invent functionalities or parameters that do not exist.
2.  **The Core of Dependency**: The sole condition for node B to depend on node A (represented as A->B) is that an input parameter of node B requires information that must be provided by the output of node A. This is the cornerstone of building the DAG.
3.  **DAG Topology Requirements**: These are two mandatory rules for the overall shape of the workflow, which you must strictly follow.
    3.1. **Must Contain Parallel Paths**: The middle part of the workflow must include steps that can be executed in parallel. This means there must be at least one 'fork point' (one node's output is used by multiple subsequent nodes, e.g., 2->3, 2->4). The workflow cannot be a simple linear chain.
    3.2. **Strict Layered Dependency**: This is the most critical structural requirement. The workflow must be viewable as a series of consecutive 'layers'. A node can only depend on the output of nodes from its immediately preceding layer.
    **Cross-layer dependencies are forbidden**: For example, if a path 1->2->3 exists, a 'shortcut' dependency like 1->3 is strictly prohibited. All dependencies must connect adjacent layers.
    Example: A good layered structure is 1->2, 2->3, 2->4, 3->5, 4->5, 5->6. Here, nodes 3 and 4 form a layer that depends on node 2; node 5 forms a layer that depends on nodes 3 and 4. Each layer clearly depends on its immediately preceding layer.
4.  **Strive for Meaningful Complexity**: The scenario you devise should naturally require this kind of 'phased, divided, and then consolidated' solution. For example, first prepare data (layer 1), then process different aspects of the data in parallel (layer 2), and finally merge and analyze the results (layer 3).
Note: The node numbers in the DAG must strictly correspond to the `id` of the tools. Do not change the order arbitrarily. For example, if you choose a node with id '7' as a starting point, the DAG should be presented as '7'->'8', '7'->'9', etc.

# Execution Steps
1.  **Scenario Conception**: Deeply analyze the tool list to devise a specific, valuable, real-world scenario that is naturally suited for a strictly layered, parallel model.
2.  **Tool Orchestration**: To implement the scenario, select and arrange the necessary tools. Each invoked tool is identified by its corresponding `id` from the tool list.
3.  **Dependency Analysis**: Based on the core principles of Task #2 and Task #3 (especially 3.2, layered dependency), precisely construct the dependency relationships that meet all topological requirements.
4.  **Format Output**: Package the final conceived workflow strictly according to the JSON format below.

# Output Format Requirements
Please generate **1** complex and creative DAG for me. Your final output must be a JSON object, representing a single workflow, and must strictly adhere to the following format:

```json
{
  "flow_id": "A unique, descriptive ID, for example, 'layered_market_analysis_report03'",
  "description": "A brief description of the real-world scenario this workflow simulates. The description should be in English.",
  "tool_sequence": [
    { "node": "The key corresponding to the tool", "function_id": "The 'name' from the tool definition" },
    { "node": "The key corresponding to the tool", "function_id": "The 'name' from the tool definition" },
    ...
  ],
  "DAG": "A dependency relationship string, such as '1->2, 2->3, 2->4, 3->5, 4->5, 5->6'. Each value is the corresponding key from the tool list."
}
```

**Tool List**
candidate_tools

"""

DIFFICULTY_CONFIGS = {
    'easy': {
        'min_tools': 3,
        'max_tools': 5,
        'num_samples': 1000
    },
    'medium': {
        'min_tools': 6,
        'max_tools': 8,
        'num_samples': 1000
    },
    'hard': {
        'min_tools': 9,
        'max_tools': 13, 
        'num_samples': 1000
    }
}



def main(query_list_with_ids, output_file, max_workers=5): 
    request_payloads = []
    for task_id, query in query_list_with_ids: 
        payload = {
            "model": "deepseek-chat",
            "messages": [{"role": "user", "content": query}],
            "stream": False,
            "metadata": {"task_id": task_id}  
        }
        request_payloads.append(payload)

    with open(output_file, "a", encoding="utf-8") as f:
        print(f"Starting to process {len(request_payloads)} requests. Results will be written to {output_file}")
        
        response_generator = multi_thread_openai_generator(request_payloads, max_workers)
        
        progress_bar = tqdm(response_generator, total=len(request_payloads), desc="Processing requests")
        
        for original_payload, response_content in progress_bar:
            task_id = original_payload["metadata"]["task_id"]
            if not task_id:
                print(f"\nWarning: Could not find task_id in payload. Payload: {original_payload}")
                continue
            try:
                if isinstance(response_content, str) and response_content.startswith("ERROR:"):
                    result_data = {
                        "task_id": task_id,
                        "input_query": original_payload["messages"][-1]["content"],
                        "error": response_content
                    }
                    print(f"\nRequest failed: {result_data}")

                else:
                    try:
                        response_json = json.loads(clean_json_string(response_content))
                        
                        if isinstance(response_json, list):
                            if response_json: 
                                response_json[0]['task_id'] = task_id
                        elif isinstance(response_json, dict):
                            response_json['task_id'] = task_id
                        else:
                            raise json.JSONDecodeError("LLM response is not a JSON object or list.", response_content, 0)

                        result_data = {
                            "input_query": original_payload["messages"][-1]["content"],
                            "response": json.dumps(response_json, ensure_ascii=False)
                        }

                    except json.JSONDecodeError as je:
                        print(f"\nJSON parsing failed for task_id {task_id}: {je}")
                        result_data = {
                            "task_id": task_id,
                            "input_query": original_payload["messages"][-1]["content"],
                            "error": "Failed to parse LLM response as JSON.",
                            "raw_response": response_content
                        }
                json_line = json.dumps(result_data, ensure_ascii=False)
                f.write(json_line + '\n')
                f.flush()

            except Exception as e:
                print(f"\nAn unexpected error occurred while processing task_id {task_id}: {e}")
                error_data = {
                    "task_id": task_id,
                    "error": f"An unexpected error occurred during processing: {e}",
                    "original_payload": original_payload,
                    "raw_response": str(response_content) 
                }
                f.write(json.dumps(error_data, ensure_ascii=False) + '\n')
                f.flush()

    print(f"All tasks have been processed. Results saved to {output_file}")


def make_input_data(
    difficulty: str,
    servers_path: str = "./server_tool_output.json",
    output_log_dir: str = "./result_data"
) -> tuple[list[tuple[str, str]], str]:
    """
    Generates toolsets, task IDs, and prompts based on the specified difficulty level.

    Args:
        difficulty (str): The difficulty level ('easy', 'medium', 'hard').
        servers_path (str): The path to the source JSON file containing tools.
        output_log_dir (str): The base directory where log files will be saved.

    Returns:
        tuple[list[tuple[str, str]], str]:
            - A list of tuples, where each tuple contains a (task_id, prompt).
            - The full path to the generated toolset log file. Returns an empty string on failure.
    """
    if difficulty not in DIFFICULTY_CONFIGS:
        print(f"Error: Unknown difficulty level '{difficulty}'. Available: {list(DIFFICULTY_CONFIGS.keys())}")
        return [], ""

    config = DIFFICULTY_CONFIGS[difficulty]
    min_t, max_t, num_samples = config['min_tools'], config['max_tools'], config['num_samples']
    
    print(f"\n--- Generating data for {difficulty.upper()} difficulty ---")
    print(f"Configuration: Tool count [{min_t}-{max_t}], Target samples: {num_samples}")


    try:
        with open(servers_path, "r", encoding="utf-8") as f:
            all_servers = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error: Could not load or parse servers file at '{servers_path}'. Error: {e}")
        return [], ""

    eligible_servers = [s for s in all_servers if len(s.get('tools', [])) >= max_t]
    if not eligible_servers:
        print(f"Warning: No servers found with at least {max_t} tools to generate '{difficulty}' data.")
        return [], ""
    
    print(f"Found {len(eligible_servers)} eligible servers for sampling.")

    prompts_with_ids = []
    generated_toolsets = []

    for _ in tqdm(range(num_samples), desc=f"Generating {difficulty} toolsets"):
        selected_server = random.choice(eligible_servers)
        num_tools_to_sample = random.randint(min_t, max_t)
        
        sampled_tools = random.sample(selected_server['tools'], num_tools_to_sample)
        
        toolset_dict = {str(i): tool for i, tool in enumerate(sampled_tools, start=1)}
        
        task_id = str(uuid.uuid4())
        
        prompt_str = prompt_template.replace("candidate_tools", json.dumps(toolset_dict, ensure_ascii=False))
        
        prompts_with_ids.append((task_id, prompt_str))
        generated_toolsets.append({"task_id": task_id, "tools": toolset_dict})

    output_log_path = os.path.join(output_log_dir, difficulty, "random_tools.log")
    try:
        os.makedirs(os.path.dirname(output_log_path), exist_ok=True)
        with open(output_log_path, "w", encoding="utf-8") as f:
            for toolset_log in generated_toolsets:
                f.write(json.dumps(toolset_log, ensure_ascii=False) + "\n")
        print(f"Successfully wrote {len(generated_toolsets)} toolset logs to {output_log_path}")
    except IOError as e:
        print(f"Error: Could not write to log file at '{output_log_path}'. Error: {e}")
        return prompts_with_ids, "" 

    return prompts_with_ids, output_log_path
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate workflow datasets for a specified difficulty.")
    parser.add_argument(
        '--difficulty',
        type=str,
        default='easy',
        choices=['easy', 'medium', 'hard'],
        help="The difficulty level of the data to generate (e.g., 'easy', 'medium', 'hard')"
    )
    parser.add_argument(
        '--base_dir',
        type=str,
        default='./result_data',
        help="The root directory to store data for all difficulty levels"
    )
    parser.add_argument(
        '--servers_file',
        type=str,
        default='./server_tool_output.json', 
        help="The JSON file containing all server and tool definitions"
    )
    parser.add_argument(
        '--workers',
        type=int,
        default=50,
        help="Number of concurrent worker threads"
    )

    args = parser.parse_args()

    prompts_to_process, _ = make_input_data(
        difficulty=args.difficulty,
        servers_path=args.servers_file,
        output_log_dir=args.base_dir
    )

    if not prompts_to_process:
        print(f"No data was generated for difficulty '{args.difficulty}'. Terminating.")
    else:
        workflow_output_file = os.path.join(args.base_dir, args.difficulty, "tool2dag.log")

        main(
            query_list_with_ids=prompts_to_process,
            output_file=workflow_output_file,
            max_workers=args.workers
        )
        print(f"\n--- Data generation for {args.difficulty.upper()} difficulty is complete ---")
        