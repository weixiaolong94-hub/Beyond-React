import json
import time
import os
import argparse 
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI, APIStatusError, APIConnectionError
from tqdm import tqdm
from utils import * 
import uuid

prompt_template = """
#### **Role and Core Task**
You are a **top-tier System Process Architect** and **Intelligent Task Planner**, specializing in parsing complex human language instructions and decomposing them into efficient, automated execution plans (DAGs). Your core task is to:
- Based on the user's **complex Query** and the available **Tool List**, design a logically sound, efficient, and comprehensive tool collaboration workflow (DAG) that fully satisfies the user's requirements.
- The final output must be a **precise DAG execution plan**, ensuring all steps have correct dependencies and parallelism is maximized where possible.

---

#### **Execution Steps**
1. **Query Decomposition**: Analyze the user query, break it down into actionable sub-tasks, and identify key entities (such as time, location, subject, etc.).
2. **Task-to-Tool Mapping**: For each sub-task, match it with the most appropriate tool from the tool library (select by tool ID).
3. **Dependency Analysis**: Determine which steps need to wait for the results of other steps (e.g., you must 'search for a hotel' before you can 'book a hotel').
4. **DAG Construction & Formatting**: Synthesize the dependencies into a DAG string and strictly adhere to the JSON format for the output.

---

#### **Output Format**
2. **JSON Format**:
```json
{
    "DAG": "A dependency string (e.g., '1->2, 2->3, 2->4, 3->5, 4->5, 5->6')"
}
```
The user input is:
mcp_input
"""


def main(query_list, output_file, max_workers=5):
    """
    Prepares requests, calls a multi-threaded generator, and processes/writes results in real-time.
    
    Args:
        query_list (list): A list containing all user prompts (queries).
        output_file (str): The path to the file where results will be written.
        max_workers (int): The number of concurrent worker threads.
    """
    # Prepare payloads for all API requests.
    request_payloads = []
    for query in query_list:
        payload = {
            "model": "deepseek-chat",
            "messages": [
                {
                    "role": "user",
                    "content": query
                },
            ],
            "stream": False 
        }
        request_payloads.append(payload)

    # Open the output file in append mode to avoid overwriting previous results.
    with open(output_file, "a", encoding="utf-8") as f:
        print(f"Starting to process {len(request_payloads)} requests. Results will be written to {output_file}")
        
        # Call the generator that handles multi-threaded API calls.
        response_generator = multi_thread_openai_generator(request_payloads, max_workers)
        
        # Wrap the generator with tqdm to display a progress bar.
        progress_bar = tqdm(response_generator, total=len(request_payloads), desc="Processing requests")
        
        # Iterate through the responses as they are completed.
        for original_payload, response_content in progress_bar:
            try:
                # Check if the response is an error message.
                if isinstance(response_content, str) and response_content.startswith("ERROR:"):
                    # Build an error log record.
                    result_data = {
                        "input_query": original_payload["messages"][-1]["content"],
                        "error": response_content
                    }
                    print(f"\nRequest failed: {result_data}")
                else:
                    # If successful, build a success record.
                    result_data = {
                        "input_query": original_payload["messages"][-1]["content"],
                        "response": response_content
                    }
                
                # Convert the result to a JSON string and write it to the file.
                json_line = json.dumps(result_data, ensure_ascii=False)
                f.write(json_line + '\n')
                f.flush() # Ensure data is written to disk immediately.

            except Exception as e:
                # Handle any other unexpected errors during processing.
                print(f"\nAn error occurred while processing a result: {e}")
                error_data = {
                    "error": f"Failed to process response: {e}",
                    "original_payload": original_payload,
                    "raw_response": response_content
                }
                # f.write(json.dumps(error_data, ensure_ascii=False) + '\n')
                # f.flush()

    print(f"All tasks have been processed. Results saved to {output_file}")


def _load_toolsets_by_id(log_path: str) -> dict[str, dict]:
    """Loads toolsets from a log file and builds a dictionary keyed by task_id."""
    toolsets_map = {}
    try:
        with open(log_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    data = json.loads(line)
                    if "task_id" in data and "tools" in data:
                        toolsets_map[data["task_id"]] = data["tools"]
                except json.JSONDecodeError:
                    continue
        print(f"Successfully loaded {len(toolsets_map)} toolsets with IDs from {log_path}.")
    except FileNotFoundError:
        print(f"Error: Toolset log file not found at '{log_path}'.")
    return toolsets_map

def make_input_data(
    difficulty: str,
    base_data_dir: str
) -> list[str]:
    """
    Generates prompts for the re-planning task by matching queries with their corresponding toolsets.

    Args:
        difficulty (str): The difficulty level to process ('easy', 'medium', 'hard').
        base_data_dir (str): The base directory containing all difficulty-level data.

    Returns:
        list[str]: A list of fully-formed prompts ready for the LLM.
    """
    print(f"\n--- Preparing re-planning data for {difficulty.upper()} difficulty ---")
    

    query_log_path = os.path.join(base_data_dir, difficulty, "dag2query.log")
    tools_log_path = os.path.join(base_data_dir, difficulty, "random_tools.log")

    # Load all toolsets into a dictionary for efficient lookup.
    toolsets_by_id = _load_toolsets_by_id(tools_log_path)
    if not toolsets_by_id:
        return []


    generated_prompts = []
    total_queries, matched_queries = 0, 0

    try:
        with open(query_log_path, "r", encoding="utf-8") as f:
            for line in f:
                total_queries += 1
                try:
                    line_data = json.loads(line)
                    query_info = json.loads(clean_json_string(line_data['response']))
                    
                    task_id = query_info.get("task_id")
                    query = query_info.get("query")

                    # If a query has a matching toolset, create a prompt.
                    if task_id and query and task_id in toolsets_by_id:
                        tool_set = toolsets_by_id[task_id]
                        prompt_input = {"query": query, "tool_sequence": tool_set}
                        prompt_str = prompt_template.replace('mcp_input', json.dumps(prompt_input, ensure_ascii=False))
                        generated_prompts.append(prompt_str)
                        matched_queries += 1
                except (json.JSONDecodeError, KeyError, TypeError):
                    continue
    except FileNotFoundError:
        print(f"Error: Query log file not found at '{query_log_path}'.")
        return []

    print("-" * 20)
    print("Data preparation summary:")
    print(f"  Total queries processed: {total_queries}")
    print(f"  Successfully matched and generated prompts: {matched_queries}")
    print(f"  Unmatched or malformed queries: {total_queries - matched_queries}")
    print("-" * 20)

    return generated_prompts
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate re-planning DAGs for queries of a specified difficulty.")
    
    parser.add_argument(
        '--difficulty', 
        type=str, 
        default='easy',
        choices=['easy', 'medium', 'hard'],
        help="The difficulty level of the data to process"
    )
    parser.add_argument(
        '--base_dir',
        type=str,
        default='./result_data',
        help="The root directory containing data for all difficulty levels"
    )
    parser.add_argument(
        '--workers',
        type=int,
        default=50,
        help="Number of concurrent worker threads"
    )

    args = parser.parse_args()
    
    prompts_to_process = make_input_data(
        difficulty=args.difficulty,
        base_data_dir=args.base_dir
    )

    if not prompts_to_process:
        print(f"No prompts were generated for re-planning for difficulty '{args.difficulty}'. Terminating.")
    else:
        output_plan_file = os.path.join(args.base_dir, args.difficulty, "query2plan.log")
        
        os.makedirs(os.path.dirname(output_plan_file), exist_ok=True)
        
        main(
            query_list=prompts_to_process,
            output_file=output_plan_file,
            max_workers=args.workers
        )