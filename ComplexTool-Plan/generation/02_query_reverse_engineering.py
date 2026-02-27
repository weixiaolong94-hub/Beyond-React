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
# Role
You are a User Intent Architect. Your core mission is to reverse-engineer a complex machine process, composed of detailed tool definitions (Tool Schemas) and workflow logic (DAG), into a single, coherent, and intent-rich human natural language query.

# Goal
Based on the workflow definition and the complete set of tool schemas I provide, your task is to synthesize a **highly complex natural language query**. The complexity of this query must **directly reflect**:
1.  **The workflow's structure**: It must reflect the serial, parallel, fan-out, and fan-in logic within the DAG.
2.  **The tools' details**: It must incorporate the `required` parameters, `optional` parameters, `default` values, and the domain knowledge found in the `description` of all relevant tools.

# Input
I will provide you with a JSON object containing two parts: `workflow` and `tool_schemas`.

*   `workflow`: Describes the high-level process, including `flow_id`, `description`, `tool_sequence`, and `DAG`.
*   `tool_schemas`: A list containing the complete definition for each tool. You will need to look up detailed information here based on the `function_id` in the `tool_sequence`.

```json
{
  "workflow": {
    "flow_id": "patient_case_review",
    "description": "Retrieves a patient's latest lab results and imaging reports based on their case information, then generates a comprehensive clinical summary.",
    "tool_sequence": [
      { "node": "8", "function_id": "find_patient_case" },
      { "node": "2", "function_id": "get_lab_results" },
      { "node": "3", "function_id": "get_imaging_reports" },
      { "node": "4", "function_id": "generate_clinical_summary" }
    ],
    "DAG": "8->2,8->3,2->4,3->4"
  },
  "tool_schemas": [
    {
      "name": "find_patient_case",
      "description": "Finds the most recent patient case based on the patient's name or medical record number.",
      "inputSchema": {
        "type": "object",
        "properties": {
          "patient_name": { "description": "Patient's name", "type": "string" },
          "mrn": { "description": "Medical Record Number (MRN)", "type": "string" }
        },
        "required": ["patient_name"]
      }
    },
    {
      "name": "get_lab_results",
      "description": "Retrieves specified laboratory test results based on case information.",
      "inputSchema": {
        "type": "object",
        "properties": {
          "case_info": { "description": "Case information from find_patient_case", "type": "object" },
          "test_types": { "description": "The types of tests to query", "type": "array", "items": { "type": "string" }, "default": ["Complete Blood Count", "Comprehensive Metabolic Panel"] }
        },
        "required": ["case_info"]
      }
    },
    {
      "name": "get_imaging_reports",
      "description": "Retrieves specified imaging reports based on case information.",
      "inputSchema": {
        "type": "object",
        "properties": {
          "case_info": { "description": "Case information from find_patient_case", "type": "object" },
          "modality": { "description": "Imaging modality, e.g., CT, MRI, X-ray", "type": "string" }
        },
        "required": ["case_info"]
      }
    },
    {
      "name": "generate_clinical_summary",
      "description": "Integrates lab results and imaging reports to generate a clinical summary for physicians.",
      "inputSchema": {
        "type": "object",
        "properties": {
          "lab_results_text": { "description": "Lab results text", "type": "string" },
          "imaging_reports_text": { "description": "Imaging reports text", "type":"string" }
        },
        "required": ["lab_results_text", "imaging_reports_text"]
      }
    }
  ]
}
```
# Core Instruction: How to Synthesize a Complex Query from a DAG
1.  **Identify the Final Goal**:
    *   Find the final node in `workflow.DAG`. In `8->2,8->3,2->4,3->4`, the final node is **4**.
    *   Look up the `description` of the tool corresponding to step **4** (`generate_clinical_summary`) in `tool_schemas` to understand its core function ("integrates... to generate a... clinical summary"). This forms the core request verb of the query.

2.  **Trace and Deconstruct the Data Flow**:
    *   This DAG (`8->2,8->3,2->4,3->4`) demonstrates a **"fan-out -> parallel processing -> fan-in"** pattern. You need to trace backward from the final node `4`.
    *   **Fan-In Point**: Node `4`'s input comes from node `2` (`get_lab_results`) and node `3` (`get_imaging_reports`). This indicates that the final summary must depend on **both** "lab results" and "imaging reports".
    *   **Fan-Out Point**: The inputs for both node `2` and `3` come from the same upstream node `8` (`find_patient_case`). This reveals a key piece of information: there is a **common starting point or subject** (a specific patient's case), and the subsequent parallel operations are all based on this starting point.

3.  **Extract & Embed Context and Constraints from Schemas**:
    *   **Analyze the Starting Point (node 8: `find_patient_case`)**:
        *   Look at its `inputSchema`. The `required` field is `patient_name`. This forms the **subject and core entity** of the query. The query must revolve around a specific person's name or entity.
    *   **Analyze the Parallel Branches (node 2 & 3: `get_lab_results`, `get_imaging_reports`)**:
        *   Read their `description` ("Retrieves specified laboratory test results," "Retrieves specified imaging reports") to understand the **specific intent** of each branch.
        *   Check their `optional` and `default` parameters. `get_lab_results` has a `test_types` parameter with a `default` value of `["Complete Blood Count", "Comprehensive Metabolic Panel"]`. `get_imaging_reports` has a `modality` parameter with no default. These are excellent materials for enriching the query details, making it more specific and realistic.
    *   **Combine Intents**: Fuse the requirements from the starting point and the parallel branches. The query should not be "Find patient. Get results. Get report." but rather "Regarding [Patient Name], I need their [test types] and [imaging modality], and then...".

4.  **Weave the Final Query**:
    *   Weave all the analyzed points (final goal, starting point, parallel tasks, specific parameters) into a single, natural, and coherent sentence.
    *   **Example Synthesis**:
        *   **Starting Point (node 8)**: "I want to process the case for patient 'John Smith'..."
        *   **Parallel Tasks (node 2 & 3)**: "...please get his latest 'Complete Blood Count' and 'Comprehensive Metabolic Panel' results, and at the same time, also retrieve his 'CT' imaging report..." (Note: this embeds the `default` value for `test_types` and a fabricated value 'CT' for `modality`)
        *   **Final Goal (node 4)**: "...finally, please integrate these two pieces of information and generate a professional clinical summary for me."
    *   **Final Query**: "For the patient 'John Smith', please retrieve his latest Complete Blood Count and Comprehensive Metabolic Panel results, as well as his CT imaging report, and then use all of that information to generate a professional clinical summary."


# Output Format Specification
Your final output must be a JSON object. **Crucially, you must include the `task_id` from the input `workflow` object in your output.**

```json
{
  "task_id": "This is the task_id from the input workflow object",
  "flow_id": "This is the flow_id from the workflow",
  "query": "This is the single, complex, natural language query string synthesized according to the principles above."
}
```

# Constraints
    *   Direct Output: Your response must be the final JSON object directly, without any explanatory text or code block markers.
    *   Deep Integration: The generated query must deeply integrate details from the tool_schemas, not just the tool names.
    *   Singularity: The result must be a single, coherent sentence or question, not a list of commands.
    

# User Input
mcp_dag
"""




def main(query_list, output_file, max_workers=5):
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

    with open(output_file, "a", encoding="utf-8") as f:
        print(f"Starting to process {len(request_payloads)} requests. Results will be written to {output_file}")
        
        response_generator = multi_thread_openai_generator(request_payloads, max_workers)

        progress_bar = tqdm(response_generator, total=len(request_payloads), desc="Processing requests")
        
        for original_payload, response_content in progress_bar:
            try:

                if isinstance(response_content, str) and response_content.startswith("ERROR:"):

                    result_data = {
                        "input_query": original_payload["messages"][-1]["content"],
                        "error": response_content
                    }
                    print(f"\nRequest failed: {result_data}") 
                else:

                    result_data = {
                        "input_query": original_payload["messages"][-1]["content"],
                        "response": response_content
                    }
                

                json_line = json.dumps(result_data, ensure_ascii=False)
                f.write(json_line + '\n')
                f.flush() 

            except Exception as e:

                print(f"\nAn error occurred while processing a result: {e}")
                error_data = {
                    "error": f"Failed to process response: {e}",
                    "original_payload": original_payload,
                    "raw_response": response_content
                }
                # f.write(json.dumps(error_data, ensure_ascii=False) + '\n')
                # f.flush()

    print(f"All tasks have been processed. Results saved to {output_file}")



def _load_all_tools(tools_path: str) -> dict[str, dict]:
    tool_map = {}
    try:
        with open(tools_path, "r", encoding="utf-8") as f:
            all_tools = json.load(f)
        for tool in all_tools:
            name = tool.get("name")
            if name:
                tool_map[name] = tool
    except (FileNotFoundError, json.JSONDecodeError) as e:
      print(f"Error: Could not load or parse tools file at '{tools_path}'. Error: {e}")
    return tool_map

def make_input_data(
    difficulty: str,
    base_data_dir: str,
    tools_path: str
) -> list[str]:
    """
    Generates prompts for the query reverse-engineering task for a specified workflow difficulty.

    Args:
        difficulty (str): The difficulty level to process ('easy', 'medium', 'hard').
        base_data_dir (str): The base directory containing all difficulty-level data.
        tools_path (str): The path to the global tool definitions file.

    Returns:
        list[str]: A list of fully-formed prompts ready for the LLM.
    """
    print(f"\n--- Preparing query reverse-engineering data for {difficulty.upper()} difficulty ---")
    
    # Dynamically construct the input file path based on the difficulty.
    dag_log_path = os.path.join(base_data_dir, difficulty, "tool2dag.log")

    # Pre-load all tool definitions for efficient lookup.
    tool_definitions = _load_all_tools(tools_path)
    if not tool_definitions:
        return []

    # Initialize lists and counters for processing.
    generated_prompts = []
    processed_lines, successful_parses, skipped_lines = 0, 0, 0

    # Process the DAG log file line by line.
    try:
        with open(dag_log_path, "r", encoding="utf-8") as f:
            for line in f:
                processed_lines += 1
                try:
                    # Parse the outer JSON structure of the log line.
                    line_data = json.loads(line)
                    if "response" not in line_data:
                        skipped_lines += 1
                        continue
                    
                    # Parse the inner JSON, which contains the actual workflow from the LLM.
                    workflow = json.loads(clean_json_string(line_data['response']))
                    if isinstance(workflow, list):
                        workflow = workflow[0] if workflow else {}
                    
                    # Validate that the workflow contains the necessary data (task_id, tool_sequence).
                    task_id = workflow.get('task_id')
                    tool_sequence = workflow.get('tool_sequence')
                    if not isinstance(workflow, dict) or not task_id or not tool_sequence:
                        skipped_lines += 1
                        continue
                    
                    # Look up the full schema for each tool used in the workflow.
                    current_tool_schemas = []
                    all_tools_found = True
                    for item in tool_sequence:
                        func_id = item.get('function_id')
                        if func_id in tool_definitions:
                            current_tool_schemas.append(tool_definitions[func_id])
                        else:
                            all_tools_found = False
                            break
                    if not all_tools_found:
                        skipped_lines += 1
                        continue
                    
                    # Assemble the final input for the prompt template.
                    prompt_input = {"workflow": workflow, "tool_schemas": current_tool_schemas}
                    prompt_str = prompt_template.replace("mcp_dag", json.dumps(prompt_input, ensure_ascii=False))
                    generated_prompts.append(prompt_str)
                    successful_parses += 1

                # Skip any line that has a parsing or data structure error.
                except (json.JSONDecodeError, TypeError, IndexError):
                    skipped_lines += 1
                    continue
    except FileNotFoundError:
        print(f"Error: Log file for '{difficulty}' difficulty not found at: {dag_log_path}")
        return []

    # Print a summary of the processing results.
    print("-" * 20)
    print("Data preparation summary:")
    print(f"  Total lines processed: {processed_lines}")
    print(f"  Successfully generated prompts: {successful_parses}")
    print(f"  Lines skipped due to format/data errors: {skipped_lines}")
    print("-" * 20)
    
    return generated_prompts

    

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Generate queries for workflows of a specified difficulty.")
  
  parser.add_argument(
      '--difficulty', 
      type=str, 
      default='easy', 
      choices=['easy', 'medium', 'hard'], 
      help="The difficulty level of the data to process (e.g., 'easy', 'medium', 'hard')"
  )
  parser.add_argument(
      '--base_dir',
      type=str,
      default='./result_data',
      help="The root directory containing data for all difficulty levels"
  )
  parser.add_argument(
      '--tools_file',
      type=str,
      default='./server_tool_output.json', 
      help="The JSON file containing all tool definitions"
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
      base_data_dir=args.base_dir,
      tools_path=args.tools_file
  )

    if not prompts_to_process:
        print(f"No prompts were generated for difficulty '{args.difficulty}'. Terminating.")
    else:
        output_query_file = os.path.join(args.base_dir, args.difficulty, "dag2query.log")
        
        os.makedirs(os.path.dirname(output_query_file), exist_ok=True)
        main(
            query_list=prompts_to_process,
            output_file=output_query_file,
            max_workers=args.workers
        )