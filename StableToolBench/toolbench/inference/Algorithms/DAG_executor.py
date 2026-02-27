import re
import json
from copy import deepcopy
from collections import defaultdict, deque
from Tree.Tree import my_tree, tree_node
from Algorithms.base_search import base_search_method
from Prompts.ReAct_prompts_parallel import FORMAT_INSTRUCTIONS_SYSTEM_FUNCTION, FORMAT_INSTRUCTIONS_USER_FUNCTION, FORMAT_INSTRUCTIONS_SYSTEM_FUNCTION_GPT
from Prompts.Plan_Excutor_prompt import EXECUTOR_SYSTEM_PROMPT_GPT


class DAG_executor(base_search_method):
    def __init__(self, planning_llm, execution_llm, io_func, process_id=0, callbacks=None):
        super(DAG_executor, self).__init__(planning_llm, io_func, process_id, callbacks)
        
        self.planning_llm = planning_llm
        self.execution_llm = execution_llm
        self.io_func = io_func
        self.process_id = process_id
        self.callbacks = callbacks if callbacks is not None else []
        
        self.restart()

    def restart(self):
        self.status = 0
        self.terminal_node = None
        self.query_count = 0
        self.total_tokens = 0
        self.dag_string = ""
        self.execution_chain_json = {}
        self.forward_args = {}

    def to_json(self, answer=False, process=True):
        if process:
            json_obj = {
                "win": self.status == 1,
                "dag_string": self.dag_string,
                "execution_chain": self.execution_chain_json,
                "forward_args": self.forward_args,
            }
        else:
            json_obj = {}

        if answer and self.terminal_node:
            is_valid = self.status == 1
            final_answer_text = ""
            train_messages = []

            if is_valid:
                if self.terminal_node.description:
                     try:
                        final_answer_data = json.loads(self.terminal_node.description)
                        final_answer_text = final_answer_data.get('final_answer', '')
                     except json.JSONDecodeError:
                        final_answer_text = self.terminal_node.description

                train_messages = self.terminal_node.get_train_messages_from_this_node()

            json_obj["answer_generation"] = {
                "valid_data": is_valid,
                "final_answer": final_answer_text,
                "function": self.io_func.functions,
                "query_count": self.query_count,
                "total_tokens": self.total_tokens,
                "train_messages": train_messages,
            }
        return json_obj
    
    def _parse_and_sort_dag_by_level(self, dag_string, tool_names):
        adj = defaultdict(list)
        in_degree = defaultdict(int)
        all_nodes_in_dag = set()
        
        try:
            edges = [edge.strip() for edge in dag_string.split(',') if edge.strip()]
            
            tool_id_map = tool_names 
            
            if len(edges) == 1 and '->' not in edges[0]:
                node_idx_str = edges[0].strip()
                if node_idx_str in tool_id_map:
                    tool_name = tool_id_map[node_idx_str]['name']
                    return [[tool_name]]
                else:
                    pass # Let it fail in the main loop for a consistent error message

            if not edges and dag_string.strip() == "":
                return []

            for edge in edges:
                parts = edge.split('->')
                if len(parts) != 2 or not parts[0] or not parts[1]:
                    print(f"[DAG Parser] Invalid edge format: '{edge}'")
                    return None
                
                u_idx_str, v_idx_str = parts[0].strip(), parts[1].strip()
                if u_idx_str not in tool_names or v_idx_str not in tool_names:
                    print(f"[DAG Parser] Node index not found: {u_idx_str} or {v_idx_str}")
                    return None
                    
                u_name, v_name = tool_names[u_idx_str]['name'], tool_names[v_idx_str]['name']
                adj[u_name].append(v_name)
                in_degree[v_name] += 1
                all_nodes_in_dag.add(u_name)
                all_nodes_in_dag.add(v_name)
            
            for edge in edges:
                parts = edge.split('->')
                for part in parts:
                    node_idx_str = part.strip()
                    if node_idx_str in tool_names:
                         all_nodes_in_dag.add(tool_names[node_idx_str]['name'])

            queue = deque([node for node in all_nodes_in_dag if in_degree[node] == 0])
            sorted_levels = []
            count = 0
            
            while queue:
                level_size = len(queue)
                current_level = []
                for _ in range(level_size):
                    u = queue.popleft()
                    current_level.append(u)
                    count += 1
                    for v in adj[u]:
                        in_degree[v] -= 1
                        if in_degree[v] == 0:
                            queue.append(v)
                sorted_levels.append(current_level)
            
            if count != len(all_nodes_in_dag):
                print("[DAG Parser] Error: The provided DAG contains a cycle.")
                return None
                
            return sorted_levels

        except Exception as e:
            print(f"[DAG Parser] An unexpected error occurred: {e}")
            return None

    def _get_dag_plan(self):
        all_dict = {}
        for i, func in enumerate(self.io_func.functions, start=1):
            tool_function = func['function']
            all_dict[str(i)] = tool_function
            
        instruction = "#### **Role and Core Task**\nYou are a **top-tier System Process Architect** and **Intelligent Task Planner**, specializing in parsing complex human language instructions and decomposing them into efficient, automated execution plans (DAGs). Your core task is to:\n- Based on the user's **complex Query** and the available **Tool List**, design a logically sound, efficient, and comprehensive tool collaboration workflow (DAG) that fully satisfies the user's requirements.\n- The final output must be a **precise DAG execution plan**, ensuring all steps have correct dependencies and parallelism is maximized where possible.\n\n---\n\n#### **Execution Steps**\n1. **Query Decomposition**: Analyze the user query, break it down into actionable sub-tasks, and identify key entities (such as time, location, subject, etc.).\n2. **Task-to-Tool Mapping**: For each sub-task, match it with the most appropriate tool from the tool library (select by tool ID).\n3. **Dependency Analysis**: Determine which steps need to wait for the results of other steps (e.g., you must 'search for a hotel' before you can 'book a hotel').\n4. **DAG Construction & Formatting**: Synthesize the dependencies into a DAG string and strictly adhere to the JSON format for the output.\n\n---\n\n#### **Output Format**\n2. **JSON Format**:\n```json\n{\n    \"DAG\": \"A dependency string (e.g., '1->2, 2->3, 2->4, 3->5, 4->5, 5->6')\"\n}\n```\nThe user input is:\n"
        
        query = self.io_func.input_description
        d_tmp = {"query": query, "tool_sequence": all_dict}
        prompt = instruction + json.dumps(d_tmp, ensure_ascii=False)
        
        if self.process_id == 0:
            print("[DAG_executor] Sending request to planning model (Qwen)...")
            
        self.planning_llm.change_messages([{"role": "user", "content": prompt}]) 
        response_message, error_code, total_tokens = self.planning_llm.parse(
            tools=self.io_func.functions,
            process_id=self.process_id
        )
        
        self.query_count += 1
        self.total_tokens += total_tokens

        if error_code != 0 or not response_message.get('content'):
            print(f"[DAG_executor] Planning model failed with error code {error_code}.")
            self.dag_string = "ERROR: Planning model call failed."
            return None

        response_text = response_message['content']
        
        try:
            json_str_match = re.search(r'```json\n(.*?)\n```', response_text, re.DOTALL)
            json_str = json_str_match.group(1) if json_str_match else response_text
            
            dag_data = json.loads(json_str)
            self.dag_string = dag_data.get("DAG", "")
            
            if self.process_id == 0:
                print(f"[DAG_executor] Received DAG from planner: \"{self.dag_string}\"")

            sorted_tool_levels = self._parse_and_sort_dag_by_level(self.dag_string, all_dict)
            
            if sorted_tool_levels is None:
                print("[DAG_executor] DAG parsing or sorting failed.")
                return None
            return sorted_tool_levels
            
        except Exception as e:
            print(f"[DAG_executor] Failed to parse planning model's output: {e}")
            return None


    def _execute_plan(self, now_node, planned_tool_levels, max_steps):
        system = EXECUTOR_SYSTEM_PROMPT_GPT.replace("{task_description}", self.io_func.task_description)
        now_node.messages.append({"role": "system", "content": system})
        user = FORMAT_INSTRUCTIONS_USER_FUNCTION.replace("{input_description}", self.io_func.input_description)
        now_node.messages.append({"role": "user", "content": user})
        
        for level_idx, tool_level in enumerate(planned_tool_levels):
            if now_node.get_depth() >= max_steps:
                print("[DAG_executor] Execution reached max steps. Aborting.")
                now_node.pruned = True
                return now_node

            if self.process_id == 0:
                print(f"[DAG_executor] Level {level_idx + 1}/{len(planned_tool_levels)}: Preparing to execute: {tool_level}")

            context_summary_prompt = ""
            if level_idx > 0:
                tool_results_summary = []
                for msg in now_node.messages:
                    if msg["role"] == "tool":
                        tool_name = msg.get("name", "UnknownTool")
                        tool_content = str(msg.get("content", ""))
                        truncated_content = (tool_content[:5000] + '...') if len(tool_content) > 5000 else tool_content
                        tool_results_summary.append(f"Result from tool '{tool_name}':\n{truncated_content}")

                if tool_results_summary:
                    context_summary_prompt = (
                        "Based on the results from the previous steps, you must now proceed.\n"
                        "Here is a summary of the key information obtained so far:\n\n"
                        + "\n\n---\n\n".join(tool_results_summary)
                        + "\n\n---\n"
                    )

            if len(tool_level) == 1:
                instruction_prompt = context_summary_prompt + f"Now, you must call the tool `{tool_level[0]}`. Generate the arguments for it."
                tool_choice = {"type": "function", "function": {"name": tool_level[0]}}
            else:
                instruction_prompt = (
                    context_summary_prompt +
                    "Now, you must call all of the following tools in parallel in your next turn: "
                    f"{', '.join(f'`{tool}`' for tool in tool_level)}. Generate the arguments for each of them."
                )
                tool_choice = "auto"

            now_node.messages.append({"role": "user", "content": instruction_prompt})
            
            self.execution_llm.change_messages(now_node.messages)
            new_message, error_code, total_tokens = self.execution_llm.parse(
                tools=self.io_func.functions, 
                tool_choice=tool_choice,
                process_id=self.process_id
            )
            self.query_count += 1
            self.total_tokens += total_tokens
            now_node.messages.pop()


            if "content" in new_message and new_message.get("content"):
                thought_node = tree_node(); thought_node.node_type = "Thought"
                thought_node.description = new_message["content"]
                thought_node.io_state = deepcopy(now_node.io_state)
                thought_node.messages = deepcopy(now_node.messages)
                thought_node.father = now_node; now_node.children.append(thought_node)
                thought_node.print(self.process_id); now_node = thought_node
                if error_code != 0: now_node.pruned = True

            if "tool_calls" in new_message and new_message.get("tool_calls"):
                tool_calls = new_message["tool_calls"]
                now_node.messages.append(new_message)
                
                
                results = []
                def call_tool(t_call, current_state):
                    f_name = t_call["function"]["name"]
                    f_input = t_call["function"]["arguments"]
                    t_id = t_call['id']
                    

                    obs, stat = current_state.step(action_name=f_name, action_input=f_input)
                    return {
                        "name": f_name,
                        "input": f_input,
                        "observation": obs,
                        "status": stat,
                        "id": t_id
                    }

                if self.process_id == 0:
                    print(f"[DAG_executor] Launching {len(tool_calls)} tools in parallel...")

                with ThreadPoolExecutor(max_workers=len(tool_calls)) as executor:
                    future_to_tool = {
                        executor.submit(call_tool, tc, deepcopy(now_node.io_state)): tc 
                        for tc in tool_calls
                    }
                    
                    for future in as_completed(future_to_tool):
                        try:
                            result = future.result()
                            results.append(result)
                        except Exception as exc:
                            print(f"[DAG_executor] Tool generated an exception: {exc}")


                for res in results:
                    action_node = tree_node()
                    action_node.node_type = "Action"
                    action_node.description = res["name"]
                    action_node.io_state = deepcopy(now_node.io_state) 
                    action_node.messages = deepcopy(now_node.messages)
                    action_node.father = now_node
                    now_node.children.append(action_node)

                    input_node = tree_node()
                    input_node.node_type = "Action Input"
                    input_node.description = res["input"]
                    input_node.observation = res["observation"]
                    input_node.observation_code = res["status"]
                    input_node.io_state = deepcopy(action_node.io_state)
                    

                    now_node.io_state.update_condition(res["name"], res["observation"]) 
                    
                    input_node.messages = deepcopy(action_node.messages)
                    input_node.father = action_node
                    action_node.children.append(input_node)
                    

                    now_node = input_node
                    now_node.print(self.process_id)

                    now_node.messages.append({
                        "role": "tool", 
                        "name": res["name"],
                        "content": str(res["observation"]), 
                        "tool_call_id": res["id"],
                    })

                    if res["status"] != 0:
                        print(f"[DAG_executor] Parallel tool '{res['name']}' failed.")
                        now_node.pruned = True
                        return now_node
            else:
                print(f"[DAG_executor] Execution LLM failed to generate any tool calls for this level. Halting.")
                now_node.pruned = True
                return now_node

        if not now_node.pruned:
            if self.process_id == 0:
                print("[DAG_executor] Plan execution finished. Generating final answer...")
            
            self.execution_llm.change_messages(now_node.messages)
            tool_choice = {"type": "function", "function": {"name": "Finish"}}
            
            final_message, error_code, total_tokens = self.execution_llm.parse(
                tools=self.io_func.functions,
                tool_choice=tool_choice,
                process_id=self.process_id
            )
            self.query_count += 1
            self.total_tokens += total_tokens

            if "tool_calls" in final_message and final_message.get("tool_calls"):
                finish_call = final_message["tool_calls"][0]
                
                action_node = tree_node()
                action_node.node_type = "Action"
                action_node.description = finish_call["function"]["name"]
                action_node.io_state = deepcopy(now_node.io_state)
                action_node.messages = deepcopy(now_node.messages)
                action_node.father = now_node
                now_node.children.append(action_node)
                now_node = action_node
                
                input_node = tree_node()
                input_node.node_type = "Action Input"
                input_node.description = finish_call["function"]["arguments"]
                input_node.is_terminal = True
                input_node.io_state = deepcopy(now_node.io_state)
                input_node.io_state.success = 1
                input_node.messages = deepcopy(now_node.messages)
                input_node.father = now_node
                now_node.children.append(input_node)
                now_node = input_node

        return now_node

    def start(self, single_chain_max_step):
        self.forward_args = locals(); self.forward_args.pop("self", None)
        self.tree = my_tree(); self.tree.root.node_type = "Action Input"
        self.tree.root.io_state = deepcopy(self.io_func)
        
        planned_tool_levels = self._get_dag_plan()
        if planned_tool_levels is None:
            self.status = -1; return 0
        
        if self.process_id == 0:
            plan_str = ' -> '.join([f"({', '.join(level)})" for level in planned_tool_levels])
            print(f"[DAG_executor] Parallel execution plan: {plan_str if plan_str else 'None'}")

        final_node = self._execute_plan(self.tree.root, planned_tool_levels, single_chain_max_step)
        
        self.terminal_node = final_node
        if self.terminal_node and self.terminal_node.io_state:
            self.execution_chain_json = self.terminal_node.get_chain_result_from_this_node(use_messages=False)
            if self.terminal_node.io_state.check_success() == 1:
                self.status = 1; return 1
        
        self.status = -1; return 0