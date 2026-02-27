#!/usr/bin/env python
# coding=utf-8
import time
from termcolor import colored
from typing import Optional, List
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoConfig,
)
from toolbench.inference.utils import SimpleChatIO, generate_stream
import json
import re
from accelerate import init_empty_weights, load_checkpoint_and_dispatch


def clean_json_string(json_str):
    if json_str.startswith("```json"):
        json_str = json_str[7:] 
    if json_str.endswith("```"):
        json_str = json_str[:-3] 
    return json_str.strip()


class Qwen_Planner:
    def __init__(
            self,
            model_name_or_path: str,
            device: str="cuda:7",
            max_sequence_length: int=8192
        ) -> None:
        super().__init__()
        self.model_name = model_name_or_path
        self.max_sequence_length = max_sequence_length

        if "cuda" in device and not torch.cuda.is_available():
            print(f"Warning: CUDA device '{device}' is not available. Falling back to CPU.")
            self.device = "cpu"
        else:
            self.device = device
        
        print(f"--- [Qwen_Planner Init Start] ---")
        print(f"[Qwen_Planner] Model Path: {model_name_or_path}")
        print(f"[Qwen_Planner] Target Device: {device}")

        print(f"[Qwen_Planner] 1. Loading tokenizer from: {model_name_or_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            use_fast=False,
            model_max_length=self.max_sequence_length,
            trust_remote_code=True
        )
        
        print(f"[Qwen_Planner] 2. Loading model config from: {model_name_or_path}")
        config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path, 
            low_cpu_mem_usage=True,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True
        )

        print(f"[Qwen_Planner] Model loaded on CPU. Moving to device: {self.device}...")
        self.model.to(self.device)
        self.model.eval() 

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.use_gpu = self.device.startswith("cuda")
        self.chatio = SimpleChatIO()
        self.conversation_history = []
        print(f"--- [Qwen_Planner Init End] ---")

    def prediction(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        with torch.no_grad():
            gen_params = {
                "model": self.model_name,
                "prompt": prompt,
                "temperature": 0.8,
                "max_new_tokens": 1024,
                "stop": "</s>",
                "stop_token_ids": None,
                "echo": False
            }
            main_device = self.model.device
            generate_stream_func = generate_stream
            output_stream = generate_stream_func(self.model, self.tokenizer, gen_params, main_device, self.max_sequence_length, force_generate=True)
            outputs = self.chatio.return_output(output_stream)
            prediction = outputs.strip()
        return prediction
        
    def add_message(self, message: dict):
        self.conversation_history.append(message)

    def change_messages(self, messages: List[dict]):
        self.conversation_history = messages

    def display_conversation(self, detailed: bool=False):
        role_to_color = {
            "system": "red", "user": "green", "assistant": "blue", "function": "magenta", "tool": "magenta",
        }
        print("before_print" + "*" * 50)
        for message in self.conversation_history:
            print_obj = f"{message['role']}: {message['content']} "
            print(colored(print_obj, role_to_color.get(message["role"], "white")))
        print("end_print" + "*" * 50)

    def parse(self, tools: List[dict], process_id: int, **kwargs) -> (dict, int, int):
        prompt_tool_list_str = ""
        user_query = ""

        for msg in reversed(self.conversation_history):
            if msg['role'] == 'user':
                user_query = msg['content']
                break
        
        if not user_query:
            raise ValueError("Could not find a 'user' role message in the conversation history.")
        
        all_dict = {}
        for i, tool in enumerate(tools,start=1):

            tool_function = tool['function']
            all_dict[str(i)] = tool_function
            

        instruction = "#### **Role and Core Task**\nYou are a **top-tier System Process Architect** and **Intelligent Task Planner**, specializing in parsing complex human language instructions and decomposing them into efficient, automated execution plans (DAGs). Your core task is to:\n- Based on the user's **complex Query** and the available **Tool List**, design a logically sound, efficient, and comprehensive tool collaboration workflow (DAG) that fully satisfies the user's requirements.\n- The final output must be a **precise DAG execution plan**, ensuring all steps have correct dependencies and parallelism is maximized where possible.\n\n---\n\n#### **Execution Steps**\n1. **Query Decomposition**: Analyze the user query, break it down into actionable sub-tasks, and identify key entities (such as time, location, subject, etc.).\n2. **Task-to-Tool Mapping**: For each sub-task, match it with the most appropriate tool from the tool library (select by tool ID).\n3. **Dependency Analysis**: Determine which steps need to wait for the results of other steps (e.g., you must 'search for a hotel' before you can 'book a hotel').\n4. **DAG Construction & Formatting**: Synthesize the dependencies into a DAG string and strictly adhere to the JSON format for the output.\n\n---\n\n#### **Output Format**\n2. **JSON Format**:\n```json\n{\n    \"DAG\": \"A dependency string (e.g., '1->2, 2->3, 2->4, 3->5, 4->5, 5->6')\"\n}\n```\nThe user input is:\n"

        query = user_query
        d_tmp = {"query":query,"tool_sequence":all_dict}

        prompt = instruction + json.dumps(d_tmp,ensure_ascii=False)
        messages = [
        {"role": "user", "content": prompt}
        ]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False
        )
        predictions = self.prediction(text)

        max_retries = 10 
        attempt = 0

        while attempt <= max_retries:
            try:
                dag_string = json.loads(clean_json_string(predictions))["DAG"]
                
                edges = [edge.strip() for edge in dag_string.split(',') if edge.strip()]

                if not edges and dag_string.strip() != "":
                    raise ValueError("Invalid DAG format: contains elements but no '->' edges.")
                
                if len(edges) == 1:
                    if dag_string == "1" or dag_string == "1->" or dag_string == "1->1":
                        break

                for edge in edges:
                    parts = edge.split('->')
                    
                    if len(parts) != 2:
                        raise ValueError(f"Invalid edge format: '{edge}'. An edge must be 'source->target'.")
                    
                    if parts[0] == parts[1]:
                        raise ValueError("Error: The provided DAG contains a cycle'.")

                    for part in parts:
                        node_id = part.strip()
                        
                        if len(edges) > 1:
                            if not node_id or node_id not in all_dict.keys():

                                raise ValueError(f"Invalid node ID '{node_id}' found in DAG.")
                        else:
                            if node_id and node_id not in all_dict.keys():
                                raise ValueError(f"Invalid node ID '{node_id}' found in DAG.")

                break

            except Exception as e:
                attempt += 1
                print(f"[Parser] Attempt {attempt}/{max_retries} failed. Reason: {e}")
                print(f"[Parser] Faulty prediction was: {predictions}")
                
                if attempt < max_retries:
                    print("[Parser] Retrying...")
                    if attempt > 5:
                        prompt1 = instruction + "Attention: Please select tools strictly according to the serial number in the tool list and do not exceed the tool index" + json.dumps(d_tmp,ensure_ascii=False) 
                        messages1 = [{"role": "user", "content": prompt1}]
                        text1 = self.tokenizer.apply_chat_template(
                            messages1,
                            tokenize=False,
                            add_generation_prompt=True,
                            enable_thinking=False
                        )
                        predictions = self.prediction(text1)
                    else:
                        predictions = self.prediction(text)
                else:
                    predictions = "```json\n{\n    \"DAG\": \"'1->\"\n}\n```"
                    print("[Parser] Max retries reached. Failed to get a valid DAG.")

        decoded_token_len = len(self.tokenizer(predictions, add_special_tokens=False).input_ids)

        if process_id == 0:
            print(f"[Qwen_Planner (process({process_id}))] Raw output:\n{predictions}")
            print(f"[Qwen_Planner (process({process_id}))] Total generated tokens: {decoded_token_len}")

        message = {
            "role": "assistant",
            "content": predictions,
        }
        
        return message, 0, decoded_token_len