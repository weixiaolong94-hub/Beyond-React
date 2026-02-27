#!/usr/bin/env python
# coding=utf-8

import os
import json
import logging
import argparse
from typing import List
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def get_args():
    parser = argparse.ArgumentParser(description="Qwen Planner vLLM")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--query_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    

    parser.add_argument("--gpu_devices", type=str, default="0,1,2,3,4,5,6,7")
    parser.add_argument("--tp_size", type=int, default=8)
    
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--n", type=int, default=3)
    parser.add_argument("--max_tokens", type=int, default=4000)
    
    parser.add_argument("--cpu_offload_gb", type=int, default=15)
    parser.add_argument("--max_num_seqs", type=int, default=100)
    
    return parser.parse_args()

args = get_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_devices

from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

class QwenInferenceEngine:
    def __init__(self, args):
        self.args = args
        self.tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
        
        self.llm = LLM(
            model=args.model_path,
            tensor_parallel_size=args.tp_size,
            enforce_eager=True,
            cpu_offload_gb=args.cpu_offload_gb,
            max_num_seqs=args.max_num_seqs,
            trust_remote_code=True
        )

    def prepare_prompts(self) -> List[str]:
        if not os.path.exists(self.args.query_path):
            raise FileNotFoundError(f"输入文件不存在: {self.args.query_path}")

        with open(self.args.query_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        prompts = []
        for item in data:
            instruction = item.get('instruction', '')
            query = item.get('input', '')
            
            messages = [{"role": "user", "content": instruction + query}]
            
            formatted_text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False
            )
            prompts.append(formatted_text)
            
        return prompts

    def execute_and_save(self, prompts: List[str]):
        sampling_params = SamplingParams(
            temperature=self.args.temperature,
            top_p=self.args.top_p,
            n=self.args.n,
            max_tokens=self.args.max_tokens
        )

        outputs = self.llm.generate(prompts, sampling_params)

        output_path = Path(self.args.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(self.args.output_path, "w", encoding='utf-8') as f:
            for output in outputs:
                prompt_text = output.prompt
                for gen_output in output.outputs:
                    record = {
                        "query": prompt_text,
                        "response": gen_output.text
                    }
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
        

def main():
    try:
        engine = QwenInferenceEngine(args)
        prompts = engine.prepare_prompts()
        engine.execute_and_save(prompts)
    except Exception as e:
        logger.error(f"推理过程中出现错误: {e}", exc_info=True)

if __name__ == "__main__":
    main()