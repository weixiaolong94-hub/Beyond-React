import argparse
import json
import os
import logging


from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_prompts(query_path: str, tokenizer: AutoTokenizer) -> list[str]:
    """
    Load data from a JSON file and format it into prompts.
    """
    prompts = []
    logging.info(f"Loading prompts from {query_path}...")
    try:
        with open(query_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        logging.error(f"Error: Input file not found at {query_path}")
        return []
    except json.JSONDecodeError:
        logging.error(f"Error: Failed to parse JSON file {query_path}")
        return []

    for item in data:
        instruction = item.get('instruction', '')
        query = item.get('input', '')
        # Ensure at least one of 'instruction' or 'query' exists
        if not instruction and not query:
            logging.warning(f"Skipping an empty item: {item}")
            continue
            
        messages = [
            {"role": "user", "content": instruction + query}
        ]
        try:
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            prompts.append(text)
        except Exception as e:
            logging.error(f"Error applying chat template: {e} - Skipping this item: {item}")

    logging.info(f"Successfully loaded and formatted {len(prompts)} prompts.")
    return prompts

def save_results(outputs, output_path: str):
    """
    Save the vLLM output to a file in JSONL format.
    """
    logging.info(f"Saving results to {output_path}...")
    try:
        with open(output_path, "w", encoding='utf-8') as f:
            for output in outputs:
                prompt_text = output.prompt
                # Use a clearer variable name to avoid shadowing the outer 'output'
                for single_generation in output.outputs:
                    generated_text = single_generation.text
                    result_item = {"query": prompt_text, "response": generated_text}
                    f.write(json.dumps(result_item, ensure_ascii=False) + "\n")
        logging.info("Results saved successfully.")
    except IOError as e:
        logging.error(f"Error writing to file {output_path}: {e}")

def main(args):
    """
    Main execution function.
    """
    # 1. Load Tokenizer
    logging.info(f"Loading tokenizer from {args.model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    
    # 2. Load and prepare prompts
    prompts = load_prompts(args.query_path, tokenizer)
    if not prompts:
        logging.error("No prompts available, exiting.")
        return

    # 3. Configure sampling parameters
    sampling_params = SamplingParams(
        n=args.n,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        stop_token_ids=[tokenizer.eos_token_id] # It's recommended to add stop tokens
    )

    # 4. Initialize vLLM
    logging.info(f"Initializing vLLM with tensor_parallel_size={args.tensor_parallel_size}...")
    llm = LLM(
        model=args.model_path,
        tensor_parallel_size=args.tensor_parallel_size,
        trust_remote_code=True
    )

    # 5. Generate text
    logging.info("Starting text generation...")
    outputs = llm.generate(prompts, sampling_params)
    logging.info("Text generation completed.")

    # 6. Save results
    save_results(outputs, args.output_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run inference using a model with vLLM")
    
    # File path arguments
    parser.add_argument("--model_path", type=str, required=True, help="Path to the directory containing the model.")
    parser.add_argument("--query_path", type=str, required=True, help="Path to the JSON file containing inference prompts.")
    parser.add_argument("--output_path", type=str, required=True, help="Path to the output JSONL file for saving results.")

    # vLLM and model parameters
    parser.add_argument("--tensor_parallel_size", type=int, default=8, help="The number of GPUs to use for tensor parallelism.")
    
    # Sampling parameters
    parser.add_argument("--n", type=int, default=3, help="Number of output sequences to generate for each prompt.")
    parser.add_argument("--temperature", type=float, default=0.8, help="The temperature for sampling.")
    parser.add_argument("--top_p", type=float, default=0.95, help="The top-p (nucleus) sampling value.")
    parser.add_argument("--max_tokens", type=int, default=4096, help="The maximum number of tokens to generate.")

    args = parser.parse_args()
    main(args)