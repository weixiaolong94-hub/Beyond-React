import time
import json
import re
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import *
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI, APIStatusError, APIConnectionError
from tqdm import tqdm


client = OpenAI(
    api_key="", 
    base_url="" 
)

def clean_json_string(json_str):
    if json_str.startswith("```json"):
        json_str = json_str[7:] 
    if json_str.endswith("```"):
        json_str = json_str[:-3] 
    return json_str.strip()

    
def load_json_file(file_path):
    """
    Helper function to load a JSON Lines file.

    Args:
        file_path (str): The path to the JSON Lines file.

    Returns:
        list: A list of dictionaries, where each dictionary is a parsed JSON line.
    """
    data = []
    with open(file_path, "r") as f:
        for line in f:
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError:
                # Skip lines that are not valid JSON.
                continue
    return data

def read_jsonl(file_path):
    """
    Reads a JSON Lines file and returns its content as a list of dictionaries.

    Args:
        file_path (str): The path to the JSON Lines file.

    Returns:
        list: A list of dictionaries parsed from the file.
    """
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            # Ensure the line is not empty before parsing.
            if line.strip(): 
                data.append(json.loads(line))
    return data

def write_jsonl(file_path, data, append=False):
    """
    Writes data to a JSON Lines file.

    Args:
        file_path (str): The path to the output file.
        data (list or dict): A list of dictionaries or a single dictionary to write.
        append (bool): If True, appends to the file. If False, overwrites the file.
    """
    mode = 'a' if append else 'w'


    if isinstance(data, dict):
        data = [data]

    with open(file_path, mode, encoding='utf-8') as f:
        for item in data:

            f.write(json.dumps(item, ensure_ascii=False))
            f.write('\n')


def process_model_output_data(response):
    """
    Extracts a JSON object from a model's raw string output.

    It first tries to find a JSON object within a ```json ... ``` markdown block.
    If not found, it attempts to parse the entire response string as JSON.

    Args:
        response (str): The raw string response from the language model.

    Returns:
        dict or None: The parsed JSON object as a dictionary, or None if parsing fails.
    """
    try:
        # Check for a markdown JSON block.
        if "```json" in response:
            pattern = r'```json\n(.*?)```'
            matches = re.findall(pattern, response, re.DOTALL)
            if len(matches) < 1:
                return None
            response = matches[0]
        # Attempt to parse the extracted or original string as JSON.
        response = json.loads(response)
    except json.JSONDecodeError:
        return None
    return response

def multi_thread_openai_generator(request_payloads, max_workers=5):
    """
    Sends multiple requests to the OpenAI API concurrently using a thread pool and yields results as they complete.

    Args:
        request_payloads (list): A list of dictionaries, where each dictionary is a payload
                                 for the `client.chat.completions.create` method.
        max_workers (int): The maximum number of threads to use for concurrent requests.

    Yields:
        tuple: A tuple containing the original request payload and the response content (or an error message).
    """

    def fetch_single_response(payload):
        """
        Sends a single request to the API with a retry mechanism.
        """
        # Retry up to 3 times in case of specific API errors.
        for attempt in range(3): 
            try:
                response = client.chat.completions.create(**payload)
                return response.choices[0].message.content
            except (APIStatusError, APIConnectionError) as e:
                print(f"Request failed (Attempt {attempt + 1}/3): {e}. Retrying in 1 second...")
                if attempt == 2: # Last attempt failed
                    return f"ERROR: API request failed after 3 attempts - {str(e)}"
                time.sleep(1)
            except Exception as e:
                print(f"An unknown error occurred (Attempt {attempt + 1}/3): {e}. Retrying in 1 second...")
                if attempt == 2: # Last attempt failed
                    return f"ERROR: Unknown error after 3 attempts - {str(e)}"
                time.sleep(1)

    # Use a ThreadPoolExecutor for concurrent execution.
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Map each future to its original payload to track requests.
        future_to_payload = {
            executor.submit(fetch_single_response, payload): payload 
            for payload in request_payloads
        }
        
        # Yield results as they are completed.
        for future in as_completed(future_to_payload):
            original_payload = future_to_payload[future]
            try:
                response_content = future.result()
                yield (original_payload, response_content)
            except Exception as e:
                # Handle exceptions that might occur during future execution.
                yield (original_payload, f"ERROR: Exception in future execution - {str(e)}")

if __name__ == "__main__":
    reqs = [
        {"model": "deepseek-v3", "messages": [{"role": "user", "content": "Hello"}]},
        {"model": "deepseek-v3", "messages": [{"role": "user", "content": "Who are you?"}]}
    ]
    
    results = []
    generator = multi_thread_openai_generator(reqs, max_workers=2)
    
    for payload, response in tqdm(generator, total=len(reqs), desc="Processing requests"):
        results.append({
            "request": payload,
            "response": response
        })

    print(json.dumps(results, indent=2, ensure_ascii=False))