import json
import re
from collections import defaultdict
from typing import List, Dict, Tuple, Set, Optional



def load_json_file(file_path: str) -> list:
    """
    Loads a JSONL file (one JSON object per line) into a list of dictionaries.

    Args:
        file_path (str): The path to the JSONL file.

    Returns:
        list: A list of parsed JSON objects.
    """
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            # Ensure the line is not empty before parsing
            if line.strip():
                data.append(json.loads(line))
    return data

def write_jsonl(data: list, filename: str):
    """
    Writes a list of dictionaries to a JSONL file.

    Args:
        data (list): The list of dictionary objects to write.
        filename (str): The path to the output file.
    """
    with open(filename, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

# --- Parsing and Extraction ---

def extract_json_block(text: str) -> Optional[dict]:
    """
    Extracts a JSON object from a string, typically one enclosed in markdown-style ```json ... ``` blocks.

    Args:
        text (str): The input string containing the JSON block.

    Returns:
        Optional[dict]: The parsed JSON object as a dictionary, or None if not found or invalid.
    """
    # Regex to find content inside ```json ... ```
    pattern = re.compile(r"```json(.*?)```", re.DOTALL)
    match = pattern.search(text)
    if match:
        json_string = match.group(1).strip()
        try:
            return json.loads(json_string)
        except json.JSONDecodeError:
            print("Error: Extracted content is not valid JSON.")
            return None
    else:
        print("Error: JSON block not found.")
        return None

def parse_edges_from_dag_str(dag_str: str) -> list[tuple[int, int]]:
    """
    Parses a DAG string (e.g., '1->2, 2->3') into a list of integer-based edges.
    This function primarily serves as a strict syntax validator.

    Args:
        dag_str (str): The DAG string representation.

    Returns:
        list[tuple[int, int]]: A list of (source, target) tuples.
    
    Raises:
        ValueError: If the syntax is invalid (e.g., missing '->', empty node).
    """
    edges = []
    for pair in dag_str.split(','):
        pair = pair.strip()
        if not pair:
            continue
        if '->' not in pair:
            raise ValueError(f"Missing '->' symbol in edge -> {pair}")
        src, tgt = pair.split('->')
        src = src.strip()
        tgt = tgt.strip()
        if not src or not tgt:
            raise ValueError(f"Empty node found in edge -> {pair}")
        # Convert to int to ensure nodes are numeric
        edges.append((int(src), int(tgt)))
    return edges

# --- Graph Analysis ---

def build_graph_from_dag_str(dag_str: str) -> Optional[Tuple[Dict[str, List[str]], Set[str]]]:
    """
    Builds an adjacency list representation of a graph from a DAG string.

    Args:
        dag_str (str): The DAG string representation.

    Returns:
        Optional[Tuple[Dict[str, List[str]], Set[str]]]: 
            A tuple containing the graph (adjacency list) and a set of all nodes.
            Returns None if there's a syntax error that prevents graph construction.
    """
    if not dag_str or not dag_str.strip():
        # Handle empty or whitespace-only DAG string as a valid empty graph
        return defaultdict(list), set()
        
    graph = defaultdict(list)
    all_nodes = set()
    try:
        for part in dag_str.split(','):
            part = part.strip()
            if '->' not in part:
                return None  # Syntax error: missing '->'
            u, v = map(str.strip, part.split('->'))
            # Check if nodes can be cast to int, raising ValueError if not
            int(u)
            int(v)
            graph[u].append(v)
            all_nodes.update([u, v])
    except ValueError:
        return None # Syntax error: non-integer node
    return graph, all_nodes

def compute_graph_penalty(dag_str: str, check_connectivity: bool = True) -> float:
    """
    Analyzes a DAG string for structural validity (cycles, connectivity) and returns a penalty score.

    Args:
        dag_str (str): The DAG string to analyze.
        check_connectivity (bool): If True, checks if the graph is a single connected component.

    Returns:
        float: 
            - 0.0 for a valid DAG.
            - -10.0 if a cycle is detected.
            - -2.0 if the graph is not connected (and check_connectivity is True).
            - -99.0 for a major syntax error during graph building.
    """
    graph_data = build_graph_from_dag_str(dag_str)
    if graph_data is None:
        return -99.0  # Syntax error

    graph, all_nodes = graph_data
    if not all_nodes:
        return 0.0  # An empty graph is valid

    # --- Cycle Detection using DFS ---
    visited = set()
    rec_stack = set()  # Recursion stack for the current DFS path

    def has_cycle(v: str) -> bool:
        visited.add(v)
        rec_stack.add(v)
        for neighbor in graph.get(v, []):
            if neighbor not in visited:
                if has_cycle(neighbor):
                    return True
            elif neighbor in rec_stack:  # A back edge is found
                return True
        rec_stack.remove(v)
        return False

    # Check for cycles in all disconnected components of the graph
    for node in all_nodes:
        if node not in visited:
            if has_cycle(node):
                return -10.0  # Cycle detected

    # --- Connectivity Check (optional) ---
    if check_connectivity:
        # Build an undirected version of the graph for connectivity check
        undirected = defaultdict(list)
        for u in graph:
            for v in graph[u]:
                undirected[u].append(v)
                undirected[v].append(u)
        
        start_node = next(iter(all_nodes))
        queue = [start_node]
        visited_conn = {start_node}
        
        # Perform BFS/DFS to find all reachable nodes
        while queue:
            current = queue.pop(0)
            for neighbor in undirected.get(current, []): # Use .get for safety
                if neighbor not in visited_conn:
                    visited_conn.add(neighbor)
                    queue.append(neighbor)
        
        # If not all nodes were visited, the graph is not connected
        if len(visited_conn) != len(all_nodes):
            return -2.0  # Not a single connected component

    return 0.0  # Graph is a valid DAG

# --- Main Logic ---

def check_all_dags(filepath: str) -> Tuple[Dict[str, list], Dict[str, int]]:
    """
    Processes a JSONL file of LLM responses, validates the 'DAG' field in each,
    and categorizes them into valid and invalid groups.

    Args:
        filepath (str): The path to the input JSONL file.

    Returns:
        Tuple[Dict[str, list], Dict[str, int]]:
            - A dictionary 나누어진 into 'valid' and 'invalid' lists of items.
            - A dictionary counting the occurrences of different error types.
    """
    data = load_json_file(filepath)
    results = {"valid": [], "invalid": []}
    error_counts = {"Cycle Detected": 0, "Graph Not Connected": 0, "Syntax Error": 0}

    for item in data:
        output_str = item.get("response")
        if not output_str:
            item["errors"] = ["Syntax Error: Missing 'response' field"]
            results["invalid"].append(item)
            error_counts["Syntax Error"] += 1
            continue

        output_json = extract_json_block(output_str)
        if not output_json:
            item["errors"] = ["Syntax Error: Cannot parse JSON from response"]
            results["invalid"].append(item)
            error_counts["Syntax Error"] += 1
            continue

        dag_str = output_json.get("DAG")
        if not dag_str:
            item["errors"] = ["Syntax Error: Missing 'DAG' field in JSON"]
            results["invalid"].append(item)
            error_counts["Syntax Error"] += 1
            continue

        # Preliminary syntax check for better error messages
        try:
            parse_edges_from_dag_str(dag_str)
        except ValueError as e:
            item["errors"] = [f"Syntax Error: {str(e)}"]
            results["invalid"].append(item)
            error_counts["Syntax Error"] += 1
            continue

        # Deeper structural check
        score = compute_graph_penalty(dag_str)
        if score == 0.0:
            results["valid"].append(item)
        else:
            reasons = []
            if score == -10.0:
                reasons.append("Cycle Detected")
                error_counts["Cycle Detected"] += 1
            elif score == -2.0:
                reasons.append("Graph Not Connected")
                error_counts["Graph Not Connected"] += 1
            elif score == -99.0:
                reasons.append("Syntax Error: Failed to build graph")
                error_counts["Syntax Error"] += 1
            else:
                reasons.append("Unknown Structure Error") # Fallback for other potential negative scores
            item["errors"] = reasons
            results["invalid"].append(item)

    return results, error_counts


if __name__ == "__main__":
    filepath = "path/to/your/tool2dag.log" 
    
    import os
    if not os.path.exists(filepath):
        print(f"FATAL ERROR: Input file not found at '{filepath}'. Please set the correct path.")
    else:
        results, error_counts = check_all_dags(filepath)

        print(f"Total Valid DAGs: {len(results['valid'])}")
        print(f"Total Invalid DAGs: {len(results['invalid'])}")
        print("\nError Type Breakdown:")
        for reason, count in error_counts.items():
            print(f" - {reason}: {count}")

        # Write the results to output files
        write_jsonl(results["valid"], "valid_dags.jsonl")
        write_jsonl(results["invalid"], "invalid_dags.jsonl")
        print("\nResults have been written to 'valid_dags.jsonl' and 'invalid_dags.jsonl'.")