# -*- coding: utf-8 -*-

import re
import json
from collections import defaultdict
from typing import List, Dict, Set, Tuple, Optional


def extract_json_block(text: str) -> Optional[dict]:
    """
    Extracts a JSON object from a string.

    This function first tries to find a JSON block enclosed in ```json ... ```.
    If a block is found, it attempts to parse the content.
    If no block is found, or if parsing the block fails, it tries to parse the entire text as a JSON object.

    Args:
        text (str): The input string to parse.

    Returns:
        Optional[dict]: A dictionary if a valid JSON object is found, otherwise None.
    """
    if not isinstance(text, str):
        return None

    # Regex to find a JSON block enclosed in triple backticks with a 'json' language identifier.
    pattern = re.compile(r"```json(.*?)```", re.DOTALL)
    match = pattern.search(text)

    if match:
        json_string = match.group(1).strip()
        try:
            # Return None for empty JSON strings
            if not json_string:
                return None
            return json.loads(json_string)
        except json.JSONDecodeError:
            # If parsing the content of the ```json``` block fails, return None.
            return None

    # Fallback: If no ```json``` block is found, try to parse the whole string.
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return None


def parse_edges_from_dag_str(dag_str: str) -> Set[Tuple[str, str]]:
    """
    Parses a string representation of a Directed Acyclic Graph (DAG) to extract its edges.

    The expected format is a comma-separated string of edges, e.g., "1->2, 2->3".
    Each edge must contain '->' and consist of two valid integer node identifiers.

    Args:
        dag_str (str): The string representing the DAG edges.

    Returns:
        Set[Tuple[str, str]]: A set of tuples, where each tuple (u, v) represents
                                a directed edge from node u to node v.
    """
    if not isinstance(dag_str, str) or not dag_str.strip():
        return set()

    edges = set()
    edge_parts = dag_str.split(',')
    for part in edge_parts:
        part = part.strip()
        # Skip empty parts or parts that don't represent an edge.
        if "->" not in part:
            continue

        nodes = part.split('->')
        if len(nodes) == 2:
            u, v = nodes[0].strip(), nodes[1].strip()
            try:
                # Ensure both nodes are non-empty and can be converted to integers.
                if u and v:
                    int(u)
                    int(v)
                    edges.add((u, v))
            except ValueError:
                # Ignore edges with non-integer node identifiers.
                continue
    return edges


def build_graph_from_dag_str(dag_str: str) -> Tuple[Dict[str, List[str]], Set[str]]:
    """
    Builds a graph representation (adjacency list) from a DAG string.

    Args:
        dag_str (str): The string representing the DAG edges, e.g., "1->2, 2->3".

    Returns:
        Tuple[Dict[str, List[str]], Set[str]]:
            - A dictionary (adjacency list) mapping each node to a list of its children.
            - A set of all unique node identifiers in the graph.
    """
    if not isinstance(dag_str, str) or not dag_str.strip():
        return defaultdict(list), set()

    graph = defaultdict(list)
    all_nodes = set()

    edge_parts = dag_str.split(',')
    for part in edge_parts:
        part = part.strip()
        if "->" not in part:
            continue

        nodes = part.split('->')
        if len(nodes) != 2:
            continue

        u, v = nodes[0].strip(), nodes[1].strip()
        try:
            # Validate that nodes are non-empty and represent integers.
            if u and v:
                int(u)
                int(v)
                graph[u].append(v)
                all_nodes.add(u)
                all_nodes.add(v)
        except ValueError:
            # Ignore edges with non-integer node IDs.
            continue

    return graph, all_nodes


def compute_format_penalty(solution_str: str) -> float:
    """
    (Level 0) Checks if the model output contains a valid JSON format.

    The output is considered valid if it either contains a ```json...``` block
    or if the entire string can be parsed as JSON.

    Args:
        solution_str (str): The full string output from the model.

    Returns:
        float: 0.0 for correct format, -10.0 for a penalty.
    """
    if not isinstance(solution_str, str):
        return -10.0

    # Check for a markdown-style JSON block.
    json_pattern = re.compile(r"```json.*?```", re.DOTALL)
    has_json_block = json_pattern.search(solution_str) is not None

    if has_json_block:
        return 0.0
    else:
        # If no block, check if the entire string is valid JSON.
        try:
            json.loads(solution_str)
            return 0.0
        except json.JSONDecodeError:
            return -10.0


def compute_syntax_penalty(solution_str: str) -> float:
    """
    (Level 1) Computes the syntax penalty (P_syntax) for the model's output.

    This function checks if the extracted JSON contains a "DAG" key with a
    syntactically correct string value for the graph edges.

    Args:
        solution_str (str): The full string output from the model.

    Returns:
        float: 0.0 for correct syntax, -10.0 for a penalty.
    """
    parsed_json = extract_json_block(solution_str)

    if not isinstance(parsed_json, dict):
        return -10.0

    dag_str = parsed_json.get("DAG")
    if not isinstance(dag_str, str):
        return -10.0

    # If the DAG string is not empty, validate its edge format.
    if dag_str.strip():
        edge_parts = dag_str.split(',')
        for part in edge_parts:
            part = part.strip()
            if not part: continue

            if "->" not in part: return -10.0
            nodes = part.split('->')
            if len(nodes) != 2: return -10.0

            u, v = nodes[0].strip(), nodes[1].strip()
            if not u or not v: return -10.0
            try:
                # Node identifiers must be integers.
                int(u)
                int(v)
            except ValueError:
                return -10.0

    return 0.0


def compute_graph_penalty(dag_str: str, check_connectivity: bool = True) -> float:
    """
    (Level 2) Computes the penalty for graph structure validity (P_graph = P_cycle + P_connectivity).

    It penalizes graphs with cycles (-10.0) and, optionally, graphs that are not
    connected (-2.0).

    Args:
        dag_str (str): The string representing the DAG edges.
        check_connectivity (bool): If True, checks if the graph is connected.

    Returns:
        float: A penalty score. 0.0 if valid, negative otherwise.
    """
    graph, all_nodes = build_graph_from_dag_str(dag_str)

    # An empty graph is considered valid.
    if not all_nodes:
        return 0.0

    # --- Cycle Detection using DFS ---
    p_cycle = 0.0
    visited = set()
    recursion_stack = set()
    def has_cycle_util(node):
        visited.add(node)
        recursion_stack.add(node)
        for neighbor in graph.get(node, []):
            if neighbor not in visited:
                if has_cycle_util(neighbor):
                    return True
            # If a neighbor is in the recursion stack, a cycle is found.
            elif neighbor in recursion_stack:
                return True
        recursion_stack.remove(node)
        return False

    # Check for cycles starting from each node.
    for node in list(all_nodes):
        if node not in visited:
            if has_cycle_util(node):
                p_cycle = -10.0
                break
    if p_cycle != 0.0:
        return p_cycle

    # --- Connectivity Check using BFS ---
    p_connectivity = 0.0
    if check_connectivity and all_nodes:
        # Build an undirected version of the graph for connectivity check.
        undirected_graph = defaultdict(list)
        for u, neighbors in graph.items():
            for v in neighbors:
                undirected_graph[u].append(v)
                undirected_graph[v].append(u)

        # Start BFS from an arbitrary node.
        q = [next(iter(all_nodes))]
        visited_connectivity = {q[0]}
        head = 0
        while head < len(q):
            node = q[head]
            head += 1
            for neighbor in undirected_graph.get(node, []):
                 if neighbor not in visited_connectivity:
                    visited_connectivity.add(neighbor)
                    q.append(neighbor)

        # If not all nodes were visited, the graph is not connected.
        if len(visited_connectivity) != len(all_nodes):
            p_connectivity = -2.0

    return p_cycle + p_connectivity


def compute_edge_f1_score(pred_dag_str: str, gt_dag_str: str) -> float:
    """
    (Level 3) Computes the F1 score for the predicted edges against the ground truth.

    Args:
        pred_dag_str (str): The DAG string from the model's prediction.
        gt_dag_str (str): The ground truth DAG string.

    Returns:
        float: The F1 score, between 0.0 and 1.0.
    """
    e_pred = parse_edges_from_dag_str(pred_dag_str)
    e_gt = parse_edges_from_dag_str(gt_dag_str)

    # Handle edge cases: if ground truth is empty, score is 1.0 only if prediction is also empty.
    if not e_gt:
        return 1.0 if not e_pred else 0.0
    # If prediction is empty but ground truth is not, score is 0.0.
    if not e_pred:
        return 0.0

    true_positives = len(e_pred.intersection(e_gt))

    precision = true_positives / len(e_pred)
    recall = true_positives / len(e_gt)

    if precision + recall == 0:
        return 0.0

    f1_score = 2 * (precision * recall) / (precision + recall)
    return f1_score


def compute_score(solution_str: str, ground_truth: str) -> float:
    """
    Calculates the final score using a hierarchical reward system.

    The reward hierarchy is as follows:
    - score: -10.0: The final output has a format or syntax error.
    - score: -10.0: The graph structure has an error (cycle).
    - score: -2.0:  The graph structure has an error (disconnected components).
    - score: [0.0, 10.0]: The format, syntax, and structure are correct. The reward is composed of two parts:
        - The F1 score of the edges, scaled to a range of [0, 5.0].
        - An additional bonus of 5.0 if the predicted DAG perfectly matches the ground truth.

    Args:
        solution_str (str): The complete string generated by the model.
        ground_truth (str): A JSON-formatted string containing only the ground truth "DAG".

    Returns:
        float: The final score.
    """
    try:
        gt_json = json.loads(ground_truth)
        if not isinstance(gt_json, dict):
             return -10.0
    except (json.JSONDecodeError, TypeError, ValueError) as e:
        print(f"FATAL ERROR: Failed to parse ground_truth data: {e}. Cannot compute score.")
        return -10.0

    # --- Level 0 & 1: Format and Syntax Checks ---
    format_penalty = compute_format_penalty(solution_str)
    if format_penalty != 0.0: return format_penalty

    syntax_penalty = compute_syntax_penalty(solution_str)
    if syntax_penalty != 0.0: return syntax_penalty

    pred_json = extract_json_block(solution_str)
    # Use empty string as default if "DAG" key is missing.
    pred_dag_str = pred_json.get("DAG", "")

    # --- Level 2: Graph Structure Penalties ---
    graph_penalty = compute_graph_penalty(pred_dag_str)
    if graph_penalty != 0.0: return graph_penalty

    # --- Level 3: Fidelity Rewards (F1 Score + Perfect Match Bonus) ---
    gt_dag_str = gt_json.get("DAG", "")

    # 1. Calculate the partial reward based on F1 score (max 5.0 points).
    f1_score = compute_edge_f1_score(pred_dag_str, gt_dag_str)
    f1_reward = f1_score * 5.0

    # 2. Check for a perfect match and calculate the bonus reward (max 5.0 points).
    pred_edges = parse_edges_from_dag_str(pred_dag_str)
    gt_edges = parse_edges_from_dag_str(gt_dag_str)

    perfect_match_bonus = 0.0
    if pred_edges == gt_edges:
        perfect_match_bonus = 5.0

    # The final score is the sum of the two reward components.
    return f1_reward + perfect_match_bonus