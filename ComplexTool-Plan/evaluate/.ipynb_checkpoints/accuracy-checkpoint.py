import argparse
import json
import logging
import re
from typing import List, Dict, Set, Tuple, Optional

# --- Configuration ---
# Set up logging for clear feedback
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# --- Core Helper Functions (Mostly unchanged, added logging) ---

def extract_json_block(text: str) -> Optional[dict]:
    """
    Extracts a JSON block from a text string, handling markdown and plain JSON.
    """
    if not isinstance(text, str):
        return None

    # Pattern for ```json ... ```
    pattern = re.compile(r"```json(.*?)```", re.DOTALL)
    match = pattern.search(text)
    
    json_string = ""
    if match:
        json_string = match.group(1).strip()
    else:
        # If no markdown, assume the whole text is a JSON string
        json_string = text.strip()

    if not json_string:
        return None
        
    try:
        return json.loads(json_string)
    except json.JSONDecodeError:
        logging.warning(f"Failed to decode JSON. Content preview: {json_string[:100]}...")
        return None


def parse_dag_data(data: dict) -> Tuple[Set[str], Set[str]]:
    """
    Parses a dictionary to extract sets of nodes and edges from a 'DAG' field.
    Nodes are inferred directly from the edges.
    """
    try:
        dag_string = data.get('DAG', '')
        if not dag_string:
            return set(), set()

        edges = {edge.replace(" ", "").strip() for edge in dag_string.split(',') if edge.strip()}

        nodes = set()
        for edge in edges:
            if '->' in edge:
                try:
                    source, target = edge.split('->')
                    if source: nodes.add(source)
                    if target: nodes.add(target)
                except ValueError:
                    logging.warning(f"Malformed edge found and skipped: '{edge}'.")
                    continue
        
        return nodes, edges
    
    except (AttributeError, TypeError) as e:
        logging.warning(f"Error parsing DAG data. Expected a dict with a 'DAG' key. Error: {e}")
        return set(), set()


def calculate_metrics(pred_set: Set[str], true_set: Set[str]) -> Dict[str, float]:
    """
    Calculates precision, recall, and F1-score for two sets.
    """
    intersection = len(pred_set.intersection(true_set))

    precision = intersection / len(pred_set) if len(pred_set) > 0 else (1.0 if len(true_set) == 0 else 0.0)
    recall = intersection / len(true_set) if len(true_set) > 0 else (1.0 if len(pred_set) == 0 else 0.0)
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
    return {'precision': precision, 'recall': recall, 'f1_score': f1_score}


# --- Main Logic Functions (Refactored for clarity and structure) ---

def load_and_prepare_data(file_path: str) -> List[Dict[str, dict]]:
    """
    Loads data from the input file, extracts JSON, and prepares it for evaluation.
    This replaces the original script's file-write-then-read logic.
    """
    logging.info(f"Loading and preparing data from {file_path}...")
    prepared_data = []
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            all_data = json.load(f)
    except FileNotFoundError:
        logging.error(f"Input file not found: {file_path}")
        return []
    except json.JSONDecodeError:
        logging.error(f"Could not parse input file as JSON: {file_path}")
        return []

    for i, item in enumerate(all_data):
        try:
            output = extract_json_block(item["output"])
            ground_truth = extract_json_block(item["ground_truth"])
            
            if output is not None and ground_truth is not None:
                prepared_data.append({'output': output, 'ground_truth': ground_truth})
            else:
                logging.warning(f"Skipping record {i+1} due to missing or invalid JSON in 'output' or 'ground_truth'.")

        except KeyError as e:
            logging.warning(f"Skipping record {i+1} due to missing key: {e}")
            continue
            
    logging.info(f"Successfully prepared {len(prepared_data)} records for evaluation.")
    return prepared_data


def evaluate_batch(batch_data: List[Dict[str, dict]]) -> Dict:
    """
    Processes a list of prepared data, evaluates each, and aggregates the results.
    """
    if not batch_data:
        logging.warning("No data to evaluate.")
        return {}

    all_results = []
    for item in batch_data:
        pred_nodes, pred_edges = parse_dag_data(item['output'])
        true_nodes, true_edges = parse_dag_data(item['ground_truth'])
        
        node_metrics = calculate_metrics(pred_nodes, true_nodes)
        edge_metrics = calculate_metrics(pred_edges, true_edges)
        dag_accuracy = 1.0 if pred_edges == true_edges else 0.0
        
        all_results.append({
            'node_accuracy': node_metrics,
            'edge_accuracy': edge_metrics,
            'dag_accuracy': dag_accuracy
        })
    
    # Aggregation logic
    total = len(all_results)
    agg = {'node_accuracy': {'precision': 0, 'recall': 0}, 'edge_accuracy': {'precision': 0, 'recall': 0}, 'dag_accuracy': 0}
    
    for res in all_results:
        for key in ['node_accuracy', 'edge_accuracy']:
            agg[key]['precision'] += res[key]['precision']
            agg[key]['recall'] += res[key]['recall']
        agg['dag_accuracy'] += res['dag_accuracy']
    
    # Calculate final macro-averages
    final_report = {
        'total_samples': total,
        'node_accuracy': {},
        'edge_accuracy': {},
        'dag_exact_match_accuracy': agg['dag_accuracy'] / total
    }
    for metric_type in ['node_accuracy', 'edge_accuracy']:
        avg_p = agg[metric_type]['precision'] / total
        avg_r = agg[metric_type]['recall'] / total
        avg_f1 = 2 * (avg_p * avg_r) / (avg_p + avg_r) if (avg_p + avg_r) > 0 else 0.0
        final_report[metric_type] = {'precision': avg_p, 'recall': avg_r, 'f1_score': avg_f1}
        
    return final_report


def display_and_save_report(report: Dict, output_file: Optional[str]):
    """
    Prints the final report to the console and saves it to a file if specified.
    """
    if not report:
        logging.warning("Cannot generate report: No results were aggregated.")
        return

    total_samples = report['total_samples']
    
    # Display report
    print("\n" + "="*50)
    print(f"  DAG Evaluation Summary Report ({total_samples} records)")
    print("="*50)
    
    for name, metrics in [("Node Metrics", report['node_accuracy']), ("Edge Metrics", report['edge_accuracy'])]:
        print(f"\n{name}:")
        print(f"  - Avg Precision: {metrics['precision']:.4f}")
        print(f"  - Avg Recall:    {metrics['recall']:.4f}")
        print(f"  - Avg F1-Score:  {metrics['f1_score']:.4f}")
    
    print("\nDAG Exact Match Accuracy:")
    print(f"  - Accuracy:      {report['dag_exact_match_accuracy']:.4f}")
    print("\n" + "="*50)

    # Save report to file
    if output_file:
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=4)
            logging.info(f"Evaluation report successfully saved to {output_file}")
        except IOError as e:
            logging.error(f"Failed to save report to {output_file}: {e}")


def main():
    """Main function to run the evaluation from the command line."""
    parser = argparse.ArgumentParser(description="Evaluate DAG generation performance from model outputs.")
    parser.add_argument(
        "-i", "--input-file", 
        required=True, 
        help="Path to the input JSON file containing 'output' and 'ground_truth' fields."
    )
    parser.add_argument(
        "-o", "--output-file", 
        default=None, 
        help="Optional path to save the final evaluation report as a JSON file."
    )
    args = parser.parse_args()

    # 1. Load and clean data
    prepared_data = load_and_prepare_data(args.input_file)
    
    # 2. Run evaluation and aggregation
    final_report = evaluate_batch(prepared_data)
    
    # 3. Display and optionally save the report
    display_and_save_report(final_report, args.output_file)


if __name__ == "__main__":
    main()