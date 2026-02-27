import re
import json
import logging
import argparse
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Set, Tuple, Optional

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DAGMetricsEvaluator:
    def __init__(self):
        self.re_json = re.compile(r"```json(.*?)```", re.DOTALL)

    def extract_json(self, text: str) -> Optional[dict]:
        if not isinstance(text, str):
            return None
        match = self.re_json.search(text)
        if match:
            try:
                return json.loads(match.group(1).strip())
            except json.JSONDecodeError:
                return None
        try:
            return json.loads(text)
        except (json.JSONDecodeError, TypeError):
            return None

    def parse_dag(self, data: any) -> Tuple[Set[str], Set[str]]:
        try:
            if isinstance(data, str):
                data = json.loads(data)
            
            dag_str = data.get('DAG', '')
            if not dag_str:
                return set(), set()
            
            edges = {e.replace(" ", "").strip() for e in dag_str.split(',') if e.strip()}
            nodes = set()
            for edge in edges:
                if '->' in edge:
                    parts = edge.split('->')
                    if len(parts) == 2:
                        if parts[0].strip(): nodes.add(parts[0].strip())
                        if parts[1].strip(): nodes.add(parts[1].strip())
            return nodes, edges
        except Exception:
            return set(), set()

    @staticmethod
    def calculate_f1(pred: Set[str], true: Set[str]) -> Dict[str, float]:
        intersection = len(pred.intersection(true))
        precision = intersection / len(pred) if len(pred) > 0 else (1.0 if len(true) == 0 else 0.0)
        recall = intersection / len(true) if len(true) > 0 else (1.0 if len(pred) == 0 else 0.0)
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        return {'precision': precision, 'recall': recall, 'f1_score': f1}

    def evaluate_item(self, pred_data: any, true_data: any) -> Dict:
        p_nodes, p_edges = self.parse_dag(pred_data)
        t_nodes, t_edges = self.parse_dag(true_data)
        
        return {
            'node': self.calculate_f1(p_nodes, t_nodes),
            'edge': self.calculate_f1(p_edges, t_edges),
            'exact_match': 1.0 if p_nodes == t_nodes and p_edges == t_edges else 0.0
        }

    def aggregate(self, results: List[Dict]) -> Dict:
        if not results:
            return {}
        
        count = len(results)
        sums = {
            'node_p': sum(r['node']['precision'] for r in results),
            'node_r': sum(r['node']['recall'] for r in results),
            'edge_p': sum(r['edge']['precision'] for r in results),
            'edge_r': sum(r['edge']['recall'] for r in results),
            'em': sum(r['exact_match'] for r in results)
        }
        
        avg_node_p = sums['node_p'] / count
        avg_node_r = sums['node_r'] / count
        avg_edge_p = sums['edge_p'] / count
        avg_edge_r = sums['edge_r'] / count
        
        return {
            'node': {
                'p': avg_node_p,
                'r': avg_node_r,
                'f1': (2 * avg_node_p * avg_node_r) / (avg_node_p + avg_node_r) if (avg_node_p + avg_node_r) > 0 else 0.0
            },
            'edge': {
                'p': avg_edge_p,
                'r': avg_edge_r,
                'f1': (2 * avg_edge_p * avg_edge_r) / (avg_edge_p + avg_edge_r) if (avg_edge_p + avg_edge_r) > 0 else 0.0
            },
            'exact_match': sums['em'] / count
        }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True)
    args = parser.parse_args()

    evaluator = DAGMetricsEvaluator()
    
    with open(args.input_file, "r", encoding="utf-8") as f:
        raw_data = json.load(f)
    
    processed_results = []
    for item in raw_data:
        try:
            out_json = evaluator.extract_json(item.get("output"))
            gt_json = evaluator.extract_json(item.get("ground_truth"))
            if out_json and gt_json:
                processed_results.append(evaluator.evaluate_item(out_json, gt_json))
        except Exception:
            continue

    summary = evaluator.aggregate(processed_results)
    
    print("="*50)
    print(f"Evaluation Summary (Samples: {len(processed_results)})")
    print("="*50)
    for key in ['node', 'edge']:
        res = summary[key]
        print(f"\n[{key.upper()} METRICS]")
        print(f"  Precision: {res['p']:.4f}")
        print(f"  Recall:    {res['r']:.4f}")
        print(f"  F1-Score:  {res['f1']:.4f}")
    
    print(f"\n[EXACT MATCH]")
    print(f"  Accuracy:  {summary['exact_match']:.4f}")
    print("="*50)

if __name__ == "__main__":
    main()