import json
import logging
import argparse
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluate_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--final_result_path", type=str, required=True)
    parser.add_argument("--query_split_marker", type=str, default="The user input is:\n")
    return parser.parse_args()

class ResultMerger:
    def __init__(self, args):
        self.args = args
        self.merged_map = {}

    def load_model_outputs(self):
        if not Path(self.args.output_path).exists():
            raise FileNotFoundError(f"Output path not found: {self.args.output_path}")
        
        count = 0
        with open(self.args.output_path, "r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line)
                raw_query = item.get("query", "")
                
                if self.args.query_split_marker in raw_query:
                    user_input = raw_query.split(self.args.query_split_marker)[-1].split('<|im_end|>')[0].strip()
                else:
                    user_input = raw_query.strip()
                
                self.merged_map[user_input] = {
                    "input": user_input,
                    "output": item.get("response", "")
                }
                count += 1
        logger.info(f"Loaded {count} model outputs.")

    def merge_with_ground_truth(self):
        if not Path(self.args.evaluate_path).exists():
            raise FileNotFoundError(f"Evaluate path not found: {self.args.evaluate_path}")

        with open(self.args.evaluate_path, "r", encoding="utf-8") as f:
            gt_data = json.load(f)

        match_count = 0
        for gt_item in gt_data:
            gt_input = gt_item.get("input", "")
            gt_output = gt_item.get("output", "")
            
            if gt_input in self.merged_map:
                self.merged_map[gt_input]["ground_truth"] = gt_output
                match_count += 1
        
        logger.info(f"Successfully matched {match_count} items with ground truth.")

    def save(self):
        output_file = Path(self.args.final_result_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        final_list = list(self.merged_map.values())
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(final_list, f, indent=2, ensure_ascii=False)
        logger.info(f"Final merged results saved to {self.args.final_result_path}")

def main():
    args = get_args()
    try:
        merger = ResultMerger(args)
        merger.load_model_outputs()
        merger.merge_with_ground_truth()
        merger.save()
    except Exception as e:
        logger.error(f"Error during merging process: {e}", exc_info=True)

if __name__ == "__main__":
    main()