from tqdm import tqdm
from rouge_score import rouge_scorer
from pathlib import Path
import argparse
from typing import Dict


def parse_cla():
    parser = argparse.ArgumentParser()
    parser.add_argument("-model_output_dir", type=Path)
    parser.add_argument("-gt_output_dir", type=Path)
    return parser.parse_args()


def evaluate_summary(model_output: str, ground_truth: str) -> Dict:
    """
    measured rouge-1, rouge-2 and rouge-L between two passages of text
    """
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    return scorer.score(ground_truth, model_output)


def main():
    args = parse_cla()
    rouge_1 = {
        "prompt 1": {"precision": 0, "recall": 0, "f1": 0},
        "prompt 2": {"precision": 0, "recall": 0, "f1": 0},
        "prompt 3": {"precision": 0, "recall": 0, "f1": 0},
        "prompt 4": {"precision": 0, "recall": 0, "f1": 0},
        "prompt 5": {"precision": 0, "recall": 0, "f1": 0},
    }

    rouge_2 = {
            "prompt 1": {"precision": 0, "recall": 0, "f1": 0},
            "prompt 2": {"precision": 0, "recall": 0, "f1": 0},
            "prompt 3": {"precision": 0, "recall": 0, "f1": 0},
            "prompt 4": {"precision": 0, "recall": 0, "f1": 0},
            "prompt 5": {"precision": 0, "recall": 0, "f1": 0},
        }
    
    rouge_L = {
            "prompt 1": {"precision": 0, "recall": 0, "f1": 0},
            "prompt 2": {"precision": 0, "recall": 0, "f1": 0},
            "prompt 3": {"precision": 0, "recall": 0, "f1": 0},
            "prompt 4": {"precision": 0, "recall": 0, "f1": 0},
            "prompt 5": {"precision": 0, "recall": 0, "f1": 0},
        }

    for pred_file in tqdm(args.model_output_dir.iterdir()):
        gt_file = args.gt_output_dir.joinpath(pred_file.name.partition("input.txt")[0]+"output.txt")
        with open(pred_file) as opened_pred:
            pred = opened_pred.read()
        with open(gt_file) as opened_gt:
            gt = opened_gt.read()

        results = evaluate_summary(model_output=pred, ground_truth=gt)  

        prompt_num = int(gt_file.name.partition("prompt")[2].partition("_")[0])

        rouge_1[f"prompt {prompt_num}"]["precision"] += results["rouge1"].precision
        rouge_1[f"prompt {prompt_num}"]["recall"] += results["rouge1"].recall
        rouge_1[f"prompt {prompt_num}"]["f1"] += results["rouge1"].fmeasure

        rouge_2[f"prompt {prompt_num}"]["precision"] += results["rouge2"].precision
        rouge_2[f"prompt {prompt_num}"]["recall"] += results["rouge2"].recall
        rouge_2[f"prompt {prompt_num}"]["f1"] += results["rouge2"].fmeasure

        rouge_L[f"prompt {prompt_num}"]["precision"] += results["rougeL"].precision
        rouge_L[f"prompt {prompt_num}"]["recall"] += results["rougeL"].recall
        rouge_L[f"prompt {prompt_num}"]["f1"] += results["rougeL"].fmeasure

    rouge_1 = {k: {k2: v2/10 for k2, v2 in v.items()} for k, v in rouge_1.items()}
    rouge_2 = {k: {k2: v2/10 for k2, v2 in v.items()} for k, v in rouge_2.items()}
    rouge_L = {k: {k2: v2/10 for k2, v2 in v.items()} for k, v in rouge_L.items()}


if __name__ == "__main__":
    main()
