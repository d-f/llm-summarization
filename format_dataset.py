from pathlib import Path
from tqdm import tqdm
import json
import datasets
from typing import Dict, Type, List
import argparse


def parse_cla():
    """parses command-line arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument("-save_path", type=Path)
    return parser.parse_args()


def dataset_dict(content:str, summary:str) -> Dict:
    """
    takes a input and a summary and concatenates together with 
    stanford alpaca syntax for summarization

    keyword arguments:
    content -- content which should be summarized
    summary -- summary of the content
    """
    input_txt = f"### Instruction \nWrite a concise summary of the following text \n### Input \n{content}"
    output_txt = f"### Output {summary}"
    return {"prompt": input_txt, "completion": output_txt}


def load_dataset() -> Type[datasets.Dataset]:
    """returns tldr dataset"""
    return datasets.load_dataset("webis/tldr-17")


def save_list(tldr_dataset:Type[datasets.Dataset]) -> List:
    """saves list of dataset dictionaries"""
    save_list = []
    for text_dict in tqdm(tldr_dataset["train"]):
        prompt = dataset_dict(content=text_dict["content"],  summary=text_dict["summary"])
        save_list.append(prompt)
    return save_list


def save_jsonl(save_list:List, save_path:Path):
    """saves jsonl file"""
    with open(save_path, mode="w") as opened_jsonl:
        for json_dict in save_list:
            json.dump(json_dict, opened_jsonl)
            opened_jsonl.write("\n")


def main():
    args = parse_cla()
    dataset = load_dataset()
    s_list = save_list(tldr_dataset=dataset)
    save_jsonl(save_list=s_list, save_path=args.save_path)


if __name__ == "__main__":
    main()
