import random
import argparse
from pathlib import Path
from typing import Dict, Tuple
import json


def parse_cla():
    """
    parses command-line arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-feedback_folder", type=Path)
    parser.add_argument("-val_prop", type=float)
    parser.add_argument("-save_folder", type=Path)
    parser.add_argument("-train_filename")
    parser.add_argument("-val_filename")
    return parser.parse_args()


def open_ai_data(json_folder: Path) -> Dict:
    """
    reads all of the open ai JSON files and
    results list of loaded json strings
    """
    open_ai = []
    for folder_path in json_folder.iterdir():
        with open(folder_path) as opened_json:
            openai_data = list(opened_json)
        for json_str in openai_data:
            result = json.loads(json_str)
            open_ai.append(result)
    return open_ai


def save_ds(json_data, json_path):
    """
    saves dataset
    """
    with open(json_path, mode="w") as opened_json:
        json.dump(json_data, opened_json)


def partition_ds(ds, val_prop) -> Tuple:
    """
    partitions datset into train and validation partitions
    """
    val_amt = int(len(ds)*val_prop)
    val_part = random.sample(ds, k=val_amt)
    train_part = [x for x in ds if x not in val_part]
    return train_part, val_part


def main():
    args = parse_cla()
    feedback_data = open_ai_data(args.feedback_folder)
    train_ds, val_ds = partition_ds(ds=feedback_data, val_prop=args.val_prop)
    save_ds(json_data=train_ds, json_path=args.save_folder.joinpath(args.train_filename))
    save_ds(json_data=val_ds, json_path=args.save_folder.joinpath(args.val_filename))


if __name__ == "__main__":
    main()
