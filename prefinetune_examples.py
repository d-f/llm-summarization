import datasets
import pandas as pd
from pathlib import Path
import argparse


def parse_cla():
    parser = argparse.ArgumentParser()
    parser.add_argument("-save_path", type=Path)
    parser.add_argument("-num_ex", type=int)
    return parser.parse_args()


def load_dataset():
    return datasets.load_dataset("webis/tldr-17")


def generate_prompt(text_body):
    prompt = f"### Instruction: Write a concise summary of the following text.\n```{text_body}```\nSUMMARY:"
    return prompt


def save_prompts(tldr_dataset, num_prompts, save_path):
    i = 0

    for text_dict in tldr_dataset["train"]:
        if i == num_prompts:
            break
        prompt = generate_prompt(text_dict["content"])
        with open(save_path.joinpath(f"example_prompt{i}.txt"), mode="w") as opened_txt:
            opened_txt.write(prompt)
        i += 1
    

def main():
    args = parse_cla()
    tldr_dataset = load_dataset()
    save_prompts(tldr_dataset=tldr_dataset, num_prompts=args.num_ex, save_path=args.save_path)


if __name__ == "__main__":
    main()
