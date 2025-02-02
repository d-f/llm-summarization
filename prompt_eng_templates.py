from pathlib import Path
import argparse
from jinja2 import Template
from datasets import load_dataset
from typing import List, Dict


def parse_cla():
    parser = argparse.ArgumentParser()
    parser.add_argument("-eval_json", type=str)
    parser.add_argument("-num_ex", type=int)
    parser.add_argument("-input_dir", type=Path)
    parser.add_argument("-output_dir", type=Path)
    parser.parse_args()


def create_templates() -> List:
    """
    creates a list of Jinja2 templates
    """
    template_list = []

    template_1 = Template(
        "### Instruction \nWrite a concise summary of the following text \n### Input \n{{ passage }}\n### Output \n{{ summary }}"
        )

    template_2 = Template(
        "### Instruction \nExtract and summarize the main points from this text while preserving essential context \n### Input \n{{ passage }} \n### Output \n{{ summary }}"
        )

    template_3 = Template(
        "### Instruction \nProvide a two-level summary: \n1. One-sentence overview \n2. Three key takeaways \n### Input \n{{ passage }} \n### Output \n{{ summary }}"
        )

    template_4 = Template(
        "### Instruction \nSummarize this text for a general audience, highlighting what makes it noteworthy or interesting \n### Input \n{{ passage }} \n### Output \n{{ summary }}"
        )
    
    template_5 = Template(
        "### Instruction \nYou are an information compressing machine that takes in a large corpus of text and outputs the most compressed, relevant summary possible. Please compress the following passage \n### Input \n{{ passage }} \n### Output \n{{ summary }}"
        )
    
    template_list.append(template_1)
    template_list.append(template_2)
    template_list.append(template_3)
    template_list.append(template_4)
    template_list.append(template_5)
    
    return template_list


def filter_human_examples(json_ds: Dict, num_ex: int) -> List:
    """
    gets a subset of the dataset that has human-derived summarizations
    """
    examples = []
    ex_counter = 0
    for ds_dict in json_ds:
        if ex_counter == num_ex:
            break
        for summary in ds_dict["summaries"]:
            if summary["policy"] == "ref":
                examples.append((
                    ds_dict["info"]["post"],
                    summary["text"]
                ))
                ex_counter += 1
    return examples


def save_examples(template_list: List, human_ex: List, input_dir: Path, output_dir: Path) -> None:
    """
    renders each exmaple with each template and saves the text file
    """
    template_num = 1
    for template in template_list:
        example_num = 1
        for example in human_ex:
            text = template.render(passage=example[0], summary=example[1])
            partition = text.partition("### Output \n")
            input_text = partition[0] + partition[1]
            output_text = partition[2]

            with open(input_dir.joinpath(f"example_prompt{template_num}_ex{example_num}_input.txt"), mode="w") as opened_txt:
                opened_txt.write(input_text)

            with open(output_dir.joinpath(f"example_prompt{template_num}_ex{example_num}_output.txt"), mode="w") as opened_txt:
                opened_txt.write(output_text)
            
            example_num += 1
        template_num += 1


def main():
    args = parse_cla()
    template_list = create_templates()
    val_ds = load_dataset("json", data_files=args.eval_json, split="train")
    human_ex = filter_human_examples(val_ds, num_ex=args.num_ex)
    save_examples(
        template_list=template_list, 
        human_ex=human_ex, 
        input_dir=args.input_dir, 
        output_dir=args.output_dir
        )


if __name__ == "__main__":
    main()
