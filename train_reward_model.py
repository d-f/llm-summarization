from datasets import load_dataset
from transformers import AutoTokenizer, BitsAndBytesConfig
from pathlib import Path
from typing import Dict, Type, Tuple
import argparse
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from trl import RewardTrainer, RewardConfig
from peft import LoraConfig, TaskType


def parse_cla():
    parser = argparse.ArgumentParser()
    parser.add_argument("-model_folder", type=Path)
    parser.add_argument("-train_json", type=str)
    parser.add_argument("-val_json", type=str)
    parser.add_argument("-model_save_name", type=str)
    parser.add_argument("-r", type=int)
    parser.add_argument("-lora_a", type=int)
    parser.add_argument("-lora_dropout", type=float)
    parser.add_argument("-load_4bit", action="store_true")
    parser.add_argument("-quant_type")
    parser.add_argument("-dtype")
    parser.add_argument("-dbl_quant", action="store_true")
    parser.add_argument("-lr", type=float)
    parser.add_argument("-bf16", action="store_true")
    parser.add_argument("-max_len", type=int)
    parser.add_argument("-batch_size", type=int)
    parser.add_argument("-output_dir")
    parser.add_argument("-target_mods", type=list, nargs="+", default=["q_proj", "v_proj"])
    return parser.parse_args()


def quant_config(
        load_in_4bit:bool , bnb_4bit_quant_type:str, bnb_4bit_compute_dtype:str, 
        bnb_4bit_use_double_quant:bool
        ) -> BitsAndBytesConfig:
    """
    defines quantization configuration
    
    keyword arguments:
    load_in_4bit -- 4-bit precision
    bnb_4bit_quant_type -- quantizationd data type {nf4, fp4}
    bnb_4bit_compute_dtype -- data type for computation
    bnb_4bit_use_double_quant -- nested quantization
    """

    return BitsAndBytesConfig(
        load_in_4bit=load_in_4bit,
        bnb_4bit_quant_type=bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=bnb_4bit_compute_dtype,
        bnb_4bit_use_double_quant=bnb_4bit_use_double_quant,
    )


def preprocess(examples:Dict, tokenizer:Type[AutoTokenizer]) -> Dict:
    new_examples = {
            "input_ids_chosen": [],
            "attention_mask_chosen": [],
            "input_ids_rejected": [],
            "attention_mask_rejected": [],
        }
    for batch_idx in range(len(examples["summaries"])):
        choice = examples["choice"][batch_idx]
        chosen = examples["summaries"][batch_idx][choice]["text"]

        if choice == 0:
            rejected = examples["summaries"][batch_idx][1]["text"]
        else:
            rejected = examples["summaries"][batch_idx][0]["text"]

        tokenized_chosen = tokenizer(chosen)
        tokenized_rejected = tokenizer(rejected)

        new_examples["input_ids_chosen"].append(tokenized_chosen["input_ids"])
        new_examples["attention_mask_chosen"].append(tokenized_chosen["attention_mask"])
        new_examples["input_ids_rejected"].append(tokenized_rejected["input_ids"])
        new_examples["attention_mask_rejected"].append(tokenized_rejected["attention_mask"])

    return new_examples


def prepare_datasets(train_ds, val_ds, tokenizer:Type[AutoTokenizer]) -> Tuple:
    """
    prepares dataset by mapping the inputs with the preprocess function
    """
    train_ds = train_ds.map(lambda x: preprocess(x, tokenizer=tokenizer), batched=True)
    val_ds = val_ds.map(lambda x: preprocess(x, tokenizer=tokenizer), batched=True)

    return train_ds, val_ds


def prepare_tokenizer(model_folder, use_fast):
    """
    prepares tokenizer by initializing AutoTokenizer and
    setting the pad_token as eos_token
    """
    tokenizer = AutoTokenizer.from_pretrained(model_folder, use_fast=use_fast)
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def prepare_model(model_folder, bnb_config):
    """
    prepares a model by initializing AutoModelForSequenceClassification
    and setting the pad_token_id to the eos_token_is
    """
    model = AutoModelForSequenceClassification.from_pretrained(
        model_folder, quantization_config=bnb_config, num_labels=1
    )
    model.config.pad_token_id = model.config.eos_token_id
    return model

  
def main():
    args = parse_cla()
    train_ds = load_dataset("json", data_files=args.train_json, split="train")
    val_ds = load_dataset("json", data_files=args.val_json, split="train")
    tokenizer = prepare_tokenizer(model_folder=args.model_folder, use_fast=False)
    train_ds, val_ds = prepare_datasets(train_ds=train_ds, val_ds=val_ds, tokenizer=tokenizer)
    peft_config = LoraConfig(
    task_type=TaskType.SEQ_CLS,
    inference_mode=False,
    r=args.r,
    lora_alpha=args.lora_a,
    lora_dropout=args.lora_dropout,
    target_modules=args.target_mods
)
    bnb_config = quant_config(
        load_in_4bit=args.load_4bit,
        bnb_4bit_quant_type=args.quant_type,
        bnb_4bit_compute_dtype=args.dtype,
        bnb_4bit_use_double_quant=args.dbl_quant                
        )
    config = RewardConfig(
        output_dir=args.output_dir, 
        do_eval=True, 
        do_train=True, 
        learning_rate=args.lr,
        bf16=args.bf16,
        max_length=args.max_len,
        remove_unused_columns=False,
        per_device_train_batch_size=args.batch_size
        )
    model = prepare_model(model_folder=args.model_folder, bnb_config=bnb_config)
    trainer = RewardTrainer(
        model=model,
        tokenizer=tokenizer,
        args=config,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        peft_config=peft_config,
    )
    trainer.train()
    trainer.save_model(args.model_save_name)





if __name__ == "__main__":
    main()
