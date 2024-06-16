from pathlib import Path
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
import peft
from trl import SFTConfig, SFTTrainer
from datasets import load_dataset
from typing import Type, List
import argparse


def parse_cla():
    """parses command-line arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument("-load_4bit", action="store_true")
    parser.add_argument("-quant_type")
    parser.add_argument("-dtype")
    parser.add_argument("-dbl_quant", action="store_true")
    parser.add_argument("-model_dir", type=Path)
    parser.add_argument("-lora_a", type=int)
    parser.add_argument("-lora_drop", type=float)
    parser.add_argument("-r", type=int)
    parser.add_argument("-bias")
    parser.add_argument("-task_type")
    parser.add_argument("-target_mods", type=list, nargs="+", default=["q_proj", "v_proj"])
    parser.add_argument("-ds_json")
    parser.add_argument("-packing", action="store_true")
    parser.add_argument("-ds_txt_field")
    parser.add_argument("-output_dir", type=Path)
    parser.add_argument("-batch_size", type=int)
    parser.add_argument("-bf16", action="store_true")
    parser.add_argument("-max_len", type=int)
    parser.add_argument("-eval_strat")
    parser.add_argument("-do_eval", action="store_true")
    parser.add_argument("-model_save_path")
    return parser.parse_args()


def quant_config(
    load_in_4bit:bool, 
    bnb_4bit_quant_type:str, 
    bnb_4bit_compute_dtype:str, 
    bnb_4bit_use_double_quant:bool
) -> Type[BitsAndBytesConfig]:
    """
    defines quantization configuration
    
    keyword arguments:
    load_in_4bit -- 4-bit precision
    bnb_4bit_quant_type -- quantization data type {nf4, fp4}
    bnb_4bit_compute_dtype -- data type for computation
    bnb_4bit_use_double_quant -- nested quantization
    """

    return BitsAndBytesConfig(
        load_in_4bit=load_in_4bit,
        bnb_4bit_quant_type=bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=bnb_4bit_compute_dtype,
        bnb_4bit_use_double_quant=bnb_4bit_use_double_quant,
    )


def causal_lm(model_dir:Path, quant_config:Type[BitsAndBytesConfig]) -> Type[AutoModelForCausalLM]:
    """loads causal llm"""
    return AutoModelForCausalLM.from_pretrained(
        model_dir,
        quantization_config=quant_config
    )


def lora_config(
    lora_alpha:int, 
    lora_dropout:float, 
    r:int, 
    bias:str, 
    task_type:str, 
    target_modules: List
) -> Type[peft.LoraConfig]:
    """
    defines lora configuration

    keyword arguments:
    lora_alpha -- alpha parameters for lora scaling
    lora_dropout -- dropout probability for lora layers
    r -- lora attention rank
    bias -- bias type
    task_type -- string for task such as "CAUSAL_LM"
    target_modules -- modules to apply adapter to
    """
    return peft.LoraConfig(
    lora_alpha=lora_alpha,
    lora_dropout=lora_dropout,
    r=r,
    bias=bias,
    task_type=task_type,
    target_modules=target_modules
)


def peft_model(model:Type[AutoModelForCausalLM], l_config:Type[peft.LoraConfig]) -> Type[AutoModelForCausalLM]:
    """converts model to peft model"""
    return peft.get_peft_model(model, l_config)


def quant_model(model:Type[AutoModelForCausalLM]) -> Type[AutoModelForCausalLM]:
    """prepares model for quantization"""
    return peft.prepare_model_for_kbit_training(model)


def sft_config(
        packing:bool, dataset_text_field:str, output_dir:Path, dataset_batch_size:int, 
        bf16:bool, max_seq_len:int, eval_strat:str, do_eval:bool
        ) -> Type[SFTConfig]:
    """
    defines the supervised fine tuning configuration

    keyword arguments:
    packing -- bool determining whether to pack dataset sequences
    dataset_text_field -- name of the text field in the dataset
    output_dir -- folder to save logs
    dataset_batch_size -- batch size for training
    bf16 -- bool determining whether to use bf16 data type
    max_seq_len -- maximum sequence length
    eval_strat -- string determining when to perform evaluation
    do_eval -- bool determining whether to evaluate on eval dataset
    """
    return SFTConfig(
        packing=packing, dataset_text_field=dataset_text_field, output_dir=output_dir, 
        dataset_batch_size=dataset_batch_size, bf16=bf16, eval_strategy=eval_strat, do_eval=do_eval,
        max_seq_length=max_seq_len
    )


def train_model(
        model: Type[AutoModelForCausalLM], dataset, s_config:Type[SFTConfig], 
        eval_dataset, model_save_path:str
        ):
    """trains model with SFTTrainer"""
    trainer = SFTTrainer(
        model,
        train_dataset=dataset,
        eval_dataset=eval_dataset,
        args=s_config,
    )
    trainer.train()
    trainer.save_model(model_save_path)


def main():
    args = parse_cla()
    bnb_config = quant_config(
        load_in_4bit=args.load_4bit,
        bnb_4bit_quant_type=args.quant_type,
        bnb_4bit_compute_dtype=args.dtype,
        bnb_4bit_use_double_quant=args.dbl_quant
    )

    model = causal_lm(model_dir=args.model_dir, quant_config=bnb_config)
    model = quant_model(model)
    
    l_config = lora_config(
        lora_alpha=args.lora_a,
        lora_dropout=args.lora_drop,
        r=args.r,
        bias=args.bias,
        task_type=args.task_type,
        target_modules=args.target_mods
    )

    model = peft_model(model=model, l_config=l_config)

    train_ds, val_ds = load_dataset("json", data_files=args.ds_json, split=["train[:90%]", "train[89%:100%]"])

    s_config = sft_config(
        packing=args.packing, dataset_text_field=args.ds_txt_field, output_dir=args.output_dir, 
        dataset_batch_size=args.batch_size, bf16=args.bf16, max_seq_len=args.max_len, eval_strat=args.eval_strat, do_eval=args.do_eval
        )

    train_model(model=model, dataset=train_ds, eval_dataset=val_ds, s_config=s_config, model_save_path=args.model_save_path)


if __name__ == "__main__":
    main()
