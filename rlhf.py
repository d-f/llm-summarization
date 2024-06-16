from datasets import load_dataset
from transformers import AutoTokenizer, BitsAndBytesConfig
from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer
from transformers import AutoModelForSequenceClassification
from tqdm import tqdm
import torch
import argparse


def parse_cla():
    parser = argparse.ArgumentParser()
    parser.add_argument("-ds_json")
    parser.add_argument("-tok_dir")
    parser.add_argument("-model_save_path")
    parser.add_argument("-lr", type=float)
    parser.add_argument("-batch_size", type=int)
    parser.add_argument("-mini_batch_size", type=int)
    parser.add_argument("-load_4bit", action="store_true")
    parser.add_argument("-quant_type")
    parser.add_argument("-dtype")
    parser.add_argument("-dbl_quant", action="store_true")
    parser.add_argument("-policy_dir")
    parser.add_argument("-reward_dir")
    return parser.parse_args()


def quant_config(
        load_in_4bit:bool , 
        bnb_4bit_quant_type:str, 
        bnb_4bit_compute_dtype:str, 
        bnb_4bit_use_double_quant:bool
        ) -> Type[BitsAndBytesConfig]:
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


def tokenize(sample:Dict, tokenizer:Type[AutoTokenizer]) -> Dict:
    """
    convert strings to tokens

    keyword arguments:
    sample -- sample to be tokenized
    tokenizer -- model to tokenize the sample
    """
    choice_idx = sample["choice"]
    sample["input_ids"] = tokenizer.encode(sample["summaries"][choice_idx]["text"])
    return sample


def prepare_tokenizer(tokenizer_dir:str) -> Type[AutoTokenizer]:
    """
    loads AutoTokenizer from files saved within
    tokenizer_dir and sets the pad_token to eos_token
    """
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def align(
        ppo_trainer:Type[PPOTrainer], 
        tokenizer:Type[AutoTokenizer], 
        reward_model:Type[AutoModelForSequenceClassification], 
        model_save_path:str
        ) -> None:
    """
    aligns model with proximal policy optimization

    keyword arguments:
    ppo_trainer -- TRL PPOTrainer object
    tokenizer -- loaded tokenizer model
    reward_model -- model to provide the reward to RLHF loop
    model_save_path -- path of folder to save model files in
    """
    for batch in tqdm(ppo_trainer.dataloader):
        response = ppo_trainer.generate(batch["input_ids"])
        batch_res = [tokenizer.decode(x.squeeze()) for x in response]

        texts = [q + r for q, r in zip(batch["input_ids"], batch_res)]
        reward_signal = reward_model(texts)
        rewards = [torch.tensor(output[1]["score"]) for output in reward_signal]

        stats = ppo_trainer.step(batch["input_ids"], response, rewards)
        ppo_trainer.log_stats(stats, batch, rewards)

    ppo_trainer.save_model(model_save_path)


def main():
    args = parse_cla()
    dataset = load_dataset("json", data_files=args.ds_json, split="train")
    config = PPOConfig(
        learning_rate=args.lr,
        batch_size=args.batch_size,
        mini_batch_size=args.mini_batch_size,
    )
    bnb_config = quant_config(
            load_in_4bit=args.load_4bit,
            bnb_4bit_quant_type=args.quant_type,
            bnb_4bit_compute_dtype=args.dtpye,
            bnb_4bit_use_double_quant=args.dbl_quant                
            )
    policy_model = AutoModelForCausalLMWithValueHead.from_pretrained(
        args.policy_dir, 
        quantization_config=bnb_config
        )
    reward_model = AutoModelForSequenceClassification.from_pretrained(
        args.reward_dir, 
        num_labels=1, 
        quantization_config=bnb_config
        )

    tokenizer = prepare_tokenizer(tokenizer_dir=args.tok_dir)
    dataset = dataset.map(lambda x: tokenize(x, tokenizer=tokenizer))
    ppo_trainer = PPOTrainer(
        model=policy_model,  
        config=config,
        dataset=dataset,
        tokenizer=tokenizer
    )
    align(
        ppo_trainer=ppo_trainer, 
        tokenizer=tokenizer, 
        reward_model=reward_model, 
        model_save_path=args.model_save_path
        )


if __name__ == "__main__":
    main()
