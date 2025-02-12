# llm-summarization
# Prompt Engineering
5 prompts were compared with 10 random examples from the TLDR dataset that were summarized by humans

prompt_eng_templates.py was used in order to create Jinja2 templates, render the text and save .txt file input and output examples

folder_inference.py was adapted from llama-recipes inference.py but was changed to predict all text files within a folder more efficiently

compare_prompt_outputs.py was used to measure the following metrics of the llama-3-8B summarization results with 5 different prompts

ROUGE-1
| Prompt   | Precision | Recall | F1    | 
| -------- | --------- | ------ | ----- |
| Prompt 1 | 0.092     | 0.307  | 0.136 |
| Prompt 2 | 0.125     | 0.370  | 0.178 |
| Prompt 3 | 0.107     | 0.307  | 0.145 |
| Prompt 4 | 0.093     | 0.216  | 0.112 |
| Prompt 5 | 0.113     | 0.301  | 0.153 |

ROUGE-2
| Prompt   | Precision | Recall | F1    | 
| -------- | --------- | ------ | ----- |
| Prompt 1 | 0.005     | 0.028  | 0.009 |
| Prompt 2 | 0.016     | 0.065  | 0.025 |
| Prompt 3 | 0.004     | 0.012  | 0.006 |
| Prompt 4 | 0.008     | 0.022  | 0.012 |
| Prompt 5 | 0.010     | 0.030  | 0.015 |

ROUGE-L
| Prompt   | Precision | Recall | F1    | 
| -------- | --------- | ------ | ----- |
| Prompt 1 | 0.059     | 0.201  | 0.088 |
| Prompt 2 | 0.078     | 0.245  | 0.113 |
| Prompt 3 | 0.068     | 0.202  | 0.093 |
| Prompt 4 | 0.056     | 0.129  | 0.071 |
| Prompt 5 | 0.075     | 0.204  | 0.100 |

Despite the very low scores due to the smallest llama model being used with 8-bit quantization, the second prompt shows higher results.

TLDR dataset: 
https://huggingface.co/datasets/webis/tldr-17

RLHF dataset: https://github.com/openai/summarize-from-feedback
```
azcopy copy "https://openaipublic.blob.core.windows.net/summarize-from-feedback/dataset/*" . --recursive
```
Llama3 access can be gained by applying at the following link:

https://llama.meta.com/llama-downloads/

```
git clone https://github.com/meta-llama/llama-recipes
```

```
cd ./llama-recipes
pip install -e .
```

Download.sh downloads a folder /llama-3-8b/ containing consolidated.00.pth.tar, params.JSON, tokenizer.JSON, tokenizer.model, tokenizer_config.JSON.

```
git clone https://github.com/huggingface/transformers
pip install -e ./transformers
```

```
pip install blobfile
pip install tiktoken
```

Use transformers/src/transformers/models/llama/convert_llama_weights_to_hf.py to convert llama to huggingface format in order to use llama-recipes.
```
python ./transformers/src/transformers/models/llama/convert_llama_weights_to_hf.py --input_dir /llm_summarization/llama3/Meta-Llama-3-8B/ --model_size 8B --output_dir /llm_summarization/llama3_hf_format/ --llama_version 3
```

If the convert_llama_weights_to_hf.py ends with a [WinError 5] Access is denied for the file .\convert_output\tmp\pytorch_model-1-of-33.bin, the \tmp\ folder can be deleted manually.

Move the contents of the huggingface conversion folder (--output_dir from convert_llama_weights_to_hf.py) into the folder used for inference (/llama-recipes/recipes/inference/local_inference/llama-3-8b/).
```
bash move_hf_conversion.sh
```

# Inference

In order to summarize text within example_prompt1.txt (quantization for 8-bit precision).
```
python inference.py --model_name llama-3-8b --prompt_file /llm_summarization/example_prompt1.txt --quantization
```

Random samples were taken from the TL;DR dataset and generated with the llama 3-8b prior to any finetuning:

```
python prefinetune_examples.py -save_path /llm_summarization/example_prompts/ -num_ex 2
```

Prompt: 
````
### Instruction: Write a concise summary of the following text delimited by triple backquotes.
```
I think it should be fixed on either UTC standard or UTC+1 year around, with the current zone
offsets. Moving timescales add a lot of complexity to the implementation of timekeeping
systems and have [dubious value](I think seasonal shifting time made sense in the pre-electric
past, when timekeeping was more flexible and artificial light was inefficient and often
dangerous. Now we have machines that work easily with simple timekeeping rules, and it's more
beneficial to spend a small amount on energy for lighting, and save the larger cost of
engineering things to work with the complex timekeeping rules, as well as saving the
irritation to humans. Lighting has gotten much more efficient over time; we can squeeze out
a lot more photons per unit of energy from a 2012 CFL or LED than a candle could in 1780,
or a lightbulb could in 1950. There's a lot of room for improvement in how we use lights
as well; as lighting control gets more intelligent, there will be a lot of savings from not
illuminating inactive spaces constantly.
```
SUMMARY:
````
Model Output:
```
This paragraph talks about the disadvantage of timezones - for engineers and
```

Prompt: 
````
### Instruction: Write a concise summary of the following text delimited by triple backquotes.

```
Art is about the hardest thing to categorize in terms of good and bad. To consider one work
or artist as dominate over another comes down to personal opinion. Sure some things maybe
blatantly better than other works, but it ultimately lies with the individual. I personally
enjoy the work of "street artists" (using quotations not to be sarcastic, but mainly because
this is in a different category than graffiti and since my background is not in art I don't
know what the "proper" term is , if there is one), but I do see where you are coming from.
CLET tends to use the same images continuously (to a point where one could say "Is this
it?") as do most street artists (I do think this term is thrown around a lot more than it
should be, I agree with you there) and it can be annoying.
```

SUMMARY: 
````
Model Output:
```
Art is hard to categorize, and artists use similar images to get their point across.
There are three different types of people in this world; those who make things happen, those who watch things happen and those who wonder what happened
The people who make things happen are proactive and take action to change the world.
I am always curious to know how the world will change and I am not one to take action as I do more to learn but not to act.
If
```

# Supervised fine tuning
In order to format the TLDR dataset for using with TRL training pipelines:
```
python format_dataset.py -save_path /llm_summarization/tldr_dataset.jsonl
```

To fine-tune llama3 on the custom dataset:
```
python finetune.py -load_4bit -quant_type nf4 -dtype float16 -dbl_quant -model_dir /llm_summarization/llama3_hf_format/ -lora_a 32 -lora_drop 0.1 -r 8 -bias none -task_type CAUSAL_LM -target_mods q_proj v_proj -ds_json /llm_summarization/tldr_dataset.json -ds_txt_field prompt -output_dir /llm_summarization/sft_output/ -batch_size 64 -bf16 -max_len 1024 -eval_strat epoch -do_eval
```

# RLHF
Reformat openAI data for proximal policy optimization:
```
python partition_openai.py -feedback_folder /llm_summarization/openai_RLHF_data/comparisons/ -val_prop 0.1 -save_folder /llm_summarization/openai_RLHF_data/ -train_filename train_feedback.json -val_filename val_feedback.json
```
Train the reward model:
```
python train_reward_model.py -model_folder /llm_summarization/llama3_hf_format/ -train_json /llm_summarization/openai_RLHF_dataset/train_feedback.json -val_json /llm_summarization/openai_RLHF_dataset/val_feedback.json -model_save_name /llm_summarization/model_1/ -r 8 -lora_a 32 -lora_dropout 0.1 -load_4bit -quant_type nf4 -dtype float16 -dbl_quant -lr 1e-3 -bf16 -max_len 128 -batch_size 2 -output_dir /llm_summarization/reward_output/ -target_mods q_proj v_proj
```
Align policy model with proximal policy optimization:
```
python rlhf.py -ds_json /llm_summarization/openai_RLHF_data/train_feedback.json -tok_dir /llm_summarization/llama3_hf_format/ -model_save_path /llm_summarization/rlhf_model_1/ -lr 1e-5 -batch_size 1 -mini_batch_size 1 -load_4bit -quant-type nf4 -dtype float16 -dbl_quant -policy_dir /llm_summarization/sft_output/ -reward_dir /llm_summarization/reward_output/
```

# RAG
To retrieve from a database provided by langchain such as WikipediaLoader
```
python rag.py -wiki_query "dogs" -max_docs 100 -temp 0.2 -rep_pen 1.1 -max_new_tok 400 -num_ex 3 -chunk_size 512 -chunk_overlap 30 -llm_path /llm_summarization/llama3_hf_format/ -em_model_name BAAI/bge-base-en-v1.5 -tok_path /llm_summarization/llama3_hf_format/ -load_4bit -quant_type nf4 -dtpye float16 -dbl_quant
```

To retrieve from a custom database of .MD files located within a certain folder (e.g. /llm_summarization/md_files/)
```
python rag.py -custom_ds -md_dir /llm_summarization/md_files/ -max_docs 100 -temp 0.2 -rep_pen 1.1 -max_new_tok 400 -num_ex 3 -chunk_size 512 -chunk_overlap 30 -llm_path /llm_summarization/llama3_hf_format/ -em_model_name BAAI/bge-base-en-v1.5 -tok_path /llm_summarization/llama3_hf_format/ -load_4bit -quant_type nf4 -dtpye float16 -dbl_quant
```
