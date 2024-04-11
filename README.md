# llm-summarization
TLDR dataset: 
https://huggingface.co/datasets/webis/tldr-17

RLHF dataset:
from: https://github.com/openai/summarize-from-feedback
```
azcopy copy "https://openaipublic.blob.core.windows.net/summarize-from-feedback/dataset/*" . --recursive
```

potentially important paper: 
Neural Summarization of Electronic Health Records



LLama2 access can be gained by applying at the following link:

https://llama.meta.com/llama-downloads/

```
git clone https://github.com/meta-llama/llama-recipes
```

```
cd ./llama-recipes
pip install -e .
```

download.sh downloads a folder /llama-2-7b/ containing consolidated.00.pth.tar, params.JSON, tokenizer.JSON, tokenizer.model, tokenizer_config.JSON

```
git clone https://github.com/huggingface/transformers
```

use transformers/src/transformers/models/llama/convert_llama_weights_to_hf.py to convert llama to huggingface format in order to use llama-recipes

with config.JSON, pytorch_model-000001-of-000003.bin, etc in /llama-2-7b/ move this folder to C:\personal_ML\llm-ehr-summarization\llama-recipes\recipes\inference\local_inference\ in order to use with inference

in order to summarize text within example_prompt1.txt (quantization for 8-bit precision)
```
python inference.py --model_name llama-2-7b --prompt_file C:\personal_ML\llm_summarization\example_prompt1.txt --quantization
```

line 85 of /llama/llama/generation.py torch.distributed.init_process_group("nccl") -> torch.distributed.init_process_group("gloo") since nccl is not available on Windows

Random samples were taken from the TL;DR dataset and generated with the llama 2-7b prior to any finetuning:

```
python prefinetune_examples.py -save_path C:\\llm_summarization\\example_prompts\\ -num_ex 2
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
There are reasons for this timekeeping system, and there are reasons to change it.
The argument made does not convince me to change it now.
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
There are a lot of street artists and street art is hard to categorize. Some of the
art made by street artists is better than other works, but it ultimately depends on
the opinions of the viewer. One of the street artists this is written about keeps
using the same images so often, and it gets a little tiresome.
```



```
python -m llama_recipes.finetuning --dataset "custom_dataset" --custom_dataset.file "examples/custom_dataset.py:custom_dataset" [TRAINING PARAMETERS]
```





