# llm-ehr-summarization
Electronic health record summarization with LLama and MIMIC-III dataset


https://mimic.mit.edu/docs/iii/



potentially important paper: 
Neural Summarization of Electronic Health Records



LLaMa2 access can be gained by applying at the following link:

https://llama.meta.com/llama-downloads/

```
git clone https://github.com/meta-llama/llama-recipes
```

download.sh downloads a folder /llama-2-7b/ containing consolidated.00.pth.tar, params.JSON, tokenizer.JSON, tokenizer.model, tokenizer_config.JSON

```
git clone https://github.com/huggingface/transformers
```

use transformers/src/transformers/models/llama/convert_llama_weights_to_hf.py to convert llama to huggingface format in order to use llama-recipes

with config.JSON, pytorch_model-000001-of-000003.bin, etc in /llama-2-7b/ move this folder to C:\personal_ML\llm-ehr-summarization\llama-recipes\recipes\inference\local_inference\ in order to use with inference

in order to summarize text within example_prompt_1.txt (quantization for 8-bit precision)
```
python inference.py --model_name llama-2-7b --prompt_file C:\personal_ML\llm-ehr-summarization\example_prompt_1.txt --quantization
```

