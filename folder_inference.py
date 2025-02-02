# adapted from llama-recipes/recipes/inference/local_inference/inference.py

from pathlib import Path
import fire
import os
import sys
import time
import gradio as gr

import torch
from transformers import AutoTokenizer

from llama_recipes.inference.model_utils import load_model, load_peft_model

from accelerate.utils import is_xpu_available

from tqdm import tqdm

def main(
    model_name,
    save_path,
    peft_model: str=None,
    quantization: bool=False,
    max_new_tokens =100, #The maximum numbers of tokens to generate
    prompt_dir: str=None,
    seed: int=42, #seed value for reproducibility
    do_sample: bool=True, #Whether or not to use sampling ; use greedy decoding otherwise.
    min_length: int=None, #The minimum length of the sequence to be generated, input prompt + min_new_tokens
    use_cache: bool=True,  #[optional] Whether or not the model should use the past last key/values attentions Whether or not the model should use the past last key/values attentions (if applicable to the model) to speed up decoding.
    top_p: float=1.0, # [optional] If set to float < 1, only the smallest set of most probable tokens with probabilities that add up to top_p or higher are kept for generation.
    temperature: float=1.0, # [optional] The value used to modulate the next token probabilities.
    top_k: int=50, # [optional] The number of highest probability vocabulary tokens to keep for top-k-filtering.
    repetition_penalty: float=1.0, #The parameter for repetition penalty. 1.0 means no penalty.
    length_penalty: int=1, #[optional] Exponential penalty to the length that is used with beam-based generation.
    max_padding_length: int=None, # the max padding length to be used with tokenizer padding the prompts.
    use_fast_kernels: bool = False, # Enable using SDPA from PyTroch Accelerated Transformers, make use Flash Attention and Xformer memory-efficient kernels
    **kwargs
):

  def inference(user_prompt, temperature, top_p, top_k, max_new_tokens, model, **kwargs,):
    # Set the seeds for reproducibility
    if is_xpu_available():
        torch.xpu.manual_seed(seed)
    else:
        torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)


    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    batch = tokenizer(user_prompt, padding='max_length', truncation=True, max_length=max_padding_length, return_tensors="pt")
    if is_xpu_available():
        batch = {k: v.to("xpu") for k, v in batch.items()}
    else:
        batch = {k: v.to("cuda") for k, v in batch.items()}

    start = time.perf_counter()
    with torch.no_grad():
        outputs = model.generate(
            **batch,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            top_p=top_p,
            temperature=temperature,
            min_length=min_length,
            use_cache=use_cache,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            length_penalty=length_penalty,
            **kwargs
        )
    e2e_inference_time = (time.perf_counter()-start)*1000
    print(f"the inference time is {e2e_inference_time} ms")
    output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return output_text

  if prompt_dir is not None:
      assert os.path.exists(
          prompt_dir
      ), f"Provided Prompt directory does not exist {prompt_dir}"
      prompt_path = Path(prompt_dir)
      model = load_model(model_name, quantization, use_fast_kernels)
      if peft_model:
        model = load_peft_model(model, peft_model)

      model.eval()
      for prompt_file in tqdm(prompt_path.iterdir()):
        with open(str(prompt_file), "r") as f:
            user_prompt = "\n".join(f.readlines())
        output_text = inference(user_prompt, temperature, top_p, top_k, max_new_tokens, model)
        model_output = output_text.replace(user_prompt, "")
        with open(Path(save_path).joinpath(f"{prompt_file.name}_model_output.txt"), mode="w") as opened_txt:
            opened_txt.write(model_output)
  elif not sys.stdin.isatty():
      user_prompt = "\n".join(sys.stdin.readlines())
      inference(user_prompt, temperature, top_p, top_k, max_new_tokens)
  else:
      gr.Interface(
        fn=inference,
        inputs=[
            gr.components.Textbox(
                lines=9,
                label="User Prompt",
                placeholder="none",
            ),
            gr.components.Slider(
                minimum=0, maximum=1, value=1.0, label="Temperature"
            ),
            gr.components.Slider(
                minimum=0, maximum=1, value=1.0, label="Top p"
            ),
            gr.components.Slider(
                minimum=0, maximum=100, step=1, value=50, label="Top k"
            ),
            gr.components.Slider(
                minimum=1, maximum=2000, step=1, value=200, label="Max tokens"
            ),
        ],
        outputs=[
            gr.components.Textbox(
                lines=5,
                label="Output",
            )
        ],
        title="Meta Llama3 Playground",
        description="https://github.com/facebookresearch/llama-recipes",
      ).queue().launch(server_name="0.0.0.0", share=True)

if __name__ == "__main__":
    fire.Fire(main)
