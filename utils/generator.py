"""
utils/generator.py
---------------------
Loads an instruct LLM (Qwen2.5-Instruct, Llama-3.1-Instruct, Gemma, or
your own fine-tuned model) and generates the next story chapter from a
built prompt.
"""

import logging
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

logger = logging.getLogger(__name__)

DEFAULT_MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"


class StoryGenerator:
    """Wraps an instruct LLM for next-chapter generation."""

    def __init__(self, model_name: str = DEFAULT_MODEL_NAME, load_in_4bit: bool = True):
        logger.info(f"Loading generator model: {model_name} (4-bit={load_in_4bit})")
        self.model_name = model_name

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        if load_in_4bit and torch.cuda.is_available():
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name, quantization_config=bnb_config, device_map="auto"
            )
        else:
            dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
            self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype)
            if torch.cuda.is_available():
                self.model.to("cuda")

        self.model.eval()

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 1200,
        temperature: float = 0.8,
        top_p: float = 0.9,
        repetition_penalty: float = 1.15,
    ) -> str:
        """
        Generate the next chapter from a fully-built prompt.

        Target length per the project spec is 800-1500 words; max_new_tokens
        is set generously above that in token terms since tokens != words.
        """
        messages = [{"role": "user", "content": prompt}]
        input_text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(input_text, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        generated = self.tokenizer.decode(
            output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        )
        return generated.strip()
