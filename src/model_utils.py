"""Shared model loading + prompt building, so generation and hidden-state
extraction see exactly the same token sequence."""
from typing import List, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from . import config


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_model(model_name=config.MODEL_NAME, device=None):
    device = device or get_device()
    dtype = torch.float32 if device == "cpu" else torch.float16
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype)
    model.to(device).eval()
    print(f"Loaded {model_name} on {device} ({model.config.num_hidden_layers} layers)")
    return model, tokenizer


def build_prompt(tokenizer, question: str) -> str:
    content = f"{config.ANSWER_INSTRUCTION}\n\nQuestion: {question}"
    if tokenizer.chat_template:
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": content}], tokenize=False, add_generation_prompt=True
        )
    return f"{content}\nAnswer:"


def encode_prompt(tokenizer, prompt: str) -> List[int]:
    # Chat templates already insert BOS/special tokens themselves.
    return tokenizer(prompt, add_special_tokens=not tokenizer.chat_template)["input_ids"]


def encode_prompt_and_answer(tokenizer, question: str, answer: str) -> Tuple[List[int], int]:
    """Returns (input_ids, prompt_len). Answer tokens are input_ids[prompt_len:]."""
    prompt_ids = encode_prompt(tokenizer, build_prompt(tokenizer, question))
    answer_ids = tokenizer(answer, add_special_tokens=False)["input_ids"]
    return prompt_ids + answer_ids, len(prompt_ids)
