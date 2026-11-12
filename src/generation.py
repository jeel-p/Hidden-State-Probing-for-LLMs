"""Step 2a: generate answers for each question. Resumable: re-running skips
example_ids already present in the output file."""
from typing import List

import torch
from tqdm.auto import tqdm

from . import config
from .io_utils import append_jsonl, read_jsonl
from .model_utils import build_prompt, encode_prompt


@torch.no_grad()
def generate_for_question(model, tokenizer, question: str, n=config.NUM_ANSWERS_PER_QUESTION) -> List[str]:
    prompt_ids = encode_prompt(tokenizer, build_prompt(tokenizer, question))
    input_ids = torch.tensor([prompt_ids], device=model.device)
    sampling = n > 1
    out = model.generate(
        input_ids=input_ids,
        attention_mask=torch.ones_like(input_ids),
        max_new_tokens=config.MAX_NEW_TOKENS,
        do_sample=sampling,
        temperature=config.TEMPERATURE if sampling else None,
        top_p=None,
        top_k=None,
        num_return_sequences=n,
        pad_token_id=tokenizer.pad_token_id,
    )
    answers = tokenizer.batch_decode(out[:, input_ids.shape[1]:], skip_special_tokens=True)
    return [a.strip() for a in answers]


def generate_answers(model, tokenizer, questions: List[dict], out_path=config.GENERATED_PATH,
                     model_name=config.MODEL_NAME, n=config.NUM_ANSWERS_PER_QUESTION) -> List[dict]:
    done = {r["question_id"] for r in read_jsonl(out_path) if r["model_name"] == model_name}
    todo = [q for q in questions if q["question_id"] not in done]
    print(f"{len(done)} questions already done, {len(todo)} to go")

    for q in tqdm(todo, desc="Generating"):
        for j, answer in enumerate(generate_for_question(model, tokenizer, q["question"], n), start=1):
            append_jsonl(out_path, {
                "example_id": f"{q['question_id']}_A{j:03d}",
                "question_id": q["question_id"],
                "model_name": model_name,
                "generated_answer": answer,
            })
    return [r for r in read_jsonl(out_path) if r["model_name"] == model_name]
