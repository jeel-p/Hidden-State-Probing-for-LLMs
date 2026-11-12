"""Step 3: re-run the model on (prompt + generated answer) and save one
hidden-state vector per example as data/hidden_states/<example_id>.npy."""
import random
from typing import Dict, List

import numpy as np
import torch
from tqdm.auto import tqdm

from . import config
from .model_utils import encode_prompt_and_answer


def assign_splits(question_ids: List[str], ratios=config.SPLIT_RATIOS, seed=config.SEED) -> Dict[str, str]:
    ids = sorted(set(question_ids))
    random.Random(seed).shuffle(ids)
    n_train = int(len(ids) * ratios["train"])
    n_val = int(len(ids) * ratios["val"])
    split = {}
    for i, qid in enumerate(ids):
        split[qid] = "train" if i < n_train else "val" if i < n_train + n_val else "test"
    return split


def _pool(h: torch.Tensor, prompt_len: int, method: str) -> torch.Tensor:
    """h: (..., seq_len, hidden_dim) -> (..., hidden_dim) at the chosen token position."""
    if method == "answer_last_token":
        return h[..., -1, :]
    if method == "question_last_token":
        return h[..., prompt_len - 1, :]
    if method == "answer_mean":
        return h[..., prompt_len:, :].mean(dim=-2)
    raise ValueError(f"Unknown token position method: {method}")


@torch.no_grad()
def _forward_hidden_states(model, tokenizer, question: str, answer: str, method: str):
    input_ids, prompt_len = encode_prompt_and_answer(tokenizer, question, answer)
    if len(input_ids) == prompt_len and method != "question_last_token":
        raise ValueError("Empty answer; cannot take an answer-token representation")
    ids = torch.tensor([input_ids], device=model.device)
    out = model(input_ids=ids, output_hidden_states=True)
    return out.hidden_states, prompt_len


def extract_vector(model, tokenizer, question: str, answer: str,
                   layer_index=config.LAYER_INDEX, method=config.TOKEN_POSITION_METHOD) -> np.ndarray:
    hidden_states, prompt_len = _forward_hidden_states(model, tokenizer, question, answer, method)
    h = hidden_states[layer_index][0]  # (seq_len, hidden_dim)
    return _pool(h, prompt_len, method).float().cpu().numpy()


def extract_all_layer_vectors(model, tokenizer, question: str, answer: str,
                              method=config.TOKEN_POSITION_METHOD) -> np.ndarray:
    """Returns (num_layers + 1, hidden_dim): row 0 is the embedding output, row i is block i."""
    hidden_states, prompt_len = _forward_hidden_states(model, tokenizer, question, answer, method)
    h = torch.stack(hidden_states)[:, 0]  # (num_layers + 1, seq_len, hidden_dim)
    return _pool(h, prompt_len, method).float().cpu().numpy()


def _record(r: dict, question: str, layer_index, method: str, npy_path, split: str) -> dict:
    return {
        "example_id": r["example_id"],
        "question_id": r["question_id"],
        "question": question,
        "generated_answer": r["generated_answer"],
        "correctness_label": r["correctness_label"],
        "label_method": r["label_method"],
        "model_name": r["model_name"],
        "layer_index": layer_index,
        "token_position_method": method,
        "hidden_state_path": str(npy_path.relative_to(config.DATA_DIR)),
        "dataset_split": split,
    }


def extract_hidden_states(model, tokenizer, labeled: List[dict], questions: List[dict],
                          layer_index=config.LAYER_INDEX, method=config.TOKEN_POSITION_METHOD,
                          out_dir=config.HIDDEN_STATES_DIR, overwrite=True) -> List[dict]:
    n_layers = model.config.num_hidden_layers
    if not 0 <= layer_index <= n_layers:
        raise ValueError(f"layer_index {layer_index} out of range (model has {n_layers} layers)")

    out_dir.mkdir(parents=True, exist_ok=True)
    q_by_id = {q["question_id"]: q for q in questions}
    splits = assign_splits([r["question_id"] for r in labeled])
    records = []

    for r in tqdm(labeled, desc=f"Layer {layer_index} / {method}"):
        npy_path = out_dir / f"{r['example_id']}.npy"
        question = q_by_id[r["question_id"]]["question"]
        if overwrite or not npy_path.exists():
            try:
                vec = extract_vector(model, tokenizer, question, r["generated_answer"], layer_index, method)
            except ValueError as e:
                print(f"Skipping {r['example_id']}: {e}")
                continue
            np.save(npy_path, vec)
        records.append(_record(r, question, layer_index, method, npy_path, splits[r["question_id"]]))
    return records


def extract_hidden_states_all_layers(model, tokenizer, labeled: List[dict], questions: List[dict],
                                     method=config.TOKEN_POSITION_METHOD,
                                     out_dir=config.ALL_LAYERS_DIR, overwrite=False) -> List[dict]:
    """Like extract_hidden_states, but each .npy holds every layer: shape (num_layers + 1, hidden_dim).
    Stored as float16 (the model's own precision on GPU/MPS) to halve disk use."""
    out_dir.mkdir(parents=True, exist_ok=True)
    q_by_id = {q["question_id"]: q for q in questions}
    splits = assign_splits([r["question_id"] for r in labeled])
    records = []

    for r in tqdm(labeled, desc=f"All layers / {method}"):
        npy_path = out_dir / f"{r['example_id']}.npy"
        question = q_by_id[r["question_id"]]["question"]
        if overwrite or not npy_path.exists():
            try:
                vecs = extract_all_layer_vectors(model, tokenizer, question, r["generated_answer"], method)
            except ValueError as e:
                print(f"Skipping {r['example_id']}: {e}")
                continue
            np.save(npy_path, vecs.astype(np.float16))
        records.append(_record(r, question, "all", method, npy_path, splits[r["question_id"]]))
    return records
