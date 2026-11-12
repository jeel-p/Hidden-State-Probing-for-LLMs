"""Dataset v2 (additive): all 790 TruthfulQA questions, several sampled answers each, graded by Claude.

The original 500-question data (data/*.jsonl, data/hidden_states*/) is untouched; everything here
lives under data/v2/. Question ids are T001-T790 (all of TruthfulQA, in its original order), so
they can't be confused with the Q001-Q500 ids of the first sample.

    python -m src.expand questions
    python -m src.expand generate          # resumable; ~1 h on an M-series Mac
    python -m src.expand extract           # resumable; hidden states + uncertainty, one pass per answer
    python -m src.expand grade --sync --limit 5
    python -m src.expand grade             # the rest via the Message Batches API
"""
import argparse
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from tqdm.auto import tqdm

from . import config
from .data_prep import build_questions
from .hidden_states import assign_splits
from .improve import ProbeData
from .io_utils import read_jsonl, write_jsonl
from .model_utils import encode_prompt_and_answer
from .relabel import CONFIDENCE_RANK

V2_DIR = config.DATA_DIR / "v2"
QUESTIONS_PATH = V2_DIR / "questions.jsonl"
GENERATED_PATH = V2_DIR / "generated_answers.jsonl"
FEATURES_DIR = V2_DIR / "features"                # one .npz per answer (gitignored, ~1.2 GB)
DATASET_PATH = V2_DIR / "probe_dataset.jsonl"     # index: answer, question, split, feature path
CLAUDE_LABELS_PATH = V2_DIR / "labeled_answers_claude.jsonl"
N_SAMPLES = 5                                     # sampled at config.TEMPERATURE


def build_v2_questions() -> List[dict]:
    rows = build_questions(n=790)                 # n == all rows: every question, original order
    for q in rows:
        q["question_id"] = "T" + q["question_id"][1:]
    write_jsonl(QUESTIONS_PATH, rows)
    print(f"Wrote {len(rows)} questions to {QUESTIONS_PATH}")
    return rows


def generate(model, tokenizer, questions: List[dict], n: int = N_SAMPLES) -> List[dict]:
    from .generation import generate_answers
    torch.manual_seed(config.SEED)                # sampling; resumed runs continue from a new state
    return generate_answers(model, tokenizer, questions, out_path=GENERATED_PATH, n=n)


@torch.no_grad()
def _features(model, tokenizer, question: str, answer: str) -> Dict[str, np.ndarray]:
    """Everything notebooks 05-09 use, from one forward pass: last-token and mean-token hidden
    states at every layer, plus the 5 uncertainty features of src/improve.py."""
    input_ids, prompt_len = encode_prompt_and_answer(tokenizer, question, answer)
    if len(input_ids) == prompt_len:
        raise ValueError("Empty answer")
    ids = torch.tensor([input_ids], device=model.device)
    out = model(input_ids=ids, output_hidden_states=True)
    h = torch.stack(out.hidden_states)[:, 0]      # (num_layers + 1, seq_len, hidden_dim)
    logp = torch.log_softmax(out.logits[0, prompt_len - 1:-1].float(), dim=-1)
    tok_logp = logp.gather(1, ids[0, prompt_len:, None])[:, 0]
    entropy = -(logp.exp() * logp).sum(-1)
    unc = [tok_logp.mean(), tok_logp.min(), entropy.mean(), entropy.max(), len(tok_logp)]
    return {"answer_last": h[:, -1].float().cpu().numpy().astype(np.float16),
            "answer_mean": h[:, prompt_len:].mean(1).float().cpu().numpy().astype(np.float16),
            "uncertainty": np.array([float(u) for u in unc], dtype=np.float32)}


def extract(model, tokenizer) -> List[dict]:
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    q_by_id = {q["question_id"]: q for q in read_jsonl(QUESTIONS_PATH)}
    answers = read_jsonl(GENERATED_PATH)
    splits = assign_splits([a["question_id"] for a in answers])    # per question, as in step 3
    records = []
    for a in tqdm(answers, desc="Features"):
        path = FEATURES_DIR / f"{a['example_id']}.npz"
        question = q_by_id[a["question_id"]]["question"]
        if not path.exists():
            try:
                np.savez(path, **_features(model, tokenizer, question, a["generated_answer"]))
            except ValueError as e:
                print(f"Skipping {a['example_id']}: {e}")
                continue
        records.append({"example_id": a["example_id"], "question_id": a["question_id"], "question": question,
                        "generated_answer": a["generated_answer"], "model_name": a["model_name"],
                        "feature_path": str(path.relative_to(config.DATA_DIR)),
                        "dataset_split": splits[a["question_id"]]})
    write_jsonl(DATASET_PATH, records)
    return records


def grade(sync: bool = False, limit: Optional[int] = None, batch_id: Optional[str] = None) -> None:
    import anthropic
    from . import llm_judge
    llm_judge._load_dotenv()
    client = anthropic.Anthropic()
    if batch_id:
        return llm_judge.collect_batch(client, batch_id, out_path=CLAUDE_LABELS_PATH)
    q_by_id = {q["question_id"]: q for q in read_jsonl(QUESTIONS_PATH)}
    todo = llm_judge.pending(read_jsonl(DATASET_PATH), out_path=CLAUDE_LABELS_PATH)[:limit]
    print(f"{len(todo)} answers to grade with {llm_judge.JUDGE_MODEL}")
    if not todo:
        return
    if sync:
        llm_judge.run_sync(client, todo, q_by_id, out_path=CLAUDE_LABELS_PATH)
    else:
        llm_judge.collect_batch(client, llm_judge.submit_batch(client, todo, q_by_id), out_path=CLAUDE_LABELS_PATH)


def load_v2() -> Tuple[ProbeData, np.ndarray, List[dict]]:
    """ProbeData for every graded answer (y = Claude's label) plus Claude's confidence rank.
    Hidden states stay float16 to keep ~3,950 x 37 x 2048 x 2 arrays in memory; cast the slice
    you use. question_last isn't extracted for v2 (it was the weakest position), so it is empty."""
    claude = {r["example_id"]: r for r in read_jsonl(CLAUDE_LABELS_PATH) if r.get("claude_label") is not None}
    records = [r for r in read_jsonl(DATASET_PATH) if r["example_id"] in claude]
    n = len(records)
    last = np.empty((n, 37, 2048), dtype=np.float16)
    mean = np.empty_like(last)
    unc = np.empty((n, 5), dtype=np.float32)
    for i, r in enumerate(tqdm(records, desc="Loading v2 features")):
        z = np.load(config.DATA_DIR / r["feature_path"])
        last[i], mean[i], unc[i] = z["answer_last"], z["answer_mean"], z["uncertainty"]
    d = ProbeData(answer_last=last, answer_mean=mean, question_last=np.empty((n, 0, 0), dtype=np.float16),
                  uncertainty=unc, y=np.array([claude[r["example_id"]]["claude_label"] for r in records]),
                  question_id=np.array([r["question_id"] for r in records]),
                  split=np.array([r["dataset_split"] for r in records]),
                  label_margin=np.zeros(n))
    conf = np.array([CONFIDENCE_RANK[claude[r["example_id"]]["claude_confidence"]] for r in records])
    return d, conf, records


def main(argv: Optional[List[str]] = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=["questions", "generate", "extract", "grade"])
    ap.add_argument("--sync", action="store_true")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--batch-id")
    args = ap.parse_args(argv)
    if args.step == "questions":
        return build_v2_questions()
    if args.step == "grade":
        return grade(args.sync, args.limit, args.batch_id)
    from .model_utils import load_model
    model, tokenizer = load_model(config.MODEL_NAME)
    if args.step == "generate":
        generate(model, tokenizer, read_jsonl(QUESTIONS_PATH)[: args.limit])
    else:
        extract(model, tokenizer)


if __name__ == "__main__":
    main()
