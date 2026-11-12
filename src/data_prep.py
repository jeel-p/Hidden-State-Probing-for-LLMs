"""Step 1: download TruthfulQA and convert a sample of it into questions.jsonl."""
import csv
import random
import urllib.request
from pathlib import Path
from typing import List

from . import config
from .io_utils import write_jsonl


def download_truthfulqa(dest=config.RAW_CSV, url=config.TRUTHFULQA_URL, force=False) -> Path:
    dest = Path(dest)
    if dest.exists() and not force:
        print(f"Using existing {dest}")
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, dest)
    print(f"Downloaded TruthfulQA to {dest}")
    return dest


def _split_answers(cell: str) -> List[str]:
    return [a.strip() for a in (cell or "").split(";") if a.strip()]


def build_questions(raw_csv=config.RAW_CSV, n=config.NUM_QUESTIONS, seed=config.SEED) -> List[dict]:
    """Randomly sample `n` questions (seeded), kept in original TruthfulQA order."""
    with Path(raw_csv).open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if n > len(rows):
        raise ValueError(f"Requested {n} questions but TruthfulQA has only {len(rows)}")

    picked = sorted(random.Random(seed).sample(range(len(rows)), n))
    questions = []
    for i, row_idx in enumerate(picked, start=1):
        row = rows[row_idx]
        questions.append({
            "question_id": f"Q{i:03d}",
            "category": row["Category"].strip(),
            "question": row["Question"].strip(),
            "best_answer": row["Best Answer"].strip(),
            "correct_answers": _split_answers(row["Correct Answers"]),
            "incorrect_answers": _split_answers(row["Incorrect Answers"]),
            "source": row["Source"].strip(),
        })
    return questions


def prepare(n=config.NUM_QUESTIONS, seed=config.SEED, out=config.QUESTIONS_PATH) -> List[dict]:
    download_truthfulqa()
    questions = build_questions(n=n, seed=seed)
    write_jsonl(out, questions)
    print(f"Wrote {len(questions)} questions to {out}")
    return questions
