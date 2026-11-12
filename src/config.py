"""Central configuration for the pipeline. Edit values here, not in the notebooks."""
from pathlib import Path

# ---------------------------------------------------------------- paths
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RAW_CSV = DATA_DIR / "raw_questions.csv"
QUESTIONS_PATH = DATA_DIR / "questions.jsonl"              # step 1
GENERATED_PATH = DATA_DIR / "generated_answers.jsonl"      # step 2 (unlabeled)
LABELED_PATH = DATA_DIR / "labeled_answers.jsonl"          # step 2 (labeled)
HIDDEN_STATES_DIR = DATA_DIR / "hidden_states"             # step 3 (.npy files)
PROBE_DATASET_PATH = DATA_DIR / "probe_dataset.jsonl"      # step 3 (index records)
ALL_LAYERS_DIR = DATA_DIR / "hidden_states_all_layers"     # layer sweep (.npy, one per example)
ALL_LAYERS_DATASET_PATH = DATA_DIR / "probe_dataset_all_layers.jsonl"
MODELS_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"

# ---------------------------------------------------------------- step 1: data
TRUTHFULQA_URL = "https://raw.githubusercontent.com/sylinrl/TruthfulQA/main/TruthfulQA.csv"
NUM_QUESTIONS = 500
SEED = 42

# ---------------------------------------------------------------- step 2: generation
# Any Hugging Face causal LM works. It must have more than LAYER_INDEX layers.
MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
NUM_ANSWERS_PER_QUESTION = 1   # 1 -> greedy decoding; >1 -> sampling (Q001_A001, Q001_A002, ...)
TEMPERATURE = 0.7              # only used when NUM_ANSWERS_PER_QUESTION > 1
MAX_NEW_TOKENS = 64
ANSWER_INSTRUCTION = "Answer the following question truthfully in one short sentence."

# Labeling: 0 = truthful (matches a correct reference), 1 = hallucination
LABEL_METHOD = "reference_answer_evaluation"
JUDGE_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# ---------------------------------------------------------------- step 3: hidden states
# hidden_states[0] is the embedding output, hidden_states[i] is the output of block i.
LAYER_INDEX = 16
# "answer_last_token" | "question_last_token" | "answer_mean"
TOKEN_POSITION_METHOD = "answer_last_token"
# Split is done per question_id so no question leaks across splits.
SPLIT_RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}

# ---------------------------------------------------------------- step 4: probe
PROBE_C_GRID = [1e-5, 1e-4, 1e-3, 1e-2, 0.1, 1.0]   # regularisation strengths tried on val
CV_FOLDS = 5                                      # layer sweep: k-fold CV AUROC on train+val
