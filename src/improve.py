"""Improvements over the baseline probe (layer-16 last-token logistic regression).

Purely additive: reads the baseline's data files and reuses its helpers, but never
changes them. Everything here is compared on one protocol, repeated 5-fold CV on
train+val grouped by question, and only the chosen candidate is scored on test.

Extra features come from one more forward pass per example (extract_extra_features):
  - answer_mean / question_last: hidden states at every layer for the other two
    token positions (the baseline uses answer_last_token only)
  - uncertainty: how sure the model was of its own answer tokens (log-probs, entropy)
"""
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm

from . import config
from .io_utils import read_jsonl
from .model_utils import encode_prompt_and_answer

EXTRA_FEATURES_DIR = config.DATA_DIR / "features_extra"   # one .npz per example
UNCERTAINTY_NAMES = ["mean_logprob", "min_logprob", "mean_entropy", "max_entropy", "num_tokens"]
C_GRID = [1e-4, 1e-3, 1e-2]   # the baseline's sweep picked C <= 1e-2 at nearly every layer
CV_FOLDS, CV_REPEATS = 5, 3


# ---------------------------------------------------------------- extraction
@torch.no_grad()
def extract_extra_features(model, tokenizer, question: str, answer: str) -> Dict[str, np.ndarray]:
    input_ids, prompt_len = encode_prompt_and_answer(tokenizer, question, answer)
    if len(input_ids) == prompt_len:
        raise ValueError("Empty answer")
    ids = torch.tensor([input_ids], device=model.device)
    out = model(input_ids=ids, output_hidden_states=True)
    h = torch.stack(out.hidden_states)[:, 0]          # (num_layers + 1, seq_len, hidden_dim)

    # logits at position t predict token t + 1, so these rows score the answer tokens
    logp = torch.log_softmax(out.logits[0, prompt_len - 1:-1].float(), dim=-1)
    tok_logp = logp.gather(1, ids[0, prompt_len:, None])[:, 0]
    entropy = -(logp.exp() * logp).sum(-1)
    uncertainty = [tok_logp.mean(), tok_logp.min(), entropy.mean(), entropy.max(), len(tok_logp)]

    return {
        "answer_mean": h[:, prompt_len:].mean(dim=1).float().cpu().numpy().astype(np.float16),
        "question_last": h[:, prompt_len - 1].float().cpu().numpy().astype(np.float16),
        "uncertainty": np.array([float(u) for u in uncertainty], dtype=np.float32),
    }


def extract_all_extra_features(model, tokenizer, records: List[dict], out_dir=EXTRA_FEATURES_DIR,
                               overwrite=False) -> int:
    """records: rows of probe_dataset_all_layers.jsonl. Returns how many were newly extracted."""
    out_dir.mkdir(parents=True, exist_ok=True)
    n_new = 0
    for r in tqdm(records, desc="Extra features"):
        path = out_dir / f"{r['example_id']}.npz"
        if path.exists() and not overwrite:
            continue
        np.savez(path, **extract_extra_features(model, tokenizer, r["question"], r["generated_answer"]))
        n_new += 1
    return n_new


# ---------------------------------------------------------------- data
@dataclass
class ProbeData:
    answer_last: np.ndarray      # (n, num_layers + 1, hidden_dim), the baseline's features
    answer_mean: np.ndarray
    question_last: np.ndarray
    uncertainty: np.ndarray      # (n, len(UNCERTAINTY_NAMES))
    y: np.ndarray
    question_id: np.ndarray
    split: np.ndarray
    label_margin: np.ndarray     # |max_sim_incorrect - max_sim_correct|; small = borderline label

    @property
    def pool(self) -> np.ndarray:
        return np.where(np.isin(self.split, ["train", "val"]))[0]

    @property
    def test(self) -> np.ndarray:
        return np.where(self.split == "test")[0]


def load_probe_data(records: List[dict], extra_dir=EXTRA_FEATURES_DIR) -> ProbeData:
    labeled = {r["example_id"]: r for r in read_jsonl(config.LABELED_PATH)}
    last, mean, qlast, unc = [], [], [], []
    for r in records:
        last.append(np.load(config.DATA_DIR / r["hidden_state_path"]))
        z = np.load(extra_dir / f"{r['example_id']}.npz")
        mean.append(z["answer_mean"]); qlast.append(z["question_last"]); unc.append(z["uncertainty"])
    lab = [labeled[r["example_id"]] for r in records]
    return ProbeData(
        answer_last=np.stack(last).astype(np.float32),
        answer_mean=np.stack(mean).astype(np.float32),
        question_last=np.stack(qlast).astype(np.float32),
        uncertainty=np.stack(unc),
        y=np.array([r["correctness_label"] for r in records]),
        question_id=np.array([r["question_id"] for r in records]),
        split=np.array([r["dataset_split"] for r in records]),
        label_margin=np.array([abs(l["max_sim_incorrect"] - l["max_sim_correct"]) for l in lab]),
    )


# ---------------------------------------------------------------- models
# A model is fit_predict(train_idx, test_idx) -> P(hallucination) for test_idx,
# with indices into ProbeData. That lets one CV loop handle every kind of candidate.
FitPredict = Callable[[np.ndarray, np.ndarray], np.ndarray]


def layer_average(X: np.ndarray, layers: Sequence[int]) -> np.ndarray:
    """(n, num_layers + 1, hidden_dim) -> (n, hidden_dim): mean hidden state over the given layers."""
    return X[:, list(layers)].mean(axis=1)


def _lr(C, pca=None):
    steps = [StandardScaler()] + ([PCA(pca, random_state=config.SEED)] if pca else [])
    steps.append(LogisticRegression(C=C, max_iter=3000, class_weight="balanced"))
    return make_pipeline(*steps)


def _train_mask(d: ProbeData, idx: np.ndarray, min_margin: float) -> np.ndarray:
    """Drops borderline labels from training only; evaluation always uses every example."""
    return idx[d.label_margin[idx] >= min_margin] if min_margin > 0 else idx


def linear_probe(d: ProbeData, features: Callable[[np.ndarray], np.ndarray], C: float,
                 pca: Optional[int] = None, min_margin: float = 0.0) -> FitPredict:
    def fit_predict(tr, te):
        tr = _train_mask(d, tr, min_margin)
        m = _lr(C, pca).fit(features(tr), d.y[tr])
        return m.predict_proba(features(te))[:, 1]
    return fit_predict


def layer_ensemble(d: ProbeData, rep: str, layers: Sequence[int], C: float,
                   min_margin: float = 0.0) -> FitPredict:
    """One probe per layer, probabilities averaged."""
    X = getattr(d, rep)
    probes = [linear_probe(d, lambda i, L=L: X[i, L], C, min_margin=min_margin) for L in layers]
    return lambda tr, te: np.mean([p(tr, te) for p in probes], axis=0)


def average(*models: FitPredict) -> FitPredict:
    """Averages the probabilities of several models (e.g. layer ensembles over different token positions)."""
    return lambda tr, te: np.mean([m(tr, te) for m in models], axis=0)


def auto_layer(d: ProbeData, rep: str, C: float, layers: Sequence[int] = range(37),
               inner_folds: int = 5) -> FitPredict:
    """Picks the single best layer by inner CV on the training fold only, then fits it.
    Scoring this with cv_auroc gives an honest estimate of 'choose the best layer', free of the
    selection bias of reading the top layer off a sweep."""
    X = getattr(d, rep)
    def fit_predict(tr, te):
        cv = StratifiedGroupKFold(inner_folds, shuffle=True, random_state=config.SEED)
        folds = list(cv.split(tr, d.y[tr], d.question_id[tr]))
        def inner(L):
            oof = np.zeros(len(tr))
            for a, b in folds:
                oof[b] = _lr(C).fit(X[tr[a], L], d.y[tr[a]]).predict_proba(X[tr[b], L])[:, 1]
            return roc_auc_score(d.y[tr], oof)
        L = max(layers, key=inner)
        return _lr(C).fit(X[tr, L], d.y[tr]).predict_proba(X[te, L])[:, 1]
    return fit_predict


def stacked_with_uncertainty(d: ProbeData, base: FitPredict, inner_folds: int = 5) -> FitPredict:
    """Meta logistic regression on [logit(probe score), uncertainty features].
    The probe scores it trains on are out-of-fold within the training set, so it never
    sees a probe's prediction on that probe's own training data."""
    def fit_predict(tr, te):
        oof = np.zeros(len(tr))
        cv = StratifiedGroupKFold(inner_folds, shuffle=True, random_state=config.SEED)
        for a, b in cv.split(tr, d.y[tr], d.question_id[tr]):
            oof[b] = base(tr[a], tr[b])
        logit = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / np.clip(1 - p, 1e-6, 1))
        Z = lambda idx, p: np.column_stack([logit(p), d.uncertainty[idx]])
        meta = _lr(1.0).fit(Z(tr, oof), d.y[tr])
        return meta.predict_proba(Z(te, base(tr, te)))[:, 1]
    return fit_predict


# ---------------------------------------------------------------- evaluation
def cv_auroc(d: ProbeData, model: FitPredict, folds=CV_FOLDS, repeats=CV_REPEATS) -> Dict[str, float]:
    """Repeated grouped k-fold AUROC on train+val. Returns mean and std over repeats."""
    idx = d.pool
    scores = []
    for rep in range(repeats):
        cv = StratifiedGroupKFold(folds, shuffle=True, random_state=config.SEED + rep)
        oof = np.zeros(len(idx))
        for a, b in cv.split(idx, d.y[idx], d.question_id[idx]):
            oof[b] = model(idx[a], idx[b])
        scores.append(roc_auc_score(d.y[idx], oof))
    return {"cv_auroc": float(np.mean(scores)), "cv_std": float(np.std(scores))}


def best_over_C(d: ProbeData, make: Callable[[float], FitPredict], Cs=C_GRID, **cv_kw) -> Dict:
    """CV every C and keep the best. Returns {'C', 'cv_auroc', 'cv_std'}."""
    results = [{"C": C, **cv_auroc(d, make(C), **cv_kw)} for C in Cs]
    return max(results, key=lambda r: r["cv_auroc"])


def test_auroc(d: ProbeData, model: FitPredict, n_boot=2000, seed=config.SEED) -> Dict:
    """Fit on all of train+val, score test once. Also returns the test probabilities and a
    bootstrap 95% interval (resampling test questions)."""
    p = model(d.pool, d.test)
    y = d.y[d.test]
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() != y[i].max():
            boots.append(roc_auc_score(y[i], p[i]))
    return {"test_auroc": float(roc_auc_score(y, p)), "ci95": np.percentile(boots, [2.5, 97.5]).tolist(), "probs": p}


def paired_bootstrap(y: np.ndarray, p_a: np.ndarray, p_b: np.ndarray, n_boot=2000, seed=config.SEED) -> Dict:
    """AUROC(b) - AUROC(a) on the same resampled test questions."""
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() != y[i].max():
            diffs.append(roc_auc_score(y[i], p_b[i]) - roc_auc_score(y[i], p_a[i]))
    diffs = np.array(diffs)
    return {"diff": float(roc_auc_score(y, p_b) - roc_auc_score(y, p_a)),
            "ci95": np.percentile(diffs, [2.5, 97.5]).tolist(), "p_b_better": float((diffs > 0).mean())}
