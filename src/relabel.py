"""Retraining on Claude's labels (step 2c, src/llm_judge.py), plus a deeper neural probe.

Additive: the baseline labels, data files and earlier modules are unchanged. The same hidden
states are reused; only the target y is swapped for Claude's label.

DualPositionNet is the new neural network. It sees two token positions (the last answer token
and the mean over the answer's tokens) at every late layer, learns a softmax weight per layer
for each position, and feeds both weighted states through an MLP head.
"""
import dataclasses
from typing import Dict, Sequence, Tuple

import numpy as np
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score

from . import config
from .improve import FitPredict, ProbeData
from .io_utils import read_jsonl
from .llm_judge import CLAUDE_LABELS_PATH
from .model_compare import _TorchClassifier

CONFIDENCE_RANK = {"low": 0, "medium": 1, "high": 2}


def with_claude_labels(d: ProbeData, path=CLAUDE_LABELS_PATH) -> Tuple[ProbeData, np.ndarray, np.ndarray]:
    """Returns (data restricted to answers Claude labelled, with y = Claude's label,
    confidence rank per example, the old embedding-similarity label per example).
    d must be loaded from probe_dataset_all_layers.jsonl, whose order this relies on."""
    claude = {r["example_id"]: r for r in read_jsonl(path) if r.get("claude_label") is not None}
    ids = [r["example_id"] for r in read_jsonl(config.ALL_LAYERS_DATASET_PATH)]
    keep = np.array([i in claude for i in ids])
    sub = {f.name: getattr(d, f.name)[keep] for f in dataclasses.fields(d)}
    old_y = sub["y"].copy()
    kept_ids = [i for i, k in zip(ids, keep) if k]
    sub["y"] = np.array([claude[i]["claude_label"] for i in kept_ids])
    conf = np.array([CONFIDENCE_RANK[claude[i]["claude_confidence"]] for i in kept_ids])
    return ProbeData(**sub), conf, old_y


def confident_only(d: ProbeData, conf: np.ndarray, min_rank: int, model_factory) -> FitPredict:
    """Wraps a fit_predict factory so training drops labels below a confidence rank; scoring keeps all."""
    def fit_predict(tr, te):
        return model_factory(tr[conf[tr] >= min_rank], te)
    return fit_predict


# ---------------------------------------------------------------- the new network
class _DualPosition(torch.nn.Module):
    def __init__(self, n_pos, n_layers, dim, hidden, dropout):
        super().__init__()
        self.layer_logits = torch.nn.Parameter(torch.zeros(n_pos, n_layers))   # starts as plain averages
        self.drop_in = torch.nn.Dropout(dropout)
        self.head = torch.nn.Sequential(
            torch.nn.Linear(n_pos * dim, hidden), torch.nn.GELU(), torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden, 1))

    def forward(self, x):                                   # x: (n, positions, layers, dim)
        w = torch.softmax(self.layer_logits, dim=-1)        # one weighting per position
        pooled = (w[None, :, :, None] * x).sum(2)           # (n, positions, dim)
        return self.head(self.drop_in(pooled.flatten(1)))


class DualPositionNet(_TorchClassifier):
    """Input (n, positions, layers, hidden_dim). Layer weights get their own faster learning rate."""

    def __init__(self, hidden=128, layer_lr=0.05, **kw):
        super().__init__(**kw)
        self.hidden, self.layer_lr = hidden, layer_lr

    def _net(self, X):
        return _DualPosition(X.shape[1], X.shape[2], X.shape[3], self.hidden, self.dropout)

    def _param_groups(self, net):
        return [{"params": [net.layer_logits], "lr": self.layer_lr, "weight_decay": 0.0},
                {"params": net.head.parameters()}]

    def layer_weights(self) -> np.ndarray:
        return np.mean([torch.softmax(n.layer_logits, -1).detach().numpy() for n in self.nets_], axis=0)


# ---------------------------------------------------------------- evaluation on fixed splits
def split_metrics(d: ProbeData, model: FitPredict, train_splits: Sequence[str], eval_split: str) -> Dict:
    """Fit on the given splits, score one split: accuracy at p > 0.5, balanced accuracy, F1, AUROC."""
    tr = np.where(np.isin(d.split, list(train_splits)))[0]
    te = np.where(d.split == eval_split)[0]
    p, y = model(tr, te), d.y[te]
    pred = p > 0.5
    return {"n": int(len(te)), "accuracy": accuracy_score(y, pred), "balanced_accuracy": balanced_accuracy_score(y, pred),
            "f1": f1_score(y, pred), "auroc": roc_auc_score(y, p), "probs": p}


def evaluate_everywhere(d: ProbeData, model: FitPredict, repeats: int = 3) -> Dict:
    """One model, three views: grouped 5-fold CV on train+val (repeated), val (fit on train),
    and test (fit on train+val). AUROC and accuracy at p > 0.5 for each; test probabilities too."""
    from sklearn.model_selection import StratifiedGroupKFold
    idx, aucs, accs = d.pool, [], []
    for r in range(repeats):
        oof = np.zeros(len(idx))
        cv = StratifiedGroupKFold(5, shuffle=True, random_state=config.SEED + r)
        for a, b in cv.split(idx, d.y[idx], d.question_id[idx]):
            oof[b] = model(idx[a], idx[b])
        aucs.append(roc_auc_score(d.y[idx], oof)); accs.append(accuracy_score(d.y[idx], oof > 0.5))
    v = split_metrics(d, model, ["train"], "val")
    t = split_metrics(d, model, ["train", "val"], "test")
    return {"cv_auroc": float(np.mean(aucs)), "cv_auroc_std": float(np.std(aucs)), "cv_accuracy": float(np.mean(accs)),
            "val_accuracy": v["accuracy"], "val_auroc": v["auroc"],
            "test_accuracy": t["accuracy"], "test_balanced_accuracy": t["balanced_accuracy"],
            "test_f1": t["f1"], "test_auroc": t["auroc"], "test_probs": t["probs"]}
