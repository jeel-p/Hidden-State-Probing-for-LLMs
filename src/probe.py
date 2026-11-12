"""Step 4: logistic-regression probes on saved hidden states, for one layer or every layer."""
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm

from . import config

SPLITS = ("train", "val", "test")


def load_splits(records: List[dict]) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """{split: (X, y)}. X is (n, hidden_dim) for single-layer files, (n, num_layers + 1, hidden_dim) for all-layer files."""
    out = {}
    for name in SPLITS:
        rs = [r for r in records if r["dataset_split"] == name]
        X = np.stack([np.load(config.DATA_DIR / r["hidden_state_path"]) for r in rs]).astype(np.float32)
        y = np.array([r["correctness_label"] for r in rs])
        out[name] = (X, y)
    return out


def evaluate(probe, X: np.ndarray, y: np.ndarray) -> dict:
    p = probe.predict_proba(X)[:, 1]
    return {"auroc": roc_auc_score(y, p), "accuracy": accuracy_score(y, p > 0.5), "f1": f1_score(y, p > 0.5)}


def make_probe(C: float):
    return make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=2000, class_weight="balanced"))


def cv_auroc(X: np.ndarray, y: np.ndarray, groups=None, Cs=config.PROBE_C_GRID,
             folds=config.CV_FOLDS, seed=config.SEED) -> Tuple[float, float]:
    """Mean k-fold AUROC for the best C. Returns (auroc, C). Pass groups (question ids) so
    answers to one question never sit on both sides of a fold."""
    if groups is None:
        cv = StratifiedKFold(folds, shuffle=True, random_state=seed)
    else:
        cv = StratifiedGroupKFold(folds, shuffle=True, random_state=seed)
    return max((cross_val_score(make_probe(C), X, y, groups=groups, cv=cv, scoring="roc_auc").mean(), C)
               for C in Cs)


def train_probe(splits: Dict[str, Tuple[np.ndarray, np.ndarray]], Cs=config.PROBE_C_GRID, verbose=False):
    """Fits one probe per C on train, keeps the best by val AUROC. Returns (probe, C, metrics per split)."""
    X_tr, y_tr = splits["train"]
    X_va, y_va = splits["val"]
    best = None
    for C in Cs:
        probe = make_probe(C)
        probe.fit(X_tr, y_tr)
        auc = roc_auc_score(y_va, probe.predict_proba(X_va)[:, 1])
        if verbose:
            print(f"C={C}: val AUROC={auc:.3f}")
        if best is None or auc > best[0]:
            best = (auc, C, probe)
    _, best_C, probe = best
    return probe, best_C, {s: evaluate(probe, X, y) for s, (X, y) in splits.items()}


def layer_sweep(all_layer_splits: Dict[str, Tuple[np.ndarray, np.ndarray]], layers=None,
                Cs=config.PROBE_C_GRID, cv_groups=None) -> Tuple[pd.DataFrame, dict]:
    """Trains an independent probe on every layer. Returns (one row per layer, {layer: probe}).

    Each row has the val-selected probe's train/val/test metrics, plus cv_auroc: k-fold AUROC on
    train+val pooled (C picked by CV). cv_groups are the question ids of train then val, in order."""
    n_layers = all_layer_splits["train"][0].shape[1]
    layers = range(n_layers) if layers is None else layers
    X_pool = np.concatenate([all_layer_splits["train"][0], all_layer_splits["val"][0]])
    y_pool = np.concatenate([all_layer_splits["train"][1], all_layer_splits["val"][1]])
    rows, probes = [], {}
    for layer in tqdm(layers, desc="Layer sweep"):
        splits = {s: (X[:, layer], y) for s, (X, y) in all_layer_splits.items()}
        probe, C, metrics = train_probe(splits, Cs)
        probes[layer] = probe
        row = {"layer_index": layer, "C": C}
        for s, m in metrics.items():
            row.update({f"{s}_{k}": v for k, v in m.items()})
        row["cv_auroc"], row["cv_C"] = cv_auroc(X_pool[:, layer], y_pool, cv_groups, Cs)
        rows.append(row)
    return pd.DataFrame(rows), probes
