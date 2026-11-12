"""Compares classifier families on the improved probe's setup, with logistic regression as the baseline.

Additive: builds on src/improve.py (same data, same CV protocol, same clear-cut-label option)
without changing it. Every model gets a small hyper-parameter grid picked by CV.

Classic models take the layer-averaged hidden state, shape (n, hidden_dim). The neural
LayerWeightedNet takes the stacked late layers, shape (n, num_layers, hidden_dim), and learns
how much to trust each layer instead of averaging them equally.
"""
from typing import Callable, Dict, List, Sequence

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from . import config
from .improve import FitPredict, ProbeData, _train_mask


def _scores(model, X) -> np.ndarray:
    """Any score that ranks examples works for AUROC: probabilities if available, else margins."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    return model.decision_function(X)


def sk_model(d: ProbeData, features: Callable[[np.ndarray], np.ndarray], make: Callable[[], object],
             min_margin: float = 0.0) -> FitPredict:
    def fit_predict(tr, te):
        tr = _train_mask(d, tr, min_margin)
        return _scores(make().fit(features(tr), d.y[tr]), features(te))
    return fit_predict


# ---------------------------------------------------------------- neural networks
class _TorchClassifier:
    """Full-batch training with AdamW and a class-balanced loss. Small data, so no early
    stopping: a fixed number of epochs plus dropout and weight decay keep it in check.
    Several seeds are trained and averaged, because one small net on ~250 examples is noisy."""

    def __init__(self, epochs=200, lr=1e-3, weight_decay=1e-2, dropout=0.3, seeds=3):
        self.epochs, self.lr, self.weight_decay, self.dropout, self.seeds = epochs, lr, weight_decay, dropout, seeds

    def _net(self, X: torch.Tensor) -> torch.nn.Module:
        raise NotImplementedError

    def _param_groups(self, net: torch.nn.Module) -> List[dict]:
        return [{"params": net.parameters()}]

    def _standardize(self, X: np.ndarray, fit: bool) -> torch.Tensor:
        if fit:
            self.mu_, self.sd_ = X.mean(0, keepdims=True), X.std(0, keepdims=True) + 1e-6
        return torch.tensor((X - self.mu_) / self.sd_, dtype=torch.float32)

    def fit(self, X: np.ndarray, y: np.ndarray):
        dev = "cuda" if torch.cuda.is_available() else "cpu"   # CPU elsewhere; CUDA only speeds this up
        Xt, yt = self._standardize(X, fit=True).to(dev), torch.tensor(y, dtype=torch.float32, device=dev)
        pos_weight = torch.tensor((1 - y.mean()) / max(y.mean(), 1e-6), device=dev)
        loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        self.nets_ = []
        for seed in range(self.seeds):
            torch.manual_seed(config.SEED + seed)
            net = self._net(Xt).to(dev)
            opt = torch.optim.AdamW(self._param_groups(net), lr=self.lr, weight_decay=self.weight_decay)
            net.train()
            for _ in range(self.epochs):
                opt.zero_grad()
                loss_fn(net(Xt).squeeze(-1), yt).backward()
                opt.step()
            self.nets_.append(net.eval().cpu())
        return self

    @torch.no_grad()
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        Xt = self._standardize(X, fit=False)
        p = np.mean([torch.sigmoid(net(Xt).squeeze(-1)).numpy() for net in self.nets_], axis=0)
        return np.column_stack([1 - p, p])


class MLP(_TorchClassifier):
    """hidden_dim -> hidden -> 1, with dropout on the input and the hidden layer."""

    def __init__(self, hidden=256, **kw):
        super().__init__(**kw)
        self.hidden = hidden

    def _net(self, X):
        return torch.nn.Sequential(
            torch.nn.Dropout(self.dropout), torch.nn.Linear(X.shape[-1], self.hidden), torch.nn.GELU(),
            torch.nn.Dropout(self.dropout), torch.nn.Linear(self.hidden, 1))


class _LayerWeighted(torch.nn.Module):
    def __init__(self, n_layers, dim, dropout, hidden):
        super().__init__()
        self.layer_logits = torch.nn.Parameter(torch.zeros(n_layers))   # starts as a plain average
        self.drop = torch.nn.Dropout(dropout)
        self.head = (torch.nn.Linear(dim, 1) if not hidden else
                     torch.nn.Sequential(torch.nn.Linear(dim, hidden), torch.nn.GELU(),
                                         torch.nn.Dropout(dropout), torch.nn.Linear(hidden, 1)))

    def forward(self, x):                       # x: (n, layers, dim)
        w = torch.softmax(self.layer_logits, 0)
        return self.head(self.drop((w[None, :, None] * x).sum(1)))


class LayerWeightedNet(_TorchClassifier):
    """Learns a weighted average over layers (softmax weights), then a linear or 1-hidden-layer head.
    With equal weights and a linear head this is exactly the layer-averaged logistic regression."""

    def __init__(self, hidden=0, layer_lr=0.05, **kw):
        super().__init__(**kw)
        self.hidden, self.layer_lr = hidden, layer_lr

    def _net(self, X):
        return _LayerWeighted(X.shape[1], X.shape[2], self.dropout, self.hidden)

    def _param_groups(self, net):
        # the 19 layer weights get their own faster learning rate and no decay; at the head's
        # learning rate they barely move from a plain average
        return [{"params": [net.layer_logits], "lr": self.layer_lr, "weight_decay": 0.0},
                {"params": net.head.parameters()}]

    def layer_weights(self) -> np.ndarray:
        return np.mean([torch.softmax(n.layer_logits, 0).detach().numpy() for n in self.nets_], axis=0)


# ---------------------------------------------------------------- the model zoo
def model_zoo(seed=config.SEED) -> Dict[str, Dict]:
    """{name: {'family', 'input', 'grid': {label: factory}}}. input is 'avg' (layer-averaged,
    (n, dim)) or 'stack' (late layers stacked, (n, layers, dim))."""
    lr = lambda C, l1=False: make_pipeline(StandardScaler(), LogisticRegression(
        C=C, l1_ratio=1.0 if l1 else 0.0, solver="liblinear" if l1 else "lbfgs", max_iter=3000, class_weight="balanced"))
    return {
        "Logistic regression (baseline)": {"family": "linear", "input": "avg",
            "grid": {f"C={C}": (lambda C=C: lr(C)) for C in (1e-4, 1e-3, 1e-2)}},
        "Logistic regression, L1 (sparse)": {"family": "linear", "input": "avg",
            "grid": {f"C={C}": (lambda C=C: lr(C, l1=True)) for C in (0.1, 0.3, 1.0)}},
        "Linear SVM": {"family": "linear", "input": "avg",
            "grid": {f"C={C}": (lambda C=C: make_pipeline(StandardScaler(), SVC(kernel="linear", C=C, class_weight="balanced")))
                     for C in (1e-5, 1e-4, 1e-3)}},
        "RBF-kernel SVM": {"family": "kernel", "input": "avg",
            "grid": {f"C={C}": (lambda C=C: make_pipeline(StandardScaler(), SVC(kernel="rbf", C=C, gamma="scale", class_weight="balanced")))
                     for C in (0.3, 1.0, 3.0)}},
        "k-nearest neighbours": {"family": "instance", "input": "avg",
            "grid": {f"k={k}": (lambda k=k: make_pipeline(StandardScaler(), KNeighborsClassifier(k, metric="cosine", weights="distance")))
                     for k in (15, 31, 61)}},
        "Gaussian naive Bayes (PCA 32)": {"family": "probabilistic", "input": "avg",
            "grid": {"": lambda: make_pipeline(StandardScaler(), PCA(32, random_state=seed), GaussianNB())}},
        "Random forest": {"family": "trees", "input": "avg",
            "grid": {f"max_features={m}": (lambda m=m: RandomForestClassifier(500, max_features=m, min_samples_leaf=3,
                     class_weight="balanced", n_jobs=-1, random_state=seed)) for m in ("sqrt", 0.1)}},
        "Gradient boosting (PCA 64)": {"family": "trees", "input": "avg",
            "grid": {f"lr={l}": (lambda l=l: make_pipeline(StandardScaler(), PCA(64, random_state=seed), HistGradientBoostingClassifier(
                     learning_rate=l, max_iter=200, max_depth=3, l2_regularization=1.0, class_weight="balanced", random_state=seed)))
                     for l in (0.03, 0.1)}},
        "Neural net: MLP, 1 hidden layer": {"family": "neural", "input": "avg",
            "grid": {f"h={h},epochs={e}": (lambda h=h, e=e: MLP(hidden=h, dropout=0.5, epochs=e))
                     for h in (64, 256) for e in (50, 150)}},
        "Neural net: learned layer weights + linear": {"family": "neural", "input": "stack",
            "grid": {f"epochs={e}": (lambda e=e: LayerWeightedNet(hidden=0, dropout=0.5, epochs=e, lr=3e-4))
                     for e in (100, 200)}},
        "Neural net: learned layer weights + MLP": {"family": "neural", "input": "stack",
            "grid": {f"h=64,epochs={e}": (lambda e=e: LayerWeightedNet(hidden=64, dropout=0.5, epochs=e, lr=3e-4))
                     for e in (100, 200)}},
    }
