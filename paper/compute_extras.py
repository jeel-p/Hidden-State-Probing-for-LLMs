"""Numbers the paper needs that the notebooks don't save: test-set bootstrap intervals and paired
comparisons on dataset v2, and DualPositionNet's learned layer weights. Writes paper/extras.json.

    .venv/bin/python paper/compute_extras.py
"""
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import config                                               # noqa: E402
from src.expand import load_v2                                       # noqa: E402
from src.improve import layer_average, linear_probe                   # noqa: E402
from src.io_utils import read_jsonl                                  # noqa: E402
from src.relabel import DualPositionNet                              # noqa: E402

LATE = list(range(18, 37))
N_BOOT = 2000


def _resample(groups, rng):
    """Question-level (cluster) bootstrap: the 5 answers to a question are correlated, so whole
    questions are resampled; resampling single answers would make the intervals too narrow."""
    qs = np.unique(groups)
    pick = rng.choice(qs, len(qs))
    by_q = {q: np.where(groups == q)[0] for q in qs}
    return np.concatenate([by_q[q] for q in pick])


def bootstrap_ci(y, p, groups, metric, seed=config.SEED):
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(N_BOOT):
        i = _resample(groups, rng)
        if y[i].min() != y[i].max():
            vals.append(metric(y[i], p[i]))
    return np.percentile(vals, [2.5, 97.5]).tolist()


def paired_cluster_bootstrap(y, p_a, p_b, groups, seed=config.SEED):
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(N_BOOT):
        i = _resample(groups, rng)
        if y[i].min() != y[i].max():
            diffs.append(roc_auc_score(y[i], p_b[i]) - roc_auc_score(y[i], p_a[i]))
    diffs = np.array(diffs)
    return {"diff": float(roc_auc_score(y, p_b) - roc_auc_score(y, p_a)),
            "ci95": np.percentile(diffs, [2.5, 97.5]).tolist(), "p_b_better": float((diffs > 0).mean())}


def main():
    d, conf, _ = load_v2()
    d.answer_last = d.answer_last.astype(np.float32)
    d.answer_mean = d.answer_mean.astype(np.float32)
    tr, te, y = d.pool, d.test, d.y[d.test]
    g = d.question_id[te]
    avg = layer_average(d.answer_last, LATE)
    dual_x = np.stack([d.answer_last[:, LATE], d.answer_mean[:, LATE]], 1)

    # same settings notebook 10 picked by CV
    dual = DualPositionNet(hidden=64, dropout=0.5, epochs=150, lr=3e-4).fit(dual_x[tr], d.y[tr])
    probs = {
        "DualPositionNet": dual.predict_proba(dual_x[te])[:, 1],
        "LR, layers 18-36 averaged": linear_probe(d, lambda i: avg[i], 1e-3)(tr, te),
        "LR, layer 16": linear_probe(d, lambda i: d.answer_last[i, 16], 1e-3)(tr, te),
    }
    acc = lambda yy, pp: accuracy_score(yy, pp > 0.5)
    out = {"bootstrap": f"{N_BOOT} question-level resamples of the test set", "test_n": int(len(y)), "test_questions": int(len(set(g))), "test_majority": float(max(y.mean(), 1 - y.mean())), "models": {}, "paired": {}}
    for name, p in probs.items():
        out["models"][name] = {"test_accuracy": acc(y, p), "test_accuracy_ci95": bootstrap_ci(y, p, g, acc),
                               "test_auroc": roc_auc_score(y, p), "test_auroc_ci95": bootstrap_ci(y, p, g, roc_auc_score)}
    for base in ("LR, layers 18-36 averaged", "LR, layer 16"):
        out["paired"][f"DualPositionNet - {base} (AUROC)"] = paired_cluster_bootstrap(y, probs[base], probs["DualPositionNet"], g)
    w = dual.layer_weights()                      # (2 positions, 19 layers)
    out["dual_layer_weights"] = {pos: dict(zip(map(str, LATE), w[k].tolist()))
                                 for k, pos in enumerate(["answer_last", "answer_mean"])}

    # dataset statistics for Table 1
    old = read_jsonl(config.LABELED_PATH)
    v1c = read_jsonl(config.DATA_DIR / "labeled_answers_claude.jsonl")
    v2c = read_jsonl(config.DATA_DIR / "v2" / "labeled_answers_claude.jsonl")
    count = lambda rs, k: {"n": len(rs), "hallucination_rate": float(np.mean([r[k] for r in rs]))}
    out["datasets"] = {"v1_embedding_labels": count(old, "correctness_label"),
                       "v1_claude_labels": count(v1c, "claude_label"),
                       "v2_claude_labels": count(v2c, "claude_label"),
                       "v2_confidence": {c: sum(r["claude_confidence"] == c for r in v2c) for c in ("high", "medium", "low")},
                       "v1_confidence": {c: sum(r["claude_confidence"] == c for r in v1c) for c in ("high", "medium", "low")},
                       "judge_cost_usd": {"v1": sum(r["input_tokens"] * 4 + r["output_tokens"] * 20 for r in v1c[5:]) / 2e6
                                                + sum(r["input_tokens"] * 4 + r["output_tokens"] * 20 for r in v1c[:5]) / 1e6,
                                          "v2": sum(r["input_tokens"] * 4 + r["output_tokens"] * 20 for r in v2c[5:]) / 2e6
                                                + sum(r["input_tokens"] * 4 + r["output_tokens"] * 20 for r in v2c[:5]) / 1e6}}
    (Path(__file__).parent / "extras.json").write_text(json.dumps(out, indent=2, default=float))
    print(json.dumps({k: out[k] for k in ("models", "paired")}, indent=1, default=float))


if __name__ == "__main__":
    main()
