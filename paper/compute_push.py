"""Numbers for the paper's "Pushing accuracy further" section: consistency across the 5 sampled
answers, ensembles, CV-tuned thresholds, accuracy by judge confidence, and a LoRA-tuned detector.
Writes paper/push.json. Needs the v2 features (python -m src.expand extract). --lora also needs a
CUDA GPU and `pip install peft`; it takes about an hour on an RTX 5060 Ti.

    .venv/bin/python paper/compute_push.py
    .venv/bin/python paper/compute_push.py --lora
"""
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import config                                               # noqa: E402
from src.expand import load_v2                                       # noqa: E402
from src.improve import _lr, layer_average                           # noqa: E402
from src.model_compare import sk_model                               # noqa: E402
from src.relabel import DualPositionNet                              # noqa: E402

LATE = list(range(18, 37))
OUT = Path(__file__).parent / "push.json"


def logit(p):
    return np.log(np.clip(p, 1e-6, 1 - 1e-6) / np.clip(1 - p, 1e-6, 1))


def lr(X, y, C=1e-3):
    """The probes' usual logistic regression on rows of X (cast from float16 per fit)."""
    return lambda tr, te: _lr(C).fit(X[tr].astype(np.float32), y[tr]).predict_proba(X[te].astype(np.float32))[:, 1]


def stacked(d, bases, meta, extra=None, siblings=False, inner=5):
    """Meta-model on the base models' out-of-fold logits, optionally plus the mean logit of the same
    question's other answers and extra per-answer features."""
    def Z(idx, logits):
        cols = [logits]
        if siblings:
            q = d.question_id[idx]
            sib = [(logits[q == q[k], 0].sum() - logits[k, 0]) / max((q == q[k]).sum() - 1, 1) for k in range(len(idx))]
            cols.append(np.array(sib)[:, None])
        if extra is not None:
            cols.append(extra[idx])
        return np.hstack(cols)
    base = lambda tr, te: np.column_stack([logit(b(tr, te)) for b in bases])
    def fit_predict(tr, te):
        oof = np.zeros((len(tr), len(bases)))
        for a, b in StratifiedGroupKFold(inner, shuffle=True, random_state=config.SEED).split(tr, d.y[tr], d.question_id[tr]):
            oof[b] = base(tr[a], tr[b])
        return meta().fit(Z(tr, oof), d.y[tr]).predict_proba(Z(te, base(tr, te)))[:, 1]
    return fit_predict


def evaluate(d, model):
    """notebook 10's three views (one CV round, same folds), keeping every probability."""
    pool, y = d.pool, d.y
    oof = np.zeros(len(pool))
    for a, b in StratifiedGroupKFold(5, shuffle=True, random_state=config.SEED).split(pool, y[pool], d.question_id[pool]):
        oof[b] = model(pool[a], pool[b])
    return {"oof": oof, "val": model(np.where(d.split == "train")[0], np.where(d.split == "val")[0]),
            "test": model(pool, d.test)}


def metrics(d, p, t=0.5):
    yv = d.y[d.split == "val"]
    return {"cv_auroc": roc_auc_score(d.y[d.pool], p["oof"]), "cv_accuracy": accuracy_score(d.y[d.pool], p["oof"] > t),
            "val_accuracy": accuracy_score(yv, p["val"] > t),
            "test_accuracy": accuracy_score(d.y[d.test], p["test"] > t), "test_auroc": roc_auc_score(d.y[d.test], p["test"])}


def lora_detector(d, records, epochs=3, lr_=2e-4, rank=16, batch=8):
    """Qwen2.5-3B with LoRA adapters on every projection and a linear head on the last answer
    token, trained on exactly the prompt + answer the probes read. bfloat16, one seed."""
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from src.model_utils import encode_prompt_and_answer
    tok = AutoTokenizer.from_pretrained(config.MODEL_NAME)
    ids = [encode_prompt_and_answer(tok, r["question"], r["generated_answer"])[0] for r in records]
    dev = "cuda"

    class Net(torch.nn.Module):
        def __init__(self):
            super().__init__()
            base = AutoModelForCausalLM.from_pretrained(config.MODEL_NAME, dtype=torch.bfloat16)
            self.lm = get_peft_model(base.model, LoraConfig(r=rank, lora_alpha=2 * rank, lora_dropout=0.1, target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
            self.head = torch.nn.Linear(base.config.hidden_size, 1)

        def forward(self, x, m, last):
            h = self.lm(input_ids=x, attention_mask=m).last_hidden_state
            return self.head(h[torch.arange(len(last)), last].float()).squeeze(-1)

    def batches(idx, rng=None):
        idx = rng.permutation(idx) if rng is not None else idx
        for s in range(0, len(idx), batch):
            b = idx[s:s + batch]
            L = max(len(ids[i]) for i in b)
            x = torch.full((len(b), L), tok.pad_token_id)
            m = torch.zeros((len(b), L), dtype=torch.long)
            for k, i in enumerate(b):
                x[k, :len(ids[i])] = torch.tensor(ids[i])
                m[k, :len(ids[i])] = 1
            yield b, x.to(dev), m.to(dev), torch.tensor([len(ids[i]) - 1 for i in b], device=dev)

    def fit_predict(tr, te):
        torch.manual_seed(config.SEED)
        net = Net().to(dev)
        params = [p for p in net.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(params, lr=lr_, weight_decay=0.01)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, lr_, total_steps=epochs * -(-len(tr) // batch), pct_start=0.1)
        pos_weight = torch.tensor((1 - d.y[tr].mean()) / d.y[tr].mean(), device=dev)
        yt = torch.tensor(d.y, dtype=torch.float32, device=dev)
        rng = np.random.default_rng(config.SEED)
        net.train()
        for _ in range(epochs):
            for b, x, m, last in batches(tr, rng):
                loss = torch.nn.functional.binary_cross_entropy_with_logits(net(x, m, last), yt[b], pos_weight=pos_weight)
                opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step(); sched.step()
        net.eval()
        with torch.no_grad():
            out = np.concatenate([torch.sigmoid(net(x, m, last)).float().cpu().numpy() for _, x, m, last in batches(te)])
        del net, opt
        torch.cuda.empty_cache()
        return out
    return fit_predict


def main(with_lora=False):
    d, conf, records = load_v2()                  # hidden states stay float16; slices are cast per fit
    q, y = d.question_id, d.y
    groups = [np.where(q == g)[0] for g in np.unique(q)]
    def per_question(X, fn):                      # fn maps one question's rows to one value per row; no labels used
        out = np.empty((len(X),) + fn(X[groups[0]]).shape[1:])
        for i in groups:
            out[i] = fn(X[i])
        return out
    rows = lambda v, Z: np.repeat(v[None], len(Z), 0)                # one per-question value -> every row
    qmean = lambda X: per_question(X, lambda Z: rows(Z.mean(0), Z))

    avg = layer_average(d.answer_last, LATE).astype(np.float32)
    avg_m = layer_average(d.answer_mean, LATE).astype(np.float32)
    h = d.answer_mean[:, 25].astype(np.float32)
    h /= np.linalg.norm(h, axis=1, keepdims=True)
    agree = per_question(h, lambda H: ((H @ H.T).sum(1) - 1) / max(len(H) - 1, 1))   # similarity to the other samples
    cons = np.column_stack([agree, qmean(agree), qmean(d.uncertainty), per_question(d.uncertainty, lambda U: rows(U.std(0), U))])

    dual = np.stack([d.answer_last[:, LATE], d.answer_mean[:, LATE]], 1)
    dual_net = lambda seeds: sk_model(d, lambda i: dual[i].astype(np.float32),
                                      lambda: DualPositionNet(hidden=64, dropout=0.5, epochs=150, lr=3e-4, seeds=seeds))
    per_layer = [lr(getattr(d, r)[:, L], y) for r in ("answer_last", "answer_mean") for L in range(14, 37, 2)]

    P = {"LR, layers 18-36 avg.": evaluate(d, lr(avg, y)),
         "+ mean-token and question-mean states": evaluate(d, lr(np.hstack([avg, avg_m, qmean(avg), qmean(avg_m)]), y)),
         "+ sibling scores, stacked": evaluate(d, stacked(d, [lr(avg, y)], lambda: _lr(1.0), extra=cons, siblings=True)),
         "Consistency features alone": evaluate(d, lr(cons, y, C=1.0)),
         "DualPositionNet": evaluate(d, dual_net(3)),
         "DualPositionNet, 10 seeds": evaluate(d, dual_net(10)),
         "Stacked per-layer LR": evaluate(d, stacked(d, per_layer, lambda: LogisticRegression(C=0.1, max_iter=3000))),
         "LR, layer 25": evaluate(d, lr(d.answer_last[:, 25], y))}
    mean_of = lambda names: {k: np.mean([P[n][k] for n in names], 0) for k in ("oof", "val", "test")}
    ENS = "Ensemble: DualPositionNet + stacked layers + layer 25"
    P[ENS] = mean_of(["DualPositionNet, 10 seeds", "Stacked per-layer LR", "LR, layer 25"])
    if with_lora:
        P["LoRA detector"] = evaluate(d, lora_detector(d, records))
        P["Ensemble + LoRA detector"] = mean_of([ENS, "LoRA detector"])

    out = {"protocol": "same splits and CV folds as notebook 10 (one round of grouped 5-fold CV on train+val)", "models": {}}
    for name, p in P.items():
        ts = np.linspace(0.3, 0.7, 41)
        t = float(ts[np.argmax([accuracy_score(y[d.pool], p["oof"] > t) for t in ts])])
        out["models"][name] = {**metrics(d, p), "cv_threshold": t, "at_cv_threshold": metrics(d, p, t)}
    oof, c = P["DualPositionNet"]["oof"], conf[d.pool]
    out["dual_by_judge_confidence"] = {nm: {"n": int((c == k).sum()), "cv_accuracy": accuracy_score(y[d.pool][c == k], oof[c == k] > .5),
                                            "cv_auroc": roc_auc_score(y[d.pool][c == k], oof[c == k])}
                                       for k, nm in enumerate(["low", "medium", "high"])}
    n_hall = per_question(y.astype(float), lambda v: np.full(len(v), v.sum()))
    sizes = per_question(y.astype(float), lambda v: np.full(len(v), len(v)))
    out["labels_by_question"] = {"share_questions_all_same": float(np.mean([y[i].min() == y[i].max() for i in groups])),
                                 "sibling_label_majority_accuracy": float(np.mean(((n_hall - y) / (sizes - 1) > 0.5) == y))}
    if not with_lora and OUT.exists():            # keep the LoRA rows of an earlier --lora run
        out["models"].update({k: v for k, v in json.loads(OUT.read_text())["models"].items() if "LoRA" in k})
    OUT.write_text(json.dumps(out, indent=2, default=float))
    for k, v in out["models"].items():
        print(f"{k:55s} " + " ".join(f"{m} {v[m]:.3f}" for m in ("cv_auroc", "cv_accuracy", "val_accuracy", "test_accuracy", "test_auroc")))


if __name__ == "__main__":
    main(with_lora="--lora" in sys.argv)
