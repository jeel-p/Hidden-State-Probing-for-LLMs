"""Figures for the paper, written as PDFs to paper/figures/. Reads only saved results and labels.

    .venv/bin/python paper/make_figures.py
"""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
RES, DATA, OUT = ROOT / "results", ROOT / "data", Path(__file__).parent / "figures"
OUT.mkdir(exist_ok=True)
TAG = "Qwen2.5-3B-Instruct"
GREY, BLUE, GREEN, ORANGE, PURPLE, RED = "#7f8c8d", "#2f6fb0", "#2e8b57", "#d68910", "#7d5ba6", "#c0392b"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.spines.top": False,
                     "axes.spines.right": False, "savefig.bbox": "tight", "savefig.pad_inches": 0.02})


def rows(name):
    with (RES / name).open() as f:
        return list(csv.DictReader(f))


def jsonl(path):
    return [json.loads(l) for l in path.open() if l.strip()]


def save(fig, name):
    fig.savefig(OUT / name)
    plt.close(fig)


# 1. Pipeline
fig, ax = plt.subplots(figsize=(7, 1.35))
ax.set_xlim(0, 7); ax.set_ylim(0, 1.35); ax.axis("off")
steps = [("TruthfulQA\nquestions", "500 (v1)\n790 (v2)"), ("Qwen2.5-3B\nanswers", "greedy (v1)\n5 samples (v2)"),
         ("Labels", "embedding judge\nClaude judge"), ("Hidden\nstates", "37 layers × 2048\n2 token positions"),
         ("Probe", "LR, SVM, trees,\nneural nets")]
w, gap = 1.22, 0.2
for i, (t, sub) in enumerate(steps):
    x = 0.05 + i * (w + gap)
    ax.add_patch(FancyBboxPatch((x, 0.12), w, 1.1, boxstyle="round,pad=0.01,rounding_size=0.06", fc="#eef3f9", ec=BLUE, lw=0.8))
    ax.text(x + w / 2, 0.88, t, ha="center", va="center", fontsize=8.5, fontweight="bold")
    ax.text(x + w / 2, 0.42, sub, ha="center", va="center", fontsize=7.2, color="#333")
    if i < len(steps) - 1:
        ax.annotate("", xy=(x + w + gap - 0.03, 0.67), xytext=(x + w + 0.03, 0.67),
                    arrowprops=dict(arrowstyle="-|>", color=GREY, lw=0.9))
save(fig, "pipeline.pdf")

# 2. Label clarity: probe accuracy on old labels, and agreement with Claude, by old-label margin
clar = json.loads((RES / f"label_clarity_{TAG}_old_labels.json").read_text())["by_label_margin"]
old = {r["example_id"]: r for r in jsonl(DATA / "labeled_answers.jsonl")}
claude = {r["example_id"]: r["claude_label"] for r in jsonl(DATA / "labeled_answers_claude.jsonl")}
margin = np.array([abs(old[i]["max_sim_incorrect"] - old[i]["max_sim_correct"]) for i in claude])
agree = np.array([old[i]["correctness_label"] == claude[i] for i in claude])
edges = [0, 0.02, 0.05, 0.1, 0.2, 1.01]
labels = ["<0.02", "0.02–0.05", "0.05–0.10", "0.10–0.20", "≥0.20"]
agr = [agree[(margin >= lo) & (margin < hi)].mean() for lo, hi in zip(edges, edges[1:])]
acc = [v["cv_accuracy"] for v in clar.values()]
fig, ax = plt.subplots(figsize=(4.2, 2.4))
x = np.arange(5); bw = 0.38
ax.bar(x - bw / 2, np.array(acc) * 100, bw, color=BLUE, label="Probe CV accuracy (old labels)")
ax.bar(x + bw / 2, np.array(agr) * 100, bw, color=ORANGE, label="Agreement, old vs Claude labels")
ax.axhline(50, color=RED, ls=":", lw=0.8)
ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=7.5)
ax.set_xlabel("Old-label margin |sim(incorrect) − sim(correct)|"); ax.set_ylabel("%"); ax.set_ylim(40, 95)
ax.legend(fontsize=7, frameon=False, loc="upper left")
save(fig, "label_clarity.pdf")

# 3. Layer sweep under three label/data regimes
fig, ax = plt.subplots(figsize=(4.6, 2.5))
for name, color, lw, lab in ((f"layer_sweep_{TAG}_answer_last_token.csv", GREY, 1.2, "Embedding labels, 500 answers"),
                             (f"layer_sweep_{TAG}_claude_labels.csv", BLUE, 1.2, "Claude labels, 500 answers"),
                             (f"layer_sweep_{TAG}_v2_claude_labels.csv", GREEN, 1.8, "Claude labels, 3,950 answers")):
    r = rows(name)
    ax.plot([int(q["layer_index"]) for q in r], [float(q["cv_auroc"]) for q in r], color=color, lw=lw, label=lab)
ax.axhline(0.5, color=RED, ls=":", lw=0.8)
ax.axvline(16, color="#999", ls="--", lw=0.6); ax.text(16.4, 0.855, "baseline L16", fontsize=6.5, color="#666")
ax.set_xlabel("Layer (0 = embeddings)"); ax.set_ylabel("CV AUROC (single-layer LR)")
ax.set_xlim(-0.5, 36.5); ax.set_ylim(0.45, 0.88)
ax.legend(fontsize=7, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)
save(fig, "layer_sweep.pdf")

# 4. All models on v2: CV AUROC bars, test AUROC markers
v2 = [r for r in rows(f"all_models_{TAG}_v2_claude_labels.csv") if r["model"] != "Layer-averaged state 18-36"]
v2.sort(key=lambda r: float(r["cv_auroc"]))
def color(m):
    if m.startswith("Neural net"): return PURPLE
    if m.startswith("Baseline"): return GREY
    if any(k in m for k in ("forest", "boosting", "neighbours", "Bayes", "RBF")): return ORANGE
    return BLUE
fig, ax = plt.subplots(figsize=(6.2, 5.2))
yy = np.arange(len(v2))
ax.barh(yy, [float(r["cv_auroc"]) for r in v2], color=[color(r["model"]) for r in v2], height=0.65)
ax.scatter([float(r["test_auroc"]) for r in v2], yy, marker="D", s=12, color="black", zorder=3, label="Test AUROC (595 answers)")
ax.set_yticks(yy); ax.set_yticklabels([r["model"].replace("Neural net: ", "NN: ") for r in v2], fontsize=7)
ax.set_xlim(0.55, 0.92); ax.set_xlabel("AUROC (bars: 5-fold CV on train+val)")
from matplotlib.patches import Patch
ax.legend(handles=[Patch(color=PURPLE, label="Neural probes"), Patch(color=BLUE, label="Linear probes"),
                   Patch(color=ORANGE, label="Non-linear classical"), Patch(color=GREY, label="Layer-16 baselines"),
                   ax.collections[-1]], fontsize=7, frameon=False, loc="upper center", bbox_to_anchor=(0.35, -0.08), ncol=3)
save(fig, "models_v2.pdf")

# 5. Data scaling: 500 vs 3,950 answers (Claude labels), CV AUROC
small = {r["model"]: float(r["cv_auroc"]) for r in rows(f"all_models_{TAG}_claude_labels.csv")}
big = {r["model"]: float(r["cv_auroc"]) for r in rows(f"all_models_{TAG}_v2_claude_labels.csv")}
names = ["Baseline: layer 16, last token", "Layer-averaged state 18-36", "Best single layer (nested CV)",
         "Neural net: MLP, 1 hidden layer", "Neural net: learned layer weights + MLP", "Neural net: DualPositionNet (new)"]
short = ["LR, layer 16", "LR, layers 18–36 avg.", "LR, best layer (nested)", "MLP", "Layer-weighted net + MLP", "DualPositionNet"]
fig, ax = plt.subplots(figsize=(3.4, 2.3))
for i, n in enumerate(names):
    ax.plot([small[n], big[n]], [i, i], color="#bbb", lw=1.2, zorder=1)
    ax.scatter(small[n], i, color=GREY, s=22, zorder=2, label="500 answers" if i == 0 else None)
    ax.scatter(big[n], i, color=GREEN, s=22, zorder=2, label="3,950 answers" if i == 0 else None)
ax.set_yticks(range(len(names))); ax.set_yticklabels(short, fontsize=7.5)
ax.set_xlabel("CV AUROC (Claude labels)"); ax.legend(fontsize=7, frameon=False, loc="lower right")
save(fig, "scaling.pdf")

# 6. DualPositionNet learned layer weights
w = json.loads((Path(__file__).parent / "extras.json").read_text())["dual_layer_weights"]
layers = [int(k) for k in w["answer_last"]]
fig, ax = plt.subplots(figsize=(3.4, 2.3))
ax.bar(np.array(layers) - 0.2, [v * 100 for v in w["answer_last"].values()], 0.4, color=BLUE, label="Last answer token")
ax.bar(np.array(layers) + 0.2, [v * 100 for v in w["answer_mean"].values()], 0.4, color=ORANGE, label="Mean over answer tokens")
ax.set_xticks(layers[::2]); ax.set_xticklabels(layers[::2], fontsize=7); ax.set_xlabel("Layer"); ax.set_ylabel("Softmax weight (%)")
ax.legend(fontsize=7, frameon=False)
save(fig, "dual_weights.pdf")

# Appendix table: all v2 probes
lines = [r"\begin{tabular}{lcccccc}", r"\toprule",
         r"Probe & CV AUROC & CV acc. & Val acc. & Val AUROC & Test acc. & Test AUROC \\", r"\midrule"]
for r in sorted(rows(f"all_models_{TAG}_v2_claude_labels.csv"), key=lambda r: -float(r["cv_auroc"])):
    name = r["model"].replace("_", r"\_").replace("&", r"\&")
    lines.append(f"{name} & {float(r['cv_auroc']):.3f} & {100*float(r['cv_accuracy']):.1f} & {100*float(r['val_accuracy']):.1f} & "
                 f"{float(r['val_auroc']):.3f} & {100*float(r['test_accuracy']):.1f} & {float(r['test_auroc']):.3f} \\\\")
lines += [r"\bottomrule", r"\end{tabular}"]
(Path(__file__).parent / "v2_table.tex").write_text("\n".join(lines) + "\n")
print("figures written to", OUT)
