import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

# Usage: python docs/diagrams.py [out_dir]   (default: docs/figures)
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent / "figures"
OUT.mkdir(parents=True, exist_ok=True)
BLUE, GREEN, RED, AMBER, GREY, INK = "#2F6FB0", "#2E8B57", "#C0392B", "#D68910", "#7F8C8D", "#1F2933"
LIGHT = {BLUE: "#E3EEF9", GREEN: "#E2F3EA", RED: "#FBE5E2", AMBER: "#FCF0DC", GREY: "#EEF1F2"}
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11})


def box(ax, x, y, w, h, title, body=None, color=BLUE, title_size=12, body_size=9.5):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                fc=LIGHT[color], ec=color, lw=2))
    if body:
        ax.text(x + w / 2, y + h * 0.72, title, ha="center", va="center", fontsize=title_size,
                fontweight="bold", color=color)
        ax.text(x + w / 2, y + h * 0.36, body, ha="center", va="center", fontsize=body_size,
                color=INK, linespacing=1.4)
    else:
        ax.text(x + w / 2, y + h / 2, title, ha="center", va="center", fontsize=title_size,
                fontweight="bold", color=color, linespacing=1.3)


def arrow(ax, x1, y1, x2, y2, color=GREY, label=None):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=18,
                                 lw=2, color=color))
    if label:
        ax.text((x1 + x2) / 2 - 0.45, (y1 + y2) / 2, label, ha="center", va="center", fontsize=9, color=INK)


def canvas(w, h):
    fig, ax = plt.subplots(figsize=(w, h), dpi=200)
    ax.set_xlim(0, w); ax.set_ylim(0, h); ax.axis("off")
    return fig, ax


def save(fig, name):
    fig.savefig(OUT / name, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# 1. Big picture: four stages
fig, ax = canvas(12, 3.2)
stages = [
    ("1. Collect\nquestions", "500 tricky questions\nwith known good and\nbad answers", BLUE),
    ("2. Ask the AI\n& grade it", "AI answers each one;\nwe mark it\ntruthful or made-up", GREEN),
    ("3. Record its\n\"thinking\"", "Save a snapshot of\nthe AI's inner signals\nfor every answer", AMBER),
    ("4. Train a\ndetector", "Learn to spot made-up\nanswers from the\nsnapshot alone", RED),
]
w, gap = 2.55, 0.5
for i, (t, b, c) in enumerate(stages):
    x = 0.2 + i * (w + gap)
    box(ax, x, 0.25, w, 2.7, t, b, c, title_size=12.5, body_size=9.5)
    if i < 3:
        arrow(ax, x + w + 0.05, 1.6, x + w + gap - 0.05, 1.6)
save(fig, "01_big_picture.png")

# 2. Grading an answer
fig, ax = canvas(11, 4.6)
box(ax, 0.2, 1.6, 3.0, 1.5, "AI's answer",
    "\"Fortune cookies originated\nin Japan...\"", BLUE)
box(ax, 4.3, 3.0, 3.0, 1.4, "List of GOOD answers",
    "\"In San Francisco\"\n\"In California\"  ...", GREEN)
box(ax, 4.3, 0.3, 3.0, 1.4, "List of BAD answers",
    "\"In China\"\n\"In Japan\"  ...", RED)
arrow(ax, 3.25, 2.6, 4.25, 3.6)
arrow(ax, 3.25, 2.1, 4.25, 1.1)
ax.text(5.8, 2.35, "Which list is it\ncloser to?\nBAD list wins\n(0.91 vs 0.89)", ha="center", va="center",
        fontsize=10, color=INK, fontweight="bold")
box(ax, 8.3, 1.6, 2.5, 1.5, "Label = 1", "made-up answer\n(hallucination)", RED)
arrow(ax, 7.35, 2.35, 8.25, 2.35)
save(fig, "02_grading.png")

# 3. Layers and the snapshot
fig, ax = canvas(11, 5.2)
n_show = 9
for i in range(n_show):
    y = 0.3 + i * 0.5
    is16 = i == 4
    color = AMBER if is16 else GREY
    label = "Layer 16  <- first snapshot taken here" if is16 else ("Layer 36 (last)" if i == n_show - 1 else
            "Layer 1 (first)" if i == 0 else "...")
    ax.add_patch(FancyBboxPatch((0.4, y), 4.2, 0.36, boxstyle="round,pad=0.01,rounding_size=0.05",
                                fc=LIGHT[color], ec=color, lw=2 if is16 else 1))
    ax.text(2.5, y + 0.18, label, ha="center", va="center", fontsize=9.5,
            fontweight="bold" if is16 else "normal", color=INK)
ax.text(2.5, 4.95, "Inside the AI: 36 processing layers", ha="center", fontsize=11.5,
        fontweight="bold", color=INK)
ax.text(2.5, 0.02, "Question + answer go in at the bottom", ha="center", fontsize=9, color=INK)
arrow(ax, 4.7, 2.48, 6.2, 2.48, color=AMBER)
box(ax, 6.3, 1.5, 4.4, 2.0, "Snapshot = 2,048 numbers",
    "[ 0.12, -1.07, 0.88, 2.31, ... ]\nsaved as  Q001_A001.npy\n(one file per answer)", AMBER)
ax.text(8.5, 1.05, "Later, a snapshot was taken at\nevery layer to compare them", ha="center", va="center",
        fontsize=9, color=INK, style="italic")
save(fig, "03_snapshot.png")

# 4. Train / practice / final exam split
fig, ax = canvas(11, 2.6)
total_w = 10.4
parts = [("Study set (train)", 350, BLUE, "detector learns\nfrom these"),
         ("Practice test (val)", 75, AMBER, "used to tune\nsettings"),
         ("Final exam (test)", 75, GREEN, "never seen until\nthe very end")]
x = 0.3
widths = {350: 4.8, 75: 2.8}
for name, n, c, desc in parts:
    w = widths[n]
    box(ax, x, 0.9, w - 0.08, 1.4, f"{name}\n{n} questions", None, c, title_size=10.5)
    ax.text(x + w / 2, 0.45, desc, ha="center", va="center", fontsize=9, color=INK)
    x += w
save(fig, "04_split.png")

# 5. Results chart (AUROC values from results/metrics_Qwen2.5-3B-Instruct_L16_answer_last_token.json)
fig, ax = plt.subplots(figsize=(8, 3.8), dpi=200)
names = ["Study set\n(train)", "Practice test\n(val)", "Final exam\n(test)"]
vals = [0.89, 0.61, 0.65]
bars = ax.bar(names, vals, color=[BLUE, AMBER, GREEN], width=0.55)
ax.axhline(0.5, color=RED, ls="--", lw=1.5)
ax.text(2.95, 0.515, "coin flip (0.5)", color=RED, fontsize=9, ha="right")
ax.axhline(1.0, color=GREY, ls=":", lw=1.2)
ax.text(2.95, 1.015, "perfect (1.0)", color=GREY, fontsize=9, ha="right")
for b, v in zip(bars, vals):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.2f}", ha="center", fontweight="bold")
ax.set_ylim(0, 1.1); ax.set_xlim(-0.5, 3.0); ax.set_ylabel("Detection score (AUROC)")
for s in ["top", "right"]:
    ax.spines[s].set_visible(False)
save(fig, "05_results.png")

# 6. Layer comparison: cross-validated score for a detector trained at each layer
# (read from the layer-sweep CSV rather than typed by hand: 37 values)
import csv
sweep = Path(__file__).resolve().parents[1] / "results" / "layer_sweep_Qwen2.5-3B-Instruct_answer_last_token.csv"
with sweep.open() as f:
    rows = list(csv.DictReader(f))
layers = [int(r["layer_index"]) for r in rows]
cv = [float(r["cv_auroc"]) for r in rows]
best = max(layers, key=lambda i: cv[i])
fig, ax = plt.subplots(figsize=(8, 3.9), dpi=200)
ax.axvspan(-0.5, 3.5, color=LIGHT[GREY])
ax.text(3.8, 0.755, "<- first few layers: little signal", ha="left", va="top", fontsize=8.5, color=INK)
ax.plot(layers, cv, color=BLUE, lw=2.5, marker="o", ms=3.5)
ax.axhline(0.5, color=RED, ls="--", lw=1.5)
ax.text(36, 0.505, "coin flip (0.5)", color=RED, fontsize=9, ha="right", va="bottom")
for i, color, text, dy in ((16, AMBER, "layer 16\n(first choice)", -0.055), (best, GREEN, f"layer {best}\n(best)", 0.03)):
    ax.plot(i, cv[i], "o", ms=10, color=color, zorder=3)
    ax.annotate(f"{text}: {cv[i]:.2f}", (i, cv[i]), xytext=(i, cv[i] + dy), ha="center",
                fontsize=9, fontweight="bold", color=color)
ax.set_xlim(-0.5, 36.5); ax.set_ylim(0.45, 0.77)
ax.set_xlabel("Layer where the snapshot was taken  (0 = before the first layer, 36 = last)")
ax.set_ylabel("Detection score (AUROC)")
for sp in ["top", "right"]:
    ax.spines[sp].set_visible(False)
save(fig, "06_layers.png")

# 7. Original vs improved detector (read from the notebook 06 summary)
import json
imp = json.loads((Path(__file__).resolve().parents[1] / "results" /
                  "improvement_Qwen2.5-3B-Instruct_answer_last_token_L18-36avg.json").read_text())
cv = {r["group"]: r["cv_auroc"] for r in imp["ablation"] if r["group"] in ("baseline", "final")}
groups = ["Practice tests\n(average of 15)", "Final exam\n(75 questions)"]
original = [cv["baseline"], imp["test"]["baseline_shipped"]]
improved = [cv["final"], imp["test"]["improved"]]
fig, ax = plt.subplots(figsize=(8, 3.8), dpi=200)
x = range(len(groups)); w = 0.34
for off, vals, color, name in ((-w / 2, original, GREY, "Original detector (layer 16)"),
                               (w / 2, improved, GREEN, "Improved detector")):
    bars = ax.bar([i + off for i in x], vals, w, color=color, label=name)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.015, f"{v:.2f}", ha="center", fontweight="bold")
ax.axhline(0.5, color=RED, ls="--", lw=1.5, label="Coin flip (0.5)")
ax.set_xticks(list(x)); ax.set_xticklabels(groups)
ax.set_ylim(0, 0.9); ax.set_xlim(-0.5, 1.6); ax.set_ylabel("Detection score (AUROC)")
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3, frameon=False, fontsize=9.5)
for sp in ["top", "right"]:
    ax.spines[sp].set_visible(False)
save(fig, "07_improved.png")

# 8. Other kinds of detectors (read from the notebook 07 summary)
PLAIN = {
    "Logistic regression (baseline)": "Our standard detector (logistic regression)",
    "Linear SVM": "Another straight-line detector (linear SVM)",
    "Neural net: MLP, 1 hidden layer": "Small neural network",
    "Neural net: learned layer weights + linear": "Neural network that picks its own layers",
    "RBF-kernel SVM": "Curved-boundary detector (RBF SVM)",
    "Neural net: learned layer weights + MLP": "Layer-picking neural network, deeper",
    "Random forest": "Many decision trees voting (random forest)",
    "Logistic regression, L1 (sparse)": "Straight-line detector using few numbers (L1)",
    "k-nearest neighbours": "Vote of the most similar answers (k-NN)",
    "Gradient boosting (PCA 64)": "Decision trees built one after another (boosting)",
    "Gaussian naive Bayes (PCA 32)": "Simple probability counting (naive Bayes)",
}
with (Path(__file__).resolve().parents[1] / "results" /
      "model_comparison_Qwen2.5-3B-Instruct_answer_last_token_L18-36avg.csv").open() as f:
    models = sorted(csv.DictReader(f), key=lambda r: float(r["cv_auroc"]))
fig, ax = plt.subplots(figsize=(8, 4.8), dpi=200)
names = [PLAIN[r["model"]] for r in models]
vals = [float(r["cv_auroc"]) for r in models]
colors = [BLUE if r["model"].endswith("(baseline)") else GREEN if r["family"] == "neural" else GREY for r in models]
ax.barh(names, vals, color=colors, height=0.62)
for y, v in enumerate(vals):
    ax.text(v + 0.003, y, f"{v:.2f}", va="center", fontsize=9, fontweight="bold")
ax.set_xlim(0.5, 0.75); ax.set_xlabel("Detection score, average of 15 practice tests (AUROC)")
ax.tick_params(axis="y", labelsize=9.5)
for sp in ["top", "right"]:
    ax.spines[sp].set_visible(False)
save(fig, "08_models.png")

# 9. Old labels vs Claude's labels, same three detectors (read from notebooks 06, 07 and 09)
RES = Path(__file__).resolve().parents[1] / "results"
cv_old = {r["candidate"]: r["cv_auroc"] for r in imp["ablation"]}
with (RES / "model_comparison_Qwen2.5-3B-Instruct_answer_last_token_L18-36avg.csv").open() as f:
    cv_old.update({r["model"]: float(r["cv_auroc"]) for r in csv.DictReader(f)})
with (RES / "all_models_Qwen2.5-3B-Instruct_claude_labels.csv").open() as f:
    cv_new = {r["model"]: float(r["cv_auroc"]) for r in csv.DictReader(f)}
detectors = [("Original detector\n(layer 16)", "Baseline: layer 16, last token"),
             ("Later layers\naveraged", "Layer-averaged state 18-36"),
             ("Best neural network\n(picks its own layers)", "Neural net: learned layer weights + MLP")]
fig, ax = plt.subplots(figsize=(8, 3.9), dpi=200)
x = range(len(detectors)); w = 0.34
for off, src, color, name in ((-w / 2, cv_old, GREY, "Old automatic grading"),
                              (w / 2, cv_new, GREEN, "Graded by Claude")):
    vals = [src[key] for _, key in detectors]
    bars = ax.bar([i + off for i in x], vals, w, color=color, label=name)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.012, f"{v:.2f}", ha="center", fontweight="bold")
ax.axhline(0.5, color=RED, ls="--", lw=1.5, label="Coin flip (0.5)")
ax.set_xticks(list(x)); ax.set_xticklabels([n for n, _ in detectors])
ax.set_ylim(0, 0.95); ax.set_ylabel("Detection score, average of\n15 practice tests (AUROC)")
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.24), ncol=3, frameon=False, fontsize=9.5)
for sp in ["top", "right"]:
    ax.spines[sp].set_visible(False)
save(fig, "09_relabel.png")

# 10. 500 vs 3,950 graded answers, same three detectors (read from notebooks 09 and 10)
def accuracy(path):
    with (RES / path).open() as f:
        return {r["model"]: float(r["cv_accuracy"]) for r in csv.DictReader(f)}
small = accuracy("all_models_Qwen2.5-3B-Instruct_claude_labels.csv")
big = accuracy("all_models_Qwen2.5-3B-Instruct_v2_claude_labels.csv")
detectors = [("Original detector\n(layer 16)", "Baseline: layer 16, last token"),
             ("Later layers\naveraged", "Layer-averaged state 18-36"),
             ("New neural network\n(two word positions)", "Neural net: DualPositionNet (new)")]
fig, ax = plt.subplots(figsize=(8, 3.9), dpi=200)
x = range(len(detectors)); w = 0.34
for off, src, color, name in ((-w / 2, small, GREY, "500 answers"), (w / 2, big, GREEN, "3,950 answers")):
    vals = [100 * src[key] for _, key in detectors]
    bars = ax.bar([i + off for i in x], vals, w, color=color, label=name)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 1, f"{v:.0f}", ha="center", fontweight="bold")
ax.axhline(56, color=RED, ls="--", lw=1.5, label="Always guessing “truthful” (56)")   # majority rate of the v2 train+val answers (57 for the 500)
ax.set_xticks(list(x)); ax.set_xticklabels([n for n, _ in detectors])
ax.set_ylim(0, 95); ax.set_ylabel("Right answers out of 100\n(practice tests)")
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.24), ncol=3, frameon=False, fontsize=9.5)
for sp in ["top", "right"]:
    ax.spines[sp].set_visible(False)
save(fig, "10_more_data.png")

# 11. Detection score at every layer: old grading, Claude's grading, and Claude's grading with 8x the data
def sweep(path):
    with (RES / path).open() as f:
        rows = list(csv.DictReader(f))
    return [int(r["layer_index"]) for r in rows], [float(r["cv_auroc"]) for r in rows]
fig, ax = plt.subplots(figsize=(8, 3.9), dpi=200)
for path, color, lw, name in (("layer_sweep_Qwen2.5-3B-Instruct_answer_last_token.csv", GREY, 1.6, "Old grading, 500 answers"),
                              ("layer_sweep_Qwen2.5-3B-Instruct_claude_labels.csv", BLUE, 1.6, "Claude’s grading, 500 answers"),
                              ("layer_sweep_Qwen2.5-3B-Instruct_v2_claude_labels.csv", GREEN, 2.6, "Claude’s grading, 3,950 answers")):
    layers, cv = sweep(path)
    ax.plot(layers, cv, color=color, lw=lw, label=name)
best = max(range(len(cv)), key=lambda i: cv[i])
ax.plot(layers[best], cv[best], "o", ms=9, color=GREEN, zorder=3)
ax.annotate(f"best: layer {layers[best]} ({cv[best]:.2f})", (layers[best], cv[best]), xytext=(layers[best] + 1.2, cv[best] + 0.015),
            fontsize=9, fontweight="bold", color=GREEN)
ax.axhline(0.5, color=RED, ls="--", lw=1.2, label="Coin flip (0.5)")
ax.set_xlim(-0.5, 36.5); ax.set_ylim(0.45, 0.9)
ax.set_xlabel("Layer where the snapshot was taken  (0 = before the first layer, 36 = last)")
ax.set_ylabel("Detection score (AUROC)")
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, frameon=False, fontsize=9)
for sp in ["top", "right"]:
    ax.spines[sp].set_visible(False)
save(fig, "11_layers_v2.png")

# 12. Every detector on the 3,950-answer data: practice-test and final-exam accuracy
PLAIN_V2 = {
    "Neural net: DualPositionNet (new)": "New neural network (two word positions)",
    "Neural net: learned layer weights + MLP": "Neural network that picks its own layers",
    "Best single layer (nested CV)": "Best single layer (layer 25)",
    "One probe per layer 18-36, averaged": "19 detectors, one per layer, voting",
    "Neural net: learned layer weights + linear": "Layer-picking network, straight-line head",
    "Neural net: MLP, 1 hidden layer": "Small neural network",
    "Last + mean, averaged": "Last word + all words, voting",
    "RBF-kernel SVM": "Curved-boundary detector (RBF SVM)",
    "Layer-averaged + uncertainty (stacked)": "Later layers + AI’s own confidence",
    "Logistic regression (baseline)": "Standard detector, later layers averaged",
    "Layer-averaged 18-36 + high-confidence labels": "Standard detector, only sure grades",
    "Random forest": "Many decision trees voting",
    "Answer mean, layer-averaged 18-36": "Standard detector, all words averaged",
    "k-nearest neighbours": "Vote of the most similar answers",
    "Linear SVM": "Another straight-line detector",
    "Baseline: layer 16, last token": "Original detector (layer 16)",
    "Logistic regression, L1 (sparse)": "Straight-line, using few numbers",
    "Layer-averaged, PCA 64 + LR": "Snapshot simplified first, then detector",
    "Gradient boosting (PCA 64)": "Decision trees built one after another",
    "Layer 16 + high-confidence labels only": "Original detector, only sure grades",
    "Gaussian naive Bayes (PCA 32)": "Simple probability counting",
    "Layer-averaged, shrinkage LDA": "Classic statistics method (LDA)",
    "Layer-averaged, mass-mean probe": "Closest-average detector",
    "Baseline: LR layer 16 (C picked on val)": "Original recipe, tuned on one practice test",
    "Uncertainty features alone": "AI’s own confidence only (no snapshot)",
}
with (RES / "all_models_Qwen2.5-3B-Instruct_v2_claude_labels.csv").open() as f:
    rows = [r for r in csv.DictReader(f) if r["model"] != "Layer-averaged state 18-36"]   # same model as "Logistic regression (baseline)"
rows.sort(key=lambda r: float(r["cv_accuracy"]))
fig, ax = plt.subplots(figsize=(8, 9.2), dpi=200)
y = list(range(len(rows))); h = 0.38
practice = [100 * float(r["cv_accuracy"]) for r in rows]
exam = [100 * float(r["test_accuracy"]) for r in rows]
ax.barh([i - h / 2 for i in y], practice, h, color=GREEN, label="Practice tests")
ax.barh([i + h / 2 for i in y], exam, h, color=BLUE, label="Final exam (595 answers)")
for i, (p_, e_) in enumerate(zip(practice, exam)):
    ax.text(p_ + 0.4, i - h / 2, f"{p_:.0f}", va="center", fontsize=7.5)
    ax.text(e_ + 0.4, i + h / 2, f"{e_:.0f}", va="center", fontsize=7.5)
ax.axvline(61, color=RED, ls="--", lw=1.2, zorder=0, label="Always guessing “truthful” (final exam: 61)")
ax.set_yticks(y); ax.set_yticklabels([PLAIN_V2[r["model"]] for r in rows], fontsize=8.5)
ax.set_xlim(50, 85); ax.set_xlabel("Right answers out of 100")
ax.legend(loc="upper center", bbox_to_anchor=(0.35, -0.065), ncol=3, frameon=False, fontsize=8.5)
for sp in ["top", "right"]:
    ax.spines[sp].set_visible(False)
save(fig, "12_all_models_v2.png")
print("diagrams written to", OUT)
