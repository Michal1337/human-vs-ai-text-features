"""Draw the figures of the paper from the tables in data/ into figures/."""
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import figstyle as fs

# constrained layout keeps every figure at exactly the authored width
plt.rcParams.update({"figure.constrained_layout.use": True,
                     "savefig.bbox": None})

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
OUT = os.path.join(HERE, "..", "figures") + os.sep
os.makedirs(OUT, exist_ok=True)
GEN, LBL, NEG = fs.GEN, fs.LBL, 0.147
HUMAN_C, AI_C = fs.HUMAN_C, fs.AI_C
CONS_C, VAR_C = fs.CONS_C, fs.VAR_C
FAM_ORDER, FAM_C = fs.FAM_ORDER, fs.FAM_C

eff = pd.read_csv(os.path.join(DATA, "effects.csv"))
hum = pd.read_csv(os.path.join(DATA, "human_stats.csv"))
hist = pd.read_csv(os.path.join(DATA, "hist.csv"))
sent = pd.read_csv(os.path.join(DATA, "sentiment.csv"))

# The 16 cluster representatives of Section 4.2: one per group of features
# correlating at |r| >= 0.85 over the 210 model-by-genre cells.
KEEP = ["flesch_reading_ease", "smog_index", "dale_chall_readability",
        "noun_ratio", "word_count", "std_sentence_length", "Mass", "DMetric",
        "sentiment", "pronoun_count", "stopword_ratio", "verb_ratio",
        "contraction_count", "passive_voice_count", "dependency_distance",
        "discourse_marker_ratio"]
# style score of Section 5.2: signed so that positive means harder to read
# and more nominal in AI text. SMOG is left out because textstat sets it to 0
# for texts of fewer than three sentences, which ties it to length.
STYLE = {"flesch_reading_ease": -1, "dale_chall_readability": 1,
         "noun_ratio": 1}

allai = eff[eff.model == "ALL_AI"]
piv = allai.pivot_table(index="feature", columns="dataset",
                        values="cliffs_delta")[GEN].loc[KEEP]
macro, mad = piv.mean(axis=1), piv.abs().mean(axis=1)
nn = piv.abs() >= NEG
order = mad.sort_values(ascending=False).index.tolist()
per_model = eff[(eff.model != "ALL_AI") & eff.feature.isin(KEEP)]
GORD = piv.abs().mean(axis=0).sort_values(ascending=False).index.tolist()


def feat_class(f):
    sig = piv.loc[f][nn.loc[f]]
    if len(sig) == 0:
        return "negligible"
    return "variable" if (sig > 0).any() and (sig < 0).any() else "consistent"


classes = {f: feat_class(f) for f in piv.index}
FEATL = [fs.fl(f) for f in order]

# ----------------------------------------------- Fig 2: per-feature effects
fig, ax = plt.subplots(figsize=fs.size(1.0, 4.7))
for i, f in enumerate(order):
    v = piv.loc[f].values
    ax.plot([v.min(), v.max()], [i, i], color="#d2d2d2", lw=1.0, zorder=1,
            solid_capstyle="round")
    sig = np.abs(v) >= NEG
    ax.scatter(v[~sig], [i] * (~sig).sum(), s=9, color="#a8a8a8", zorder=2,
               linewidths=0)
    c = CONS_C if classes[f] == "consistent" else VAR_C
    ax.scatter(v[sig], [i] * sig.sum(), s=14, color=c, zorder=3, linewidths=0)
    ax.scatter([macro[f]], [i], s=30, marker="D", facecolor="white",
               edgecolor="#111111", linewidths=0.9, zorder=4)
ax.axvline(0, color="#555555", lw=0.8, zorder=0)
for x in (-NEG, NEG):
    ax.axvline(x, color="#aaaaaa", lw=0.7, ls=(0, (4, 3)), zorder=0)
ax.set_yticks(range(len(order)))
ax.set_yticklabels(FEATL)
ax.set_ylim(-0.8, len(order) - 0.2)
ax.invert_yaxis()
ax.set_xlim(-1.05, 1.05)
ax.set_xlabel("Cliff's $\\delta$   (one dot per genre)")
fs.xgrid(ax)
ax.annotate("larger in human text", xy=(0.48, 1.004), xycoords="axes fraction",
            ha="right", va="bottom", fontsize=fs.BASE - 0.5, color=fs.GREY)
ax.annotate("larger in AI text", xy=(0.52, 1.004), xycoords="axes fraction",
            ha="left", va="bottom", fontsize=fs.BASE - 0.5, color=fs.GREY)
handles = [
    Line2D([], [], marker="o", ls="", color=CONS_C, ms=4, label="consistent"),
    Line2D([], [], marker="o", ls="", color=VAR_C, ms=4, label="genre-dependent"),
    Line2D([], [], marker="o", ls="", color="#a8a8a8", ms=3.5, label="negligible"),
    Line2D([], [], marker="D", ls="", mfc="white", mec="#111111", ms=4.5,
           label="macro-average"),
]
fig.legend(handles=handles, loc="outside upper center", ncol=4)
fs.save(fig, OUT + "fig_feature_effects.pdf")

# ----------------------------------------------- Fig 3: distributions
SEL_F = ["flesch_reading_ease", "word_count", "noun_ratio", "sentiment"]
SEL_G = ["nyt-articles", "xsum", "writingprompts", "reddit"]
SHORT_F = {"flesch_reading_ease": "Flesch", "word_count": "Words",
           "noun_ratio": "Noun ratio", "sentiment": "Sentiment"}
fig, axes = plt.subplots(len(SEL_F), len(SEL_G), figsize=fs.size(1.0, 4.8))
for r, feat in enumerate(SEL_F):
    row_top = 0
    dd = {}
    for c, g in enumerate(SEL_G):
        ax = axes[r, c]
        sub = hist[(hist.dataset == g) & (hist.feature == feat)]
        for grp, col in [("human", HUMAN_C), ("ALL_AI", AI_C)]:
            row = sub[sub.group == grp]
            if row.empty:
                continue
            row = row.iloc[0]
            cnt = np.array([int(x) for x in row["counts"].split(";")], float)
            dens = cnt / cnt.sum()
            dd[(c, grp)] = dens
            edges = np.linspace(row["lo"], row["hi"], len(cnt) + 1)
            ctr = (edges[:-1] + edges[1:]) / 2
            ax.fill_between(ctr, dens, step="mid", alpha=0.45, color=col, lw=0)
            ax.plot(ctr, dens, drawstyle="steps-mid", color=col, lw=0.9)
            row_top = max(row_top, dens[1:-1].max())
        ax.tick_params(labelsize=fs.BASE - 1.5)
        if r == 0:
            ax.set_title(LBL[g])
    # one vertical scale per row, so peak heights compare across genres
    top = row_top * 1.12
    for c in range(len(SEL_G)):
        ax = axes[r, c]
        ax.set_ylim(0, top)
        # a boundary bin taller than the scale is cut off; print its share
        for j, (grp, col) in enumerate([("human", HUMAN_C), ("ALL_AI", AI_C)]):
            d = dd.get((c, grp))
            if d is None:
                continue
            for k, x, ha in ((0, 0.04, "left"), (-1, 0.96, "right")):
                if d[k] > top:
                    ax.text(x, 0.95 - 0.16 * j, "%d%%" % round(100 * d[k]),
                            transform=ax.transAxes, ha=ha, va="top", color=col,
                            fontsize=fs.BASE - 2)
        if c == 0:
            ax.set_yticks([0, top])
            ax.set_yticklabels(["0", "%.2f" % top])
            ax.set_ylabel(SHORT_F[feat], fontsize=fs.BASE - 0.5)
        else:
            ax.set_yticks([0, top])
            ax.tick_params(axis="y", labelleft=False)
h = [plt.Rectangle((0, 0), 1, 1, color=HUMAN_C, alpha=0.6),
     plt.Rectangle((0, 0), 1, 1, color=AI_C, alpha=0.6)]
fig.legend(h, ["Human-written", "AI-generated"], ncol=2,
           loc="outside lower center")
fig.supylabel("Share of samples per bin", fontsize=fs.BASE)
fs.save(fig, OUT + "fig_hist_grid.pdf")

# ----------------------------------------------- Fig 4: style vs length
cells = per_model.pivot_table(index=["model", "dataset"], columns="feature",
                              values="cliffs_delta")
sty = sum(cells[f] * s for f, s in STYLE.items()) / len(STYLE)
lng = cells["word_count"]
gsty = sum(piv.loc[f] * s for f, s in STYLE.items()) / len(STYLE)
glng = piv.loc["word_count"]
fig, ax = plt.subplots(figsize=fs.size(1.0, 3.3))
ax.axhspan(-NEG, NEG, color="#f2f2f2", zorder=0, lw=0)
ax.axvspan(-NEG, NEG, color="#f2f2f2", zorder=0, lw=0)
ax.axhline(0, color="#555555", lw=0.8, zorder=1)
ax.axvline(0, color="#555555", lw=0.8, zorder=1)
ax.scatter(lng, sty, s=9, color="#9a9a9a", alpha=0.6, linewidths=0, zorder=2)
ax.scatter(glng, gsty, s=34, color="#111111", zorder=3, linewidths=0)
OFF = {"blogs": (6, -7), "essays": (6, 3), "natural-questions": (6, -9),
       "nyt-articles": (-6, 5), "nyt-comments": (4, 9), "raid": (-6, 5),
       "reddit": (6, 3), "tweets": (6, -8), "writingprompts": (-6, -6),
       "xsum": (-6, 4)}
for g in GEN:
    dx, dy = OFF[g]
    ax.annotate(LBL[g], (glng[g], gsty[g]), xytext=(dx, dy),
                textcoords="offset points", fontsize=fs.BASE - 1,
                ha="left" if dx > 0 else "right", va="center")
ax.set_xlim(-1.03, 1.03)
ax.set_ylim(-0.45, 1.0)
ax.set_xlabel("Length effect: Cliff's $\\delta$ of word count")
ax.set_ylabel("Style effect: mean signed $\\delta$")
fs.xgrid(ax)
fs.ygrid(ax)
# the human baseline is the origin; say so, and say which way each axis points
NOTE = dict(fontsize=fs.BASE - 1, color=fs.GREY)
ax.annotate("human baseline\n(no difference)", xy=(0, 0), xytext=(-0.62, -0.33),
            ha="center", va="center", arrowprops=dict(arrowstyle="-|>", color=fs.GREY,
                                                      lw=0.7, shrinkB=2), **NOTE)
ax.text(-0.04, 0.965, "← AI shorter", ha="right", va="center", **NOTE)
ax.text(0.04, 0.965, "AI longer →", ha="left", va="center", **NOTE)
ax.text(-1.0, 0.03, "↑ AI harder to read", ha="left", va="bottom", **NOTE)
ax.text(-1.0, -0.03, "↓ AI easier to read", ha="left", va="top", **NOTE)
hh = [Line2D([], [], marker="o", ls="", color="#9a9a9a", ms=3.5,
             label="one model in one genre"),
      Line2D([], [], marker="o", ls="", color="#111111", ms=5,
             label="all 21 models combined")]
ax.legend(handles=hh, loc="lower right")
fs.save(fig, OUT + "fig_style_length.pdf")

# ----------------------------------------------- Fig 5: feature x genre heatmap
fig, ax = plt.subplots(figsize=fs.size(1.0, 4.9))
M = piv.loc[order, GORD].values
im = ax.imshow(M, cmap=fs.DIVERGING, vmin=-1, vmax=1, aspect="auto")
for i, j in np.ndindex(M.shape):
    v = 0.0 if abs(M[i, j]) < 0.005 else M[i, j]
    ax.text(j, i, ("%.2f" % v).replace("-", "−"), ha="center",
            va="center", fontsize=fs.BASE - 2,
            color="white" if abs(M[i, j]) > 0.6 else "#222222")
ax.set_xticks(range(len(GORD)))
ax.set_xticklabels([LBL[g] for g in GORD], rotation=45, ha="right")
ax.set_yticks(range(len(order)))
ax.set_yticklabels(FEATL)
fs.bare(ax)
cb = fig.colorbar(im, ax=ax, shrink=0.42, pad=0.02, ticks=[-1, -0.5, 0, 0.5, 1])
cb.set_label("Cliff's $\\delta$")
cb.outline.set_visible(False)
fs.save(fig, OUT + "fig_delta_heatmap.pdf")

# ----------------------------------------------- Fig 6: genre effects + length
gd = piv.abs().mean(axis=0).loc[GORD]
hw = hum[hum.feature == "word_count"].set_index("dataset")["median"]
aw = allai[allai.feature == "word_count"].set_index("dataset")["median"]
fig, axes = plt.subplots(1, 2, figsize=fs.size(1.0, 2.9))
a = axes[0]
a.barh(range(len(gd)), gd.values, color=fs.GREY, height=0.66)
a.set_yticks(range(len(gd)))
a.set_yticklabels([LBL[g] for g in gd.index])
a.invert_yaxis()
a.set_xlabel("Mean $|\\delta|$ over 16 features")
fs.xgrid(a)
a.set_xlim(0, gd.max() * 1.08)
a.set_title("(a) difference per genre", loc="left", pad=4)
b = axes[1]
gs = gd.index.tolist()
for i, g in enumerate(gs):
    b.plot([hw[g], aw[g]], [i, i], color="#c4c4c4", lw=1.2, zorder=1)
b.scatter([hw[g] for g in gs], range(len(gs)), s=22, color=HUMAN_C, zorder=3,
          label="Human-written")
b.scatter([aw[g] for g in gs], range(len(gs)), s=22, color=AI_C, zorder=3,
          label="AI-generated")
b.set_yticks(range(len(gs)))
b.set_yticklabels([])
b.tick_params(axis="y", length=0)
b.invert_yaxis()
b.set_xscale("log")
b.set_xticks([20, 50, 100, 200, 500])
b.set_xticklabels(["20", "50", "100", "200", "500"])
b.minorticks_off()
b.set_xlabel("Median length (words)")
fs.xgrid(b)
b.legend(loc="lower right")
b.set_title("(b) median length", loc="left", pad=4)
fs.save(fig, OUT + "fig_genre_effects.pdf")

# ----------------------------------------------- Fig 7: sentiment
hs = sent[sent.source == "human"].set_index("dataset")
asent = sent[sent.source != "human"]
pool = asent.assign(w=asent.share_pos * asent.n).groupby("dataset").apply(
    lambda d: d["w"].sum() / d["n"].sum(), include_groups=False)
fig, ax = plt.subplots(figsize=fs.size(1.0, 2.5))
x = np.arange(len(GEN))
w = 0.38
ax.bar(x - w / 2, [hs.loc[g, "share_pos"] for g in GEN], w, color=HUMAN_C,
       label="Human-written")
ax.bar(x + w / 2, [pool[g] for g in GEN], w, color=AI_C, label="AI-generated")
ax.set_xticks(x)
ax.set_xticklabels([LBL[g] for g in GEN], rotation=45, ha="right")
ax.set_ylabel("Share positive")
ax.set_ylim(0, 1)
ax.legend(ncol=2, loc="lower left", bbox_to_anchor=(0.0, 1.0))
fs.ygrid(ax)
fs.save(fig, OUT + "fig_sentiment.pdf")

# ----------------------------------- Fig 8: model ranking and generation pairs
SHORT = {"Meta-Llama-3.1-70B-Instruct-AWQ-INT4": "Llama-3.1-70B",
         "Meta-Llama-3.3-70B-Instruct-AWQ-INT4": "Llama-3.3-70B",
         "Llama-3.1-8B-Instruct": "Llama-3.1-8B",
         "Llama-3.2-3B-Instruct": "Llama-3.2-3B",
         "Phi-3-mini-128k-instruct": "Phi-3-mini",
         "Phi-3-small-128k-instruct": "Phi-3-small",
         "Phi-3-medium-128k-instruct": "Phi-3-medium",
         "Phi-3.5-mini-instruct": "Phi-3.5-mini",
         "Phi-4-mini-instruct": "Phi-4-mini", "phi-4": "Phi-4",
         "Ministral-8B-Instruct-2410": "Ministral-8B",
         "Mistral-Nemo-Instruct-2407": "Mistral-Nemo",
         "Qwen2-7B-Instruct": "Qwen2-7B",
         "Qwen2-72B-Instruct-AWQ": "Qwen2-72B",
         "Qwen2.5-3B-Instruct": "Qwen2.5-3B",
         "Qwen2.5-7B-Instruct": "Qwen2.5-7B",
         "Qwen2.5-14B-Instruct": "Qwen2.5-14B",
         "Qwen2.5-72B-Instruct-AWQ": "Qwen2.5-72B",
         "Falcon3-3B-Instruct": "Falcon3-3B",
         "Falcon3-7B-Instruct": "Falcon3-7B",
         "gpt-4.1-nano-2025-04-14": "gpt-4.1-nano"}

mm = (per_model.groupby(["model", "family"])["cliffs_delta"]
      .apply(lambda s: s.abs().mean()).reset_index(name="mad")
      .sort_values("mad").reset_index(drop=True))
# each model on the two components of Section 5.2, averaged over genres:
# the style effect (signed) and the absolute word-count effect
mod = pd.DataFrame({"style": sty, "length": lng.abs()}).groupby(level="model").mean()
fam_of = per_model.drop_duplicates("model").set_index("model")["family"]
PAIRS = [("Qwen2-7B-Instruct", "Qwen2.5-7B-Instruct"),
         ("Qwen2-72B-Instruct-AWQ", "Qwen2.5-72B-Instruct-AWQ"),
         ("Phi-3-mini-128k-instruct", "Phi-3.5-mini-instruct"),
         ("Phi-3.5-mini-instruct", "Phi-4-mini-instruct"),
         ("Phi-3-medium-128k-instruct", "phi-4"),
         ("Meta-Llama-3.1-70B-Instruct-AWQ-INT4",
          "Meta-Llama-3.3-70B-Instruct-AWQ-INT4")]
# label positions in data coordinates, placed by hand; labels of the dense
# centre sit in the free space beside it and are joined to their point by a
# thin leader line (last field True)
LPOS = {"Qwen2.5-7B-Instruct": (0.207, 0.280, "right", False),
        "Qwen2.5-72B-Instruct-AWQ": (0.251, 0.268, "center", False),
        "Qwen2.5-3B-Instruct": (0.290, 0.283, "left", False),
        "Phi-4-mini-instruct": (0.258, 0.326, "center", False),
        "Qwen2.5-14B-Instruct": (0.232, 0.366, "right", True),
        "Meta-Llama-3.1-70B-Instruct-AWQ-INT4": (0.232, 0.390, "right", True),
        "phi-4": (0.232, 0.414, "right", True),
        "Meta-Llama-3.3-70B-Instruct-AWQ-INT4": (0.232, 0.438, "right", True),
        "Phi-3-mini-128k-instruct": (0.327, 0.399, "center", False),
        "gpt-4.1-nano-2025-04-14": (0.407, 0.372, "left", True),
        "Llama-3.2-3B-Instruct": (0.407, 0.350, "left", True),
        "Qwen2-72B-Instruct-AWQ": (0.407, 0.328, "left", True),
        "Llama-3.1-8B-Instruct": (0.407, 0.306, "left", True),
        "Phi-3-small-128k-instruct": (0.393, 0.426, "center", False),
        "Phi-3-medium-128k-instruct": (0.450, 0.400, "left", False),
        "Qwen2-7B-Instruct": (0.342, 0.483, "left", False),
        "Falcon3-3B-Instruct": (0.319, 0.492, "right", False),
        "Falcon3-7B-Instruct": (0.262, 0.512, "left", False),
        "Phi-3.5-mini-instruct": (0.463, 0.593, "right", False),
        "Ministral-8B-Instruct-2410": (0.349, 0.152, "right", False),
        "Mistral-Nemo-Instruct-2407": (0.369, 0.178, "left", False)}
fig, ax = plt.subplots(figsize=fs.size(1.0, 4.8))
for o, n in PAIRS:
    ax.annotate("", xy=(mod.loc[n, "length"], mod.loc[n, "style"]),
                xytext=(mod.loc[o, "length"], mod.loc[o, "style"]),
                arrowprops=dict(arrowstyle="-|>", color=FAM_C[fam_of[n]], lw=1.1,
                                shrinkA=4, shrinkB=4.5), zorder=2)
for fam_ in FAM_ORDER:
    sub = mod[fam_of.loc[mod.index] == fam_]
    ax.scatter(sub["length"], sub["style"], s=30, color=FAM_C[fam_], label=fam_,
               zorder=3, linewidths=0)
for m, r in mod.iterrows():
    lx, ly, ha, lead = LPOS[m]
    ax.text(lx, ly, SHORT.get(m, m), fontsize=fs.BASE - 1.5, ha=ha,
            va="center", zorder=4)
    if lead:
        x0 = lx + (0.002 if ha == "right" else -0.002)
        ax.annotate("", xy=(r["length"], r["style"]), xytext=(x0, ly),
                    arrowprops=dict(arrowstyle="-", color="#9a9a9a", lw=0.5,
                                    shrinkA=0, shrinkB=3), zorder=1)
ax.set_xlim(0.17, 0.52)
ax.set_ylim(0.10, 0.63)
ax.set_xlabel("Length effect: mean $|\\delta|$ of word count over genres")
ax.set_ylabel("Style effect: mean over genres")
fs.xgrid(ax)
fs.ygrid(ax)
fh = [Line2D([], [], marker="o", ls="", color=FAM_C[f], ms=4.5, label=f)
      for f in FAM_ORDER]
fh.append(Line2D([], [], color="#555555", lw=1.1, marker=">", ms=4,
                 label="older to newer"))
ax.legend(handles=fh, ncol=2, loc="upper left", columnspacing=0.9)
fs.save(fig, OUT + "fig_model_dots.pdf")

# ----------------------------------------------- Fig 9: model x genre
cell = (per_model.groupby(["model", "dataset"])["cliffs_delta"]
        .apply(lambda s: s.abs().mean()).unstack()[GORD])
cell = cell.loc[mm.set_index("model").index]
fig, ax = plt.subplots(figsize=fs.size(1.0, 4.0))
im = ax.imshow(cell.values, cmap="Reds", vmin=0, vmax=cell.values.max(),
               aspect="auto")
ax.set_xticks(range(len(GORD)))
ax.set_xticklabels([LBL[g] for g in GORD], rotation=45, ha="right")
ax.set_yticks(range(len(cell)))
ax.set_yticklabels([SHORT.get(m, m) for m in cell.index])
fs.bare(ax)
cb = fig.colorbar(im, ax=ax, shrink=0.75, pad=0.02)
cb.set_label("Mean $|\\delta|$")
cb.outline.set_visible(False)
fs.save(fig, OUT + "fig_model_genre.pdf")

print("8 figures written to", OUT)
