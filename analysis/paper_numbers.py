"""Print every number quoted in the paper, computed from the tables in data/.

Sections follow the paper. Run: python paper_numbers.py
"""
import os
import sys
from math import atanh, sqrt, tanh

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
eff = pd.read_csv(os.path.join(DATA, "effects.csv"))
hum = pd.read_csv(os.path.join(DATA, "human_stats.csv"))
hist = pd.read_csv(os.path.join(DATA, "hist.csv"))
sent = pd.read_csv(os.path.join(DATA, "sentiment.csv"))

NEG = 0.147
GEN = ["blogs", "essays", "natural-questions", "nyt-articles", "nyt-comments",
       "raid", "reddit", "tweets", "writingprompts", "xsum"]
NO_NYT = [g for g in GEN if g != "nyt-articles"]
# the 16 features reported in the paper
FEAT = ["flesch_reading_ease", "smog_index", "dale_chall_readability", "noun_ratio",
        "word_count", "std_sentence_length", "Mass", "DMetric", "sentiment",
        "pronoun_count", "stopword_ratio", "verb_ratio", "contraction_count",
        "passive_voice_count", "dependency_distance", "discourse_marker_ratio"]
# style effect: Flesch Reading Ease (sign reversed), Dale-Chall, noun ratio
STYLE = {"flesch_reading_ease": -1, "dale_chall_readability": 1, "noun_ratio": 1}
TASK = {"blogs": "rewrite", "essays": "rewrite", "nyt-comments": "rewrite", "reddit": "rewrite",
        "tweets": "rewrite", "nyt-articles": "from attributes", "raid": "from attributes",
        "writingprompts": "from attributes", "xsum": "from attributes",
        "natural-questions": "context-grounded"}

eff = eff[eff.feature.isin(FEAT)]
comb = eff[eff.model == "ALL_AI"]                     # all 21 models combined
per = eff[eff.model != "ALL_AI"]                      # one model at a time
P = comb.pivot_table(index="feature", columns="dataset", values="cliffs_delta").loc[FEAT, GEN]
M = per.pivot_table(index=["model", "dataset"], columns="feature", values="cliffs_delta")[FEAT]
fam_of = per.drop_duplicates("model").set_index("model")["family"]
hmed = hum[hum.feature == "word_count"].set_index("dataset")["median"].loc[GEN]


def style(X):
    return sum(X[f] * s for f, s in STYLE.items()) / len(STYLE)


def spear(a, b):
    return pd.Series(a).rank().corr(pd.Series(b).rank())


def head(t):
    print("\n" + "=" * 78 + "\n" + t + "\n" + "=" * 78)


def genre_dependent(Px, thr, gens):
    out = []
    for f in Px.index:
        s = Px.loc[f, gens]
        s = s[s.abs() >= thr]
        if (s > 0).any() and (s < 0).any():
            out.append(f)
    return out


# ----------------------------------------------------------------------------
head("Section 4.3-4.4 and Table A.1: summaries per feature")
macro, mad = P.mean(axis=1), P.abs().mean(axis=1)
gd = genre_dependent(P, NEG, GEN)
for f in mad.sort_values(ascending=False).index:
    nn = int((P.loc[f].abs() >= NEG).sum())
    cls = "genre-dependent" if f in gd else "consistent"
    print("  %-24s macro %+.2f  mean|d| %.2f  non-negligible in %2d  %s" % (f, macro[f], mad[f], nn, cls))
print("genre-dependent features: %d" % len(gd))
print("threshold 0.25 -> %d, 0.10 -> %d" % (len(genre_dependent(P, 0.25, GEN)), len(genre_dependent(P, 0.10, GEN))))
loo = {g: genre_dependent(P, NEG, [x for x in GEN if x != g]) for g in GEN}
print("leave one genre out: " + ", ".join("%s %d" % (g, len(v)) for g, v in loo.items()))
variants = [set(genre_dependent(P, t, GEN)) for t in (0.25, NEG, 0.10)] + [set(v) for v in loo.values()]
print("genre-dependent under every variant: %s" % sorted(set.intersection(*variants)))
hidden = [f for f in gd if abs(macro[f]) < NEG]
print("genre-dependent with a negligible macro-average: %s" % hidden)

head("Section 4.3: total variation distance versus |delta|")
d = comb.set_index(["dataset", "feature"])["cliffs_delta"]
rows = []
for (g, f), grp in hist[hist.feature.isin(FEAT)].groupby(["dataset", "feature"]):
    h, a = grp[grp.group == "human"], grp[grp.group == "ALL_AI"]
    if h.empty or a.empty:
        continue
    ph = np.array([float(x) for x in h.iloc[0]["counts"].split(";")])
    pa = np.array([float(x) for x in a.iloc[0]["counts"].split(";")])
    ph, pa = ph / ph.sum(), pa / pa.sum()
    rows.append((g, f, d.loc[(g, f)], 0.5 * np.abs(ph - pa).sum(), np.minimum(ph, pa).sum()))
TV = pd.DataFrame(rows, columns=["genre", "feature", "delta", "tv", "overlap"])
print("pairs with histograms: %d   Spearman(|delta|, TV) %.2f" % (len(TV), TV.delta.abs().corr(TV.tv, method="spearman")))
print(TV[(TV.delta.abs() < NEG) & (TV.tv > 0.25)].round(3).to_string(index=False))

# ----------------------------------------------------------------------------
head("Section 5.1: features and distributions")
print("XSum median Flesch: human %.1f, all models %.1f" % (
    hum[(hum.dataset == "xsum") & (hum.feature == "flesch_reading_ease")]["median"].iloc[0],
    comb[(comb.dataset == "xsum") & (comb.feature == "flesch_reading_ease")]["median"].iloc[0]))
smog0 = hum[hum.feature == "smog_index"].set_index("dataset")["median"]
print("genres where at least half of the human texts have SMOG 0: %s" % list(smog0[smog0 == 0].index))
for f, g in [("discourse_marker_ratio", "nyt-articles")]:
    for grp in ["human", "ALL_AI"]:
        r = hist[(hist.dataset == g) & (hist.feature == f) & (hist.group == grp)].iloc[0]
        c = np.array([float(x) for x in r["counts"].split(";")])
        print("%s %s: share with a discourse marker %.3f" % (g, grp, 1 - c[0] / c.sum()))
for g in ["nyt-articles", "xsum", "writingprompts", "reddit"]:
    print("overlap in %-15s %s" % (g, ", ".join("%s %.2f" % (f, o) for f, o in
          TV[(TV.genre == g) & TV.feature.isin(["flesch_reading_ease", "word_count", "noun_ratio", "sentiment"])][["feature", "overlap"]].values)))

# ----------------------------------------------------------------------------
head("Section 5.2: style and length")
S, L = style(M), M["word_count"]
print("style effect: positive in %d of %d pairs, positive and non-negligible in %d" % (
    (S > 0).sum(), len(S), (S >= NEG).sum()))
neg = S[S < 0]
print("negative pairs by genre: %s" % neg.groupby(level="dataset").size().to_dict())
print("negative pairs outside natural-questions: %s" % [i for i in neg.index if i[1] != "natural-questions"])
ms = S.groupby(level="model").mean()
print("per-model style: %.2f (%s) to %.2f (%s); min genres positive per model %d" % (
    ms.min(), ms.idxmin(), ms.max(), ms.idxmax(), (S > 0).groupby(level="model").sum().min()))
SP = style(P.T)
print("style with all models combined: %s" % SP.round(2).to_dict())
print("length with all models combined: %s" % P.loc["word_count"].round(2).to_dict())
print("Spearman(length effect, human median length) %.2f, without NYT Articles %.2f" % (
    spear(P.loc["word_count"], hmed), spear(P.loc["word_count", NO_NYT], hmed[NO_NYT])))
W = per[per.feature == "word_count"].pivot_table(index="model", columns="dataset", values="cliffs_delta")[GEN]
for g in GEN:
    s = np.sign(P.loc["word_count", g])
    print("  %-18s combined %+.2f  models with the same sign %2d/21  range %+.2f..%+.2f" % (
        g, P.loc["word_count", g], (np.sign(W[g]) == s).sum(), W[g].min(), W[g].max()))
cells_no = [i for i in S.index if i[1] != "nyt-articles"]
print("r(style, length) over pairs %.2f, without NYT Articles %.2f" % (
    np.corrcoef(S, L)[0, 1], np.corrcoef(S[cells_no], L[cells_no])[0, 1]))
print("within-genre r(style, length) across models: %s" % {
    g: round(np.corrcoef(S.xs(g, level="dataset"), L.xs(g, level="dataset"))[0, 1], 2) for g in GEN})
for g in ["xsum", "writingprompts", "raid"]:
    w, s = L.xs(g, level="dataset"), S.xs(g, level="dataset")
    k = w.abs() < NEG
    print("%s: %d models with a negligible length effect, their style effects %.2f..%.2f" % (g, k.sum(), s[k].min(), s[k].max()))
print("per-genre correlation with the word-count effect, without NYT Articles (95% interval):")
for f in ["std_sentence_length", "pronoun_count", "contraction_count", "dependency_distance"]:
    r = np.corrcoef(P.loc[f, NO_NYT], P.loc["word_count", NO_NYT])[0, 1]
    z, se = atanh(r), 1 / sqrt(len(NO_NYT) - 3)
    print("  %-22s r %.2f  (%.2f..%.2f)" % (f, r, tanh(z - 1.96 * se), tanh(z + 1.96 * se)))
print("within-genre correlation with the word-count effect across models (median over 9 genres):")
for f in ["std_sentence_length", "pronoun_count", "contraction_count", "dependency_distance", "DMetric", "Mass", "sentiment"]:
    rs = [np.corrcoef(M[f].xs(g, level="dataset"), L.xs(g, level="dataset"))[0, 1] for g in NO_NYT]
    print("  %-22s %.2f" % (f, np.median(rs)))

# ----------------------------------------------------------------------------
head("Section 5.3: genres")
gm = P.abs().mean()
print("mean |delta| per genre: %s" % gm.round(2).to_dict())
print("spread %.1f-fold; without NYT Articles %.1f-fold (%.2f to %.2f)" % (
    gm.max() / gm.min(), gm[NO_NYT].max() / gm[NO_NYT].min(), gm[NO_NYT].min(), gm[NO_NYT].max()))
print("mean |delta| across features: %.2f to %.2f" % (mad.min(), mad.max()))


def decomp(A):
    tot = ((A - A.values.mean()) ** 2).values.sum()
    g = A.shape[0] * ((A.mean(axis=0) - A.values.mean()) ** 2).sum() / tot
    f = A.shape[1] * ((A.mean(axis=1) - A.values.mean()) ** 2).sum() / tot
    return g, f, 1 - g - f


print("variance of |delta|: genre %.0f%%, feature %.0f%%, interaction %.0f%%" % tuple(100 * x for x in decomp(P.abs())))
print("without NYT Articles: genre %.0f%%, feature %.0f%%" % tuple(100 * x for x in decomp(P[NO_NYT].abs())[:2]))
rest = [f for f in FEAT if f not in STYLE and f != "word_count"]
gr = P.loc[rest].abs().mean()
print("Spearman(genre mean |delta|, style) %.2f, (., |length|) %.2f; over the %d other features %.2f and %.2f" % (
    spear(gm, SP), spear(gm, P.loc["word_count"].abs()), len(rest), spear(gr, SP), spear(gr, P.loc["word_count"].abs())))
print("style by task type: %s" % {t: SP[[g for g in GEN if TASK[g] == t]].round(2).to_dict() for t in sorted(set(TASK.values()))})
for f, k in [("std_sentence_length", "single sentence"), ("Mass", "no repeated word")]:
    r = hist[(hist.dataset == "nyt-articles") & (hist.feature == f) & (hist.group == "human")].iloc[0]
    c = np.array([float(x) for x in r["counts"].split(";")])
    print("NYT Articles human abstracts with %s (first bin of %s): %.0f%%" % (k, f, 100 * c[0] / c.sum()))
print("medians of word count: %s" % {g: (hmed[g], comb[(comb.dataset == g) & (comb.feature == "word_count")]["median"].iloc[0]) for g in GEN})

# ----------------------------------------------------------------------------
head("Section 5.4-5.5: cancellation and sentiment")
print("SD of sentence length: macro %.2f, without NYT Articles %.2f" % (P.loc["std_sentence_length"].mean(), P.loc["std_sentence_length", NO_NYT].mean()))
for g in ["raid", "essays", "xsum", "writingprompts"]:
    v = M["std_sentence_length"].xs(g, level="dataset")
    print("  %-15s combined %+.2f, models negative %d/21" % (g, P.loc["std_sentence_length", g], (v < 0).sum()))
print("sentiment delta per genre: %s" % P.loc["sentiment"].round(2).to_dict())
print("sentiment pairs: non-negligibly positive %d, non-negligibly negative %d" % ((M["sentiment"] >= NEG).sum(), (M["sentiment"] <= -NEG).sum()))
rs = [np.corrcoef(M["sentiment"].xs(g, level="dataset"), L.xs(g, level="dataset"))[0, 1] for g in NO_NYT]
print("within-genre correlation of sentiment with length (median) %.2f" % np.median(rs))
for g in ["raid", "writingprompts"]:
    k = L.xs(g, level="dataset").abs() < NEG
    s = M["sentiment"].xs(g, level="dataset")[k]
    print("%s: sentiment of models with negligible length effect %.2f..%.2f" % (g, s.min(), s.max()))
hs = sent[sent.source == "human"].set_index("dataset")
ai = sent[sent.source != "human"]
share = ai.assign(w=ai.share_pos * ai.n).groupby("dataset").apply(lambda x: x.w.sum() / x.n.sum(), include_groups=False)
zero = ai.assign(w=ai.share_zero * ai.n).groupby("dataset").apply(lambda x: x.w.sum() / x.n.sum(), include_groups=False)
print("share with compound > 0.5, human -> AI: %s" % {g: (round(hs.loc[g, "share_pos"], 2), round(share[g], 2)) for g in GEN})
print("NYT Articles share with |compound| < 0.05: human %.3f, AI %.3f" % (hs.loc["nyt-articles", "share_zero"], zero["nyt-articles"]))

# ----------------------------------------------------------------------------
head("Section 6: models")
dist = per.groupby("model").cliffs_delta.apply(lambda s: s.abs().mean()).sort_values()
print("mean |delta| per model:")
for m, v in dist.items():
    print("  %-38s %.3f  style %.2f  |length| %.2f" % (m, v, ms[m], L.abs().xs(m, level="model").mean()))
print("families: %s" % per.assign(a=per.cliffs_delta.abs()).groupby("family").a.mean().round(3).sort_values().to_dict())
famF = per.pivot_table(index="feature", columns="family", values="cliffs_delta", aggfunc="mean").loc[FEAT]
nn = famF.abs() >= NEG
rows_nn = [f for f in FEAT if nn.loc[f].any()]
agree = [f for f in rows_nn if (famF.loc[f][nn.loc[f]] > 0).all() or (famF.loc[f][nn.loc[f]] < 0).all()]
print("features with a non-negligible family effect %d, families agree on %d, exception %s" % (
    len(rows_nn), len(agree), [f for f in rows_nn if f not in agree]))
allsign = [f for f in FEAT if (famF.loc[f] > 0).all() or (famF.loc[f] < 0).all()]
open_f = [c for c in famF.columns if c != "GPT"]
gpt_opp = [f for f in FEAT if len(set(np.sign(famF.loc[f, open_f]))) == 1 and np.sign(famF.loc[f, "GPT"]) != np.sign(famF.loc[f, open_f[0]])]
print("six family means share a sign on %d of 16; proprietary model opposite to all open families on %d" % (len(allsign), len(gpt_opp)))
famG = per.pivot_table(index=["feature", "dataset"], columns="family", values="cliffs_delta", aggfunc="mean").loc[FEAT]
nnG = famG.abs() >= NEG
cellsG = famG[nnG.any(axis=1)]
agreeG = [(famG.loc[i][nnG.loc[i]] > 0).all() or (famG.loc[i][nnG.loc[i]] < 0).all() for i in cellsG.index]
dis = [i for i, a in zip(cellsG.index, agreeG) if not a]
print("within genres: families agree in %d of %d feature-genre pairs; %d disagreements, %d involve Mass" % (
    sum(agreeG), len(cellsG), len(dis), sum(1 for i in dis if i[0] == "Mass")))
C = per.groupby(["model", "dataset"]).cliffs_delta.apply(lambda s: s.abs().mean()).unstack()[GEN]
print("largest genre per model: %s" % C.idxmax(axis=1).value_counts().to_dict())
print("smallest genre per model: %s" % C.idxmin(axis=1).value_counts().to_dict())
print("NYT Articles per model %.2f..%.2f" % (C["nyt-articles"].min(), C["nyt-articles"].max()))
gpt = "gpt-4.1-nano-2025-04-14"
print("gpt-4.1-nano: %s" % {g: (round(C.loc[gpt, g], 2), int(C[g].rank()[gpt])) for g in ["tweets", "reddit", "raid", "xsum", "nyt-articles"]})
LINES = {"Qwen2.5": ["Qwen2.5-3B-Instruct", "Qwen2.5-7B-Instruct", "Qwen2.5-14B-Instruct", "Qwen2.5-72B-Instruct-AWQ"],
         "Qwen2": ["Qwen2-7B-Instruct", "Qwen2-72B-Instruct-AWQ"],
         "Phi-3": ["Phi-3-mini-128k-instruct", "Phi-3-small-128k-instruct", "Phi-3-medium-128k-instruct"],
         "Llama-3.1": ["Llama-3.1-8B-Instruct", "Meta-Llama-3.1-70B-Instruct-AWQ-INT4"],
         "Falcon3": ["Falcon3-3B-Instruct", "Falcon3-7B-Instruct"]}
print("size lines (mean |delta|):")
for ln, mods in LINES.items():
    print("  %-10s %s" % (ln, "  ".join("%.3f" % dist[m] for m in mods)))
STEPS = [("Qwen2-7B-Instruct", "Qwen2.5-7B-Instruct"), ("Qwen2-72B-Instruct-AWQ", "Qwen2.5-72B-Instruct-AWQ"),
         ("Phi-3-mini-128k-instruct", "Phi-3.5-mini-instruct"), ("Phi-3.5-mini-instruct", "Phi-4-mini-instruct"),
         ("Phi-3-medium-128k-instruct", "phi-4"),
         ("Meta-Llama-3.1-70B-Instruct-AWQ-INT4", "Meta-Llama-3.3-70B-Instruct-AWQ-INT4")]
lm = L.abs().groupby(level="model").mean()
print("generation steps (mean |delta|, change in style, change in |length|):")
for a, b in STEPS:
    print("  %-38s -> %-38s %.3f -> %.3f  style %+.2f  length %+.2f" % (a, b, dist[a], dist[b], ms[b] - ms[a], lm[b] - lm[a]))
