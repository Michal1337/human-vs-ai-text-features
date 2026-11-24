"""One figure style for both articles.

Figures are authored at the exact text width of the elsarticle preprint layout
(390pt = 5.39in) so that \\includegraphics[width=\\textwidth] applies no scaling
and the type size on the page is the size set here.
"""
import matplotlib as mpl
import matplotlib.pyplot as plt

TEXTWIDTH = 5.39          # inches, = 390pt
BASE = 8.5                # pt; captions in the 12pt document are ~10pt

mpl.rcParams.update({
    "font.size": BASE,
    "axes.labelsize": BASE,
    "axes.titlesize": BASE + 0.5,
    "xtick.labelsize": BASE - 0.5,
    "ytick.labelsize": BASE - 0.5,
    "legend.fontsize": BASE - 0.5,
    "figure.titlesize": BASE + 0.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.axisbelow": True,
    "axes.linewidth": 0.7,
    "xtick.major.width": 0.7,
    "ytick.major.width": 0.7,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "lines.linewidth": 1.5,
    "lines.markersize": 4.5,
    "grid.color": "#e8e8e8",
    "grid.linewidth": 0.6,
    "legend.frameon": False,
    "legend.handletextpad": 0.4,
    "legend.columnspacing": 1.1,
    "legend.borderaxespad": 0.4,
    "pdf.fonttype": 42,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.01,
})

# Okabe-Ito, colourblind safe
HUMAN_C, AI_C = "#0072B2", "#D55E00"
CONS_C, VAR_C = "#117733", "#AA4499"
FAM_ORDER = ["Llama", "Phi", "Mistral", "Qwen", "Falcon", "GPT"]
FAM_C = {"Llama": "#E69F00", "Phi": "#CC79A7", "Mistral": "#009E73",
         "Qwen": "#56B4E9", "Falcon": "#999999", "GPT": "#000000"}
DIVERGING = "RdBu_r"
GREY = "#555555"

GEN = ["blogs", "essays", "natural-questions", "nyt-articles", "nyt-comments",
       "raid", "reddit", "tweets", "writingprompts", "xsum"]
LBL = {"blogs": "Blogs", "essays": "Essays", "natural-questions": "Nat. Questions",
       "nyt-articles": "NYT Articles", "nyt-comments": "NYT Comments",
       "raid": "RAID", "reddit": "Reddit", "tweets": "Tweets",
       "writingprompts": "WritingPrompts", "xsum": "XSum"}

# readable names for all 36 features
FEAT = {
    "flesch_reading_ease": "Flesch Reading Ease",
    "automated_readability_index": "Automated readability index",
    "gunning_fog_index": "Gunning fog index",
    "smog_index": "SMOG index",
    "dale_chall_readability": "Dale–Chall score",
    "average_word_length": "Average word length",
    "average_sentence_length": "Average sentence length",
    "std_sentence_length": "SD of sentence length",
    "word_count": "Word count",
    "character_count": "Character count",
    "sentence_count": "Sentence count",
    "punctuation_count": "Punctuation count",
    "TTR": "Type-token ratio",
    "RTTR": "Root type-token ratio",
    "CTTR": "Corrected type-token ratio",
    "Mass": "Maas index",
    "DMetric": "Simpson's D",
    "hapax_legomena_ratio": "Hapax legomena ratio",
    "entropy_score": "Word-frequency entropy",
    "bigram_repetition_ratio": "Bigram repetition",
    "trigram_repetition_ratio": "Trigram repetition",
    "noun_ratio": "Noun ratio",
    "verb_ratio": "Verb ratio",
    "adjective_ratio": "Adjective ratio",
    "pronoun_count": "Pronoun count",
    "conjunction_count": "Conjunction count",
    "stopword_ratio": "Stopword ratio",
    "contraction_count": "Contraction count",
    "present_participle_count": "Present participle count",
    "passive_voice_count": "Passive voice count",
    "entity_count": "Named entity count",
    "syntactic_depth": "Syntactic depth",
    "dependency_distance": "Dependency distance",
    "discourse_marker_count": "Discourse marker count",
    "discourse_marker_ratio": "Discourse marker ratio",
    "sentiment": "Sentiment",
}


def fl(f):
    return FEAT.get(f, f.replace("_", " "))


def size(frac=1.0, height=3.2):
    """Figure size: a fraction of the text width, by an explicit height."""
    return (TEXTWIDTH * frac, height)


def save(fig, path):
    fig.savefig(path)
    plt.close(fig)


def ygrid(ax):
    ax.grid(axis="y", color=mpl.rcParams["grid.color"],
            lw=mpl.rcParams["grid.linewidth"])


def xgrid(ax):
    ax.grid(axis="x", color=mpl.rcParams["grid.color"],
            lw=mpl.rcParams["grid.linewidth"])


def bare(ax):
    """Strip the frame entirely, for heatmaps."""
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
