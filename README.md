# Human-written vs AI-generated text: style and length

Code and data for the article

> M. Gromadzki, A. Wróblewska and A. Kaliska. *Consistent in Style, Variable in Length: How AI-Generated Text Differs from Human Writing Across Ten Genres and 21 Language Models.*

The article compares 6.8 million human-written texts from ten genres with 13.5 million texts generated from the same sources by 21 instruction-tuned language models, using Cliff's delta on 16 interpretable features.

## Contents

| Path | What it is |
|---|---|
| `data/effects.csv` | Cliff's delta, AUC and Mann-Whitney p-value of every feature for every genre and model, plus the mean, standard deviation and quantiles of the AI-generated texts. `model = ALL_AI` means all 21 models combined. |
| `data/human_stats.csv` | Mean, standard deviation and quantiles of every feature for the human-written texts of each genre. |
| `data/hist.csv` | Histograms (60 equal-width bins, tails clipped into the end bins) of every feature for the human texts, each model and all models combined. |
| `data/sentiment.csv` | Share of texts with VADER compound above 0.5, below -0.5 and within 0.05 of 0, per genre and source. |
| `analysis/paper_numbers.py` | Prints every number quoted in the article, section by section. |
| `analysis/make_figures.py` | Draws the figures of the article into `figures/` (shared style in `figstyle.py`). |
| `analysis/aggregate.py` | Builds the four tables in `data/` from per-text features. |
| `features/extract_features.py` | Computes the per-text features from the released corpus. |

Every number and figure in the article is computed from the four tables in `data/` (2.6 MB).

## Reproducing the results

From the released tables, which takes seconds:

```bash
pip install -r requirements.txt
python analysis/paper_numbers.py
python analysis/make_figures.py
```

From the corpus, which reruns the whole pipeline:

1. Download the corpus from Hugging Face, [Majkel1337/Detect-AI](https://huggingface.co/datasets/Majkel1337/Detect-AI) (20.2 million texts, parquet).
2. Install the spaCy model with `python -m spacy download en_core_web_sm`. The NLTK data (punkt, stopwords, VADER lexicon, tagger) are downloaded by the script.
3. Compute the features, one CSV per genre and source:
   ```bash
   python features/extract_features.py --corpus path/to/Detect-AI --out features_out --workers 32
   ```
4. Build the tables:
   ```bash
   python analysis/aggregate.py --features features_out --out data_new
   ```

Feature extraction over the full corpus is slow; for the article it was run on a computing cluster. The aggregation subsamples at most 200,000 human and 300,000 AI-generated texts per comparison with a fixed seed.

## Features

The extraction script computes 36 features per text. The article reports the following 16, and the tables contain all 36.

| Category | Features (column names) |
|---|---|
| Length | `word_count` |
| Readability | `flesch_reading_ease`, `smog_index`, `dale_chall_readability` |
| Lexical composition | `noun_ratio`, `verb_ratio`, `stopword_ratio` |
| Lexical diversity | `Mass` (Maas index), `DMetric` (Simpson's D) |
| Syntax | `std_sentence_length`, `dependency_distance`, `passive_voice_count` |
| Cohesion and style markers | `pronoun_count`, `contraction_count`, `discourse_marker_ratio` |
| Sentiment | `sentiment` (VADER compound score) |

## Corpus

The corpus, and the code that generated it from the human sources, belong to an earlier project. See [llm-ai-text-detection](https://github.com/Michal1337/llm-ai-text-detection) for the prompt templates, the generation pipeline and the detection experiments.

## License

MIT, the same as the corpus.
