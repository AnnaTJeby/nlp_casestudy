# Economic News Article Classification — NLP Case Study

Text classification pipeline that labels news articles as **Relevant** or
**Not Relevant** to the US economy, using Bag-of-Words features and
Naive Bayes / Logistic Regression / LinearSVC classifiers.

Dataset: `US-Economic-News.csv` — ~8,000 articles (~1,420 Relevant /
~6,580 Not Relevant).

## Team & Task Split
See `Task_Allocation.docx` for the full breakdown — happy to adjust if
something doesn't fit anyone's schedule. Summary:

| # | Role | 
|---|---|
| 1 | Data prep & preprocessing |
| 2 | Feature extraction (BoW) |
| 3 | Naive Bayes classifier | 
| 4 | Logistic Regression classifier | 
| 5 | LinearSVC classifier (plain + balanced) | 
| 6 | Class imbalance analysis | 
| 7 | Model comparison | 
| 8 | Improvement experiments |
| 9 | Final table, write-up, repo assembly | 

## For Members 2–9
To keep everyone's results comparable, it'd help if we all build on the
same cleaned data rather than each re-loading and re-cleaning the raw CSV
separately. `data/train_split.csv` and `data/test_split.csv` already
have the stratified 80/20 split (6,400 train / 1,600 test) ready to go —
feel free to just load those directly for your part.

Each file has two columns: `clean_text` and `label` (`Relevant` /
`Not Relevant`).

## Repo structure
All notebooks live in `notebooks/` and assume they're run from there
(`uv run jupyter lab`, then open a notebook inside `notebooks/`) — every
data/results path is written relative to that folder (`../data`, `../results`).

```
nlp_casestudy/
├── data/                              # shared train/test split (Member 1's output)
│   ├── train_split.csv
│   └── test_split.csv
├── results/                           # metric CSVs each notebook saves for the next member to consume
│   ├── feature_extraction_summary.csv     (Member 2)
│   ├── naive_bayes_results.csv            (Member 3)
│   ├── member5_linearsvc_results.csv      (Member 5)
│   ├── model_comparison_results.csv       (Member 7)
│   ├── improvement_results.csv            (Member 8)
│   ├── improvement_summary.csv            (Member 8)
│   ├── tfidf_ngram_results.csv             (clean-text TF-IDF experiment)
│   ├── tfidf_ngram_cv_results.csv          (clean-text CV model selection)
│   ├── tfidf_multitext_results.csv         (headline + body TF-IDF experiment)
│   └── tfidf_multitext_cv_results.csv      (multi-text CV model selection)
├── notebooks/
│   ├── 1_data_prep_and_split.ipynb        (Member 1)
│   ├── 2_feature_extraction.ipynb         (Member 2)
│   ├── 3_naive_bayes.ipynb                (Member 3)
│   ├── 4_logistic_regression.ipynb        (Member 4)
│   ├── 5_linearsvc.ipynb                  (Member 5)
│   ├── 6_imbalance_analysis.ipynb         (Member 6)
│   ├── 7_model_comparison.ipynb           (Member 7)
│   └── 8_improvement_experiments.ipynb    (Member 8)
├── experiments/
│   └── tfidf_ngram_multitext.py            (separate TF-IDF ablation experiment)
├── pyproject.toml / uv.lock            # dependencies (`uv sync`)
└── README.md
```

Note: `6_imbalance_analysis.ipynb` is Colab-specific — it loads Member 5's
results via a manual `files.upload()` prompt rather than reading from
`results/` directly.

## Separate TF-IDF + multi-text experiment

The baseline notebooks and their result files remain unchanged. The separate
experiment compares TF-IDF word n-grams and combined word/character n-grams
against the saved best baseline:

```bash
uv run python experiments/tfidf_ngram_multitext.py --mode clean
```

This writes `results/tfidf_ngram_results.csv` and a CV-only selection table.
Model and representation selection uses stratified cross-validation on the
training split; the test split is evaluated only after selection.

To include every available raw text field (for example, headline and article
body), place `US-Economic-News.csv` in `data/` and run:

```bash
uv run python experiments/tfidf_ngram_multitext.py --mode multitext
```

The script reconstructs the original seeded split and refuses to continue if
its label order differs from the saved baseline split. This prevents an unfair
comparison caused by evaluating the new representation on different examples.

The dataset contains `headline` and `text`, and the verified multi-text run
selected combined word/character TF-IDF with balanced Logistic Regression.
It reached 0.4690 Relevant-class F1, compared with 0.4457 for the best original
Bag-of-Words baseline. Full metrics are in
`results/tfidf_multitext_results.csv`.
