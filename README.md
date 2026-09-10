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
separately. `outputs/train_split.csv` and `outputs/test_split.csv` already
have the stratified 80/20 split (6,400 train / 1,600 test) ready to go —
feel free to just load those directly for your part.

Each file has two columns: `clean_text` and `label` (`Relevant` /
`Not Relevant`).

## Repo structure
