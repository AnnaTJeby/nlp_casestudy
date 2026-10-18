# NLP Experiment Results Comparison

## Overall Performance

| Stage | Data / Features | Best Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---:|---:|---:|---:|
| Original baseline | Clean text, Bag-of-Words, 5,000 features | Multinomial Naive Bayes | 70.31% | 33.33% | **67.25%** | 44.57% |
| TF-IDF experiment | Clean text, word unigrams + bigrams | Balanced LinearSVC | **76.50%** | **39.00%** | 57.39% | 46.44% |
| Multi-text experiment | Headline + article, word + character TF-IDF | Balanced Logistic Regression | 75.94% | 38.55% | 59.86% | **46.90%** |

## Metric Changes

| Comparison | Accuracy Change | Precision Change | Recall Change | F1 Change |
|---|---:|---:|---:|---:|
| Baseline → clean-text TF-IDF | **+6.19 points** | **+5.67 points** | −9.86 points | **+1.87 points** |
| Clean-text TF-IDF → multi-text TF-IDF | −0.56 points | −0.45 points | **+2.46 points** | **+0.46 points** |
| Baseline → final multi-text model | **+5.63 points** | **+5.22 points** | −7.39 points | **+2.33 points** |

## Confusion Matrix Comparison

| Stage | True Negatives | False Positives | False Negatives | True Positives |
|---|---:|---:|---:|---:|
| Original baseline | 934 | 382 | **93** | **191** |
| Clean-text TF-IDF | **1,061** | **255** | 121 | 163 |
| Multi-text TF-IDF | 1,045 | 271 | 114 | 170 |
| Final change from baseline | +111 | **−111** | +21 | −21 |

## Clean-Text TF-IDF Models

| Model | Selected Representation | Accuracy | Precision | Recall | F1 |
|---|---|---:|---:|---:|---:|
| Complement Naive Bayes | Word TF-IDF | **79.13%** | **41.01%** | 40.14% | 40.57% |
| Logistic Regression | Word TF-IDF, balanced | 76.69% | 39.17% | 56.69% | 46.33% |
| LinearSVC | Word TF-IDF, balanced | 76.50% | 39.00% | **57.39%** | **46.44%** |

## Multi-Text TF-IDF Models

| Model | Selected Representation | Accuracy | Precision | Recall | F1 |
|---|---|---:|---:|---:|---:|
| Complement Naive Bayes | Headline + article, word + character TF-IDF | **78.81%** | **41.27%** | 45.77% | 43.41% |
| Logistic Regression | Headline + article, word + character TF-IDF, balanced | 75.94% | 38.55% | 59.86% | **46.90%** |
| LinearSVC | Headline + article, word + character TF-IDF, balanced | 75.63% | 38.17% | **60.21%** | 46.72% |

## Experiment Changes

| Change | Result |
|---|---|
| Raw counts → TF-IDF | Reduced the influence of very frequent words |
| Unigrams → unigrams + bigrams | Added phrases such as “interest rates” and “federal reserve” |
| Article only → headline + article | Recovered seven additional Relevant articles |
| Word features → word + character features | Improved multi-text cross-validation performance |
| Unbalanced → balanced classifiers | Increased attention to the minority Relevant class |
| Fixed model settings → five-fold CV | Selected hyperparameters using training data only |
| Original test split retained | Ensured a fair comparison with the baseline |

## Recommendations

| Decision Requirement | Recommended Model |
|---|---|
| Best overall F1 | Multi-text balanced Logistic Regression |
| Highest recall | Original Multinomial Naive Bayes |
| Highest precision among tested final models | Multi-text Complement Naive Bayes |
| Fewest false positives | Clean-text balanced LinearSVC |
| Best balance between precision and recall | Multi-text balanced Logistic Regression |
| Final conclusion | F1 improved from **44.57% to 46.90%**, approximately **5.2% relative improvement** |

## Result Sources

| Experiment | File |
|---|---|
| Original baseline | `results/model_comparison_results.csv` |
| Clean-text TF-IDF | `results/tfidf_ngram_results.csv` |
| Clean-text cross-validation | `results/tfidf_ngram_cv_results.csv` |
| Multi-text TF-IDF | `results/tfidf_multitext_results.csv` |
| Multi-text cross-validation | `results/tfidf_multitext_cv_results.csv` |
