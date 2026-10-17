"""Run TF-IDF n-gram experiments without changing the baseline notebooks."""

from __future__ import annotations

import argparse
import json
import re
import string
from pathlib import Path
from typing import Callable

import pandas as pd
from joblib import Memory
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    make_scorer,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
RANDOM_STATE = 42
POS_LABEL = "Relevant"
LABELS = ["Not Relevant", "Relevant"]

LABEL_CANDIDATES = ("relevance", "relevant", "class", "label", "target")
TEXT_CANDIDATES = (
    "headline",
    "headline_text",
    "title",
    "text",
    "article",
    "body",
    "content",
    "description",
    "summary",
)
LEGACY_TEXT_PRIORITY = ("text", "article", "body", "content", "headline")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare TF-IDF n-gram models with the saved BoW baseline."
    )
    parser.add_argument(
        "--mode",
        choices=("all", "clean", "multitext"),
        default="all",
        help="Run the shared clean-text experiment, raw multi-text experiment, or both.",
    )
    parser.add_argument(
        "--raw-data",
        type=Path,
        default=DATA_DIR / "US-Economic-News.csv",
        help="Raw CSV used by the multi-text experiment.",
    )
    parser.add_argument("--cv", type=int, default=5, help="Number of stratified CV folds.")
    parser.add_argument("--jobs", type=int, default=-1, help="Parallel GridSearchCV jobs.")
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("/tmp/nlp_casestudy_tfidf_cache"),
        help="Pipeline cache outside the repository.",
    )
    return parser.parse_args()


def normalize_label(value: object) -> str:
    normalized = str(value).strip().lower()
    if normalized in {"yes", "relevant", "1", "true"}:
        return POS_LABEL
    return "Not Relevant"


def legacy_preprocess(text: object) -> str:
    value = str(text).lower()
    value = re.sub(r"</?[a-z]+\s*/?>", " ", value)
    value = value.translate(str.maketrans("", "", string.punctuation))
    value = re.sub(r"\d+", " ", value)
    value = re.sub(r"[^a-z\s]", " ", value)
    tokens = [
        token
        for token in value.split()
        if token not in ENGLISH_STOP_WORDS and len(token) > 1
    ]
    return " ".join(tokens)


def word_tfidf() -> TfidfVectorizer:
    return TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.98,
        sublinear_tf=True,
        strip_accents="unicode",
    )


def character_tfidf() -> TfidfVectorizer:
    return TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=2,
        max_features=80_000,
        sublinear_tf=True,
    )


def clean_text_representations() -> dict[str, Callable[[], object]]:
    return {
        "word_tfidf_1_2": word_tfidf,
        "word_and_char_tfidf": lambda: FeatureUnion(
            [("word", word_tfidf()), ("character", character_tfidf())]
        ),
    }


def multi_text_representations(
    text_columns: list[str],
) -> dict[str, Callable[[], object]]:
    def word_columns() -> ColumnTransformer:
        return ColumnTransformer(
            [(f"{column}_word", word_tfidf(), column) for column in text_columns]
        )

    def word_and_character_columns() -> ColumnTransformer:
        transformers = []
        for column in text_columns:
            transformers.append((f"{column}_word", word_tfidf(), column))
            transformers.append((f"{column}_character", character_tfidf(), column))
        return ColumnTransformer(transformers)

    return {
        "all_text_columns_word_tfidf_1_2": word_columns,
        "all_text_columns_word_and_char_tfidf": word_and_character_columns,
    }


def model_specs() -> dict[str, tuple[object, dict[str, list[object]]]]:
    return {
        "Complement Naive Bayes": (
            ComplementNB(),
            {"classifier__alpha": [0.1, 0.5, 1.0, 2.0, 5.0]},
        ),
        "Logistic Regression": (
            LogisticRegression(max_iter=2_000, solver="liblinear", random_state=RANDOM_STATE),
            {
                "classifier__C": [0.01, 0.1, 1.0, 10.0],
                "classifier__class_weight": [None, "balanced"],
            },
        ),
        "LinearSVC": (
            LinearSVC(max_iter=10_000),
            {
                "classifier__C": [0.01, 0.1, 1.0],
                "classifier__class_weight": [None, "balanced"],
            },
        ),
    }


def load_baseline() -> pd.Series:
    comparison_path = RESULTS_DIR / "model_comparison_results.csv"
    comparison = pd.read_csv(comparison_path)
    return comparison.loc[comparison["F1-Score"].idxmax()]


def evaluate_predictions(y_test: pd.Series, predictions: object) -> dict[str, object]:
    matrix = confusion_matrix(y_test, predictions, labels=LABELS)
    return {
        "Accuracy": accuracy_score(y_test, predictions),
        "Precision": precision_score(
            y_test, predictions, pos_label=POS_LABEL, zero_division=0
        ),
        "Recall": recall_score(y_test, predictions, pos_label=POS_LABEL, zero_division=0),
        "F1-Score": f1_score(y_test, predictions, pos_label=POS_LABEL, zero_division=0),
        "TN": int(matrix[0, 0]),
        "FP": int(matrix[0, 1]),
        "FN": int(matrix[1, 0]),
        "TP": int(matrix[1, 1]),
    }


def run_searches(
    *,
    experiment_name: str,
    X_train: object,
    X_test: object,
    y_train: pd.Series,
    y_test: pd.Series,
    representations: dict[str, Callable[[], object]],
    cv_folds: int,
    jobs: int,
    cache_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    scorer = make_scorer(f1_score, pos_label=POS_LABEL, zero_division=0)
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_STATE)
    memory = Memory(location=cache_dir, verbose=0)
    baseline = load_baseline()
    search_rows: list[dict[str, object]] = []
    final_rows: list[dict[str, object]] = []

    for model_name, (classifier, parameter_grid) in model_specs().items():
        candidates = []
        for representation_name, representation_factory in representations.items():
            pipeline = Pipeline(
                [
                    ("features", representation_factory()),
                    ("classifier", classifier),
                ],
                memory=memory,
            )
            search = GridSearchCV(
                pipeline,
                parameter_grid,
                scoring=scorer,
                cv=cv,
                n_jobs=jobs,
                refit=True,
            )
            search.fit(X_train, y_train)
            row = {
                "Experiment": experiment_name,
                "Model": model_name,
                "Representation": representation_name,
                "CV F1": search.best_score_,
                "Best Params": json.dumps(search.best_params_, sort_keys=True),
            }
            search_rows.append(row)
            candidates.append((search.best_score_, representation_name, search))
            print(
                f"{experiment_name} | {model_name} | {representation_name}: "
                f"CV F1={search.best_score_:.4f}"
            )

        best_cv_f1, best_representation, best_search = max(
            candidates, key=lambda candidate: candidate[0]
        )
        predictions = best_search.predict(X_test)
        metrics = evaluate_predictions(y_test, predictions)
        final_rows.append(
            {
                "Experiment": experiment_name,
                "Model": model_name,
                "Representation": best_representation,
                "CV F1": best_cv_f1,
                **metrics,
                "Baseline F1": baseline["F1-Score"],
                "Delta F1": metrics["F1-Score"] - baseline["F1-Score"],
                "Best Params": json.dumps(best_search.best_params_, sort_keys=True),
            }
        )

    return pd.DataFrame(final_rows), pd.DataFrame(search_rows)


def load_clean_split() -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    train = pd.read_csv(DATA_DIR / "train_split.csv")
    test = pd.read_csv(DATA_DIR / "test_split.csv")
    return (
        train["clean_text"].fillna(""),
        test["clean_text"].fillna(""),
        train["label"],
        test["label"],
    )


def find_columns(raw: pd.DataFrame) -> tuple[str, list[str], str]:
    lower_to_original = {column.lower(): column for column in raw.columns}
    label_column = next(
        (lower_to_original[name] for name in LABEL_CANDIDATES if name in lower_to_original),
        None,
    )
    text_columns = [
        lower_to_original[name] for name in TEXT_CANDIDATES if name in lower_to_original
    ]
    legacy_column = next(
        (
            lower_to_original[name]
            for name in LEGACY_TEXT_PRIORITY
            if name in lower_to_original
        ),
        None,
    )
    if label_column is None:
        raise ValueError(f"No label column found. Available columns: {list(raw.columns)}")
    if legacy_column is None or not text_columns:
        raise ValueError(f"No supported text columns found. Available columns: {list(raw.columns)}")
    return label_column, list(dict.fromkeys(text_columns)), legacy_column


def load_multi_text_split(
    raw_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, list[str]]:
    raw = pd.read_csv(raw_path, encoding="latin-1")
    label_column, text_columns, legacy_column = find_columns(raw)

    frame = raw[[*text_columns, label_column]].copy()
    frame = frame.dropna(subset=[legacy_column]).reset_index(drop=True)
    frame["label"] = frame[label_column].map(normalize_label)
    frame["legacy_clean_text"] = frame[legacy_column].map(legacy_preprocess)
    frame = frame[frame["legacy_clean_text"].str.len() > 0].reset_index(drop=True)
    frame[text_columns] = frame[text_columns].fillna("").astype(str)

    train_indices, test_indices = train_test_split(
        frame.index,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=frame["label"],
    )
    train = frame.loc[train_indices]
    test = frame.loc[test_indices]

    saved_train = pd.read_csv(DATA_DIR / "train_split.csv")
    saved_test = pd.read_csv(DATA_DIR / "test_split.csv")
    if train["label"].tolist() != saved_train["label"].tolist():
        raise ValueError(
            "The reconstructed training split does not match the baseline labels. "
            "Add a stable row ID before comparing multi-column results."
        )
    if test["label"].tolist() != saved_test["label"].tolist():
        raise ValueError(
            "The reconstructed test split does not match the baseline labels. "
            "Add a stable row ID before comparing multi-column results."
        )

    print(f"Using raw text columns: {text_columns}")
    return (
        train[text_columns],
        test[text_columns],
        train["label"],
        test["label"],
        text_columns,
    )


def save_results(
    name: str, final_results: pd.DataFrame, cv_results: pd.DataFrame
) -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    final_path = RESULTS_DIR / f"{name}_results.csv"
    cv_path = RESULTS_DIR / f"{name}_cv_results.csv"
    final_results.to_csv(final_path, index=False)
    cv_results.to_csv(cv_path, index=False)
    print(f"Saved {final_path.relative_to(PROJECT_ROOT)}")
    print(f"Saved {cv_path.relative_to(PROJECT_ROOT)}")
    print(final_results.sort_values("F1-Score", ascending=False).to_string(index=False))


def main() -> None:
    args = parse_args()
    if args.cv < 2:
        raise ValueError("--cv must be at least 2")

    if args.mode in {"all", "clean"}:
        X_train, X_test, y_train, y_test = load_clean_split()
        final_results, cv_results = run_searches(
            experiment_name="TF-IDF n-grams on shared clean_text",
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            representations=clean_text_representations(),
            cv_folds=args.cv,
            jobs=args.jobs,
            cache_dir=args.cache_dir / "clean",
        )
        save_results("tfidf_ngram", final_results, cv_results)

    if args.mode in {"all", "multitext"}:
        if not args.raw_data.exists():
            if args.mode == "multitext":
                raise FileNotFoundError(
                    f"Raw dataset not found: {args.raw_data}. "
                    "Add it or pass --raw-data PATH."
                )
            print(
                f"Skipping multi-text experiment because {args.raw_data} is missing. "
                "Add the raw CSV and rerun with --mode multitext."
            )
            return

        X_train, X_test, y_train, y_test, text_columns = load_multi_text_split(
            args.raw_data
        )
        final_results, cv_results = run_searches(
            experiment_name="TF-IDF n-grams across all text columns",
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            representations=multi_text_representations(text_columns),
            cv_folds=args.cv,
            jobs=args.jobs,
            cache_dir=args.cache_dir / "multitext",
        )
        save_results("tfidf_multitext", final_results, cv_results)


if __name__ == "__main__":
    main()
