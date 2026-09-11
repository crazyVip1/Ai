"""Train an AI-text detector on the full CSV without loading it into RAM."""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
import hashlib
import random
import re
import sys

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


DATA_FILE = "Ai_Human.csv"
OUTPUT_FILE = "ai_detector_pipeline.pkl"
TEXT_COLUMN = "text"
LABEL_CANDIDATES = ("generated", "label", "target", "class", "y")
CHUNK_SIZE = 50_000
TEST_SIZE = 50_000
HASHING_FEATURES = 2**18
RANDOM_STATE = 42
NUMERIC_COLUMNS = (
    "word_count",
    "lexical_diversity",
    "avg_word_len",
    "punct_count",
)


# Make custom transformers importable from the serialized artifact even when
# this file is executed as a script instead of imported as a module.
if __name__ == "__main__":
    sys.modules.setdefault("train_model", sys.modules[__name__])


class LinguisticFeatureExtractor(BaseEstimator, TransformerMixin):
    """Add the four linguistic features used by the detector."""

    def __init__(self, text_column=TEXT_COLUMN):
        self.text_column = text_column

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        transformed = X.copy()
        texts = transformed[self.text_column].fillna("").astype(str)
        words = texts.map(lambda value: re.findall(r"\b\w+\b", value))
        transformed["word_count"] = words.map(len).astype(np.float32)
        transformed["lexical_diversity"] = words.map(
            lambda tokens: len(set(token.lower() for token in tokens)) / len(tokens)
            if tokens
            else 0.0
        ).astype(np.float32)
        transformed["avg_word_len"] = words.map(
            lambda tokens: np.mean([len(token) for token in tokens]) if tokens else 0.0
        ).astype(np.float32)
        transformed["punct_count"] = texts.map(
            lambda value: sum(character in '.,!?;:"()' for character in value)
        ).astype(np.float32)
        return transformed


class HashingFeatureTransformer(BaseEstimator, TransformerMixin):
    """Transform text and numeric columns into a sparse, inference-ready matrix."""

    def __init__(self, text_column=TEXT_COLUMN, n_features=HASHING_FEATURES):
        self.text_column = text_column
        self.n_features = n_features
        self.vectorizer = HashingVectorizer(
            n_features=n_features,
            alternate_sign=False,
            norm="l2",
            ngram_range=(1, 2),
            dtype=np.float32,
        )
        self.numeric_scaler = StandardScaler()

    def fit(self, X, y=None):
        self.numeric_scaler.partial_fit(X[list(NUMERIC_COLUMNS)])
        return self

    def transform(self, X):
        text_matrix = self.vectorizer.transform(X[self.text_column])
        numeric_matrix = csr_matrix(
            self.numeric_scaler.transform(X[list(NUMERIC_COLUMNS)]),
            dtype=np.float32,
        )
        return hstack((text_matrix, numeric_matrix), format="csr")
 
# Give custom estimators a stable pickle location when this file is run with
# ``python train_model.py`` rather than imported as ``train_model``.
LinguisticFeatureExtractor.__module__ = "train_model"
HashingFeatureTransformer.__module__ = "train_model"


def find_column(columns, candidates, description):
    normalized = {str(column).strip().lower(): column for column in columns}
    for candidate in candidates:
        if candidate in normalized:
            return normalized[candidate]
    raise ValueError(
        f"Could not detect the {description} column. Available columns: {list(columns)}"
    )


def resolve_data_file() -> Path:
    for filename in (DATA_FILE, "AI_Human.csv", "ai_human.csv"):
        path = Path(filename)
        if path.is_file():
            return path
    raise FileNotFoundError(f"Dataset not found: expected {DATA_FILE} in the workspace root.")


def text_key(text: str) -> bytes:
    """Return a compact stable key for whole-file duplicate detection."""
    return hashlib.sha1(text.encode("utf-8")).digest()


def iter_clean_chunks(data_path, text_column, label_column):
    """Yield cleaned, globally deduplicated chunks while keeping RAM bounded."""
    seen_texts = set()
    for chunk in pd.read_csv(
        data_path,
        usecols=[text_column, label_column],
        chunksize=CHUNK_SIZE,
    ):
        chunk.dropna(subset=[text_column, label_column], inplace=True)
        chunk[text_column] = chunk[text_column].astype(str).str.strip()
        chunk = chunk[chunk[text_column] != ""].copy()
        chunk["_text_key"] = chunk[text_column].map(text_key)
        chunk.drop_duplicates(subset=["_text_key"], inplace=True)
        unseen = ~chunk["_text_key"].isin(seen_texts)
        clean_chunk = chunk.loc[unseen].copy()
        seen_texts.update(clean_chunk["_text_key"])
        if not clean_chunk.empty:
            yield clean_chunk.drop(columns=["_text_key"])


def get_label_column(header):
    return find_column(header.columns, LABEL_CANDIDATES, "target/label")


def count_unique_rows(data_path, text_column, label_column):
    counts = Counter()
    total = 0
    for chunk in iter_clean_chunks(data_path, text_column, label_column):
        counts.update(chunk[label_column].tolist())
        total += len(chunk)
    return counts, total


def holdout_sizes(label_counts, requested_size):
    total = sum(label_counts.values())
    target = min(requested_size, max(1, total // 5))
    sizes = {
        label: min(count, int(target * count / total))
        for label, count in label_counts.items()
    }
    while sum(sizes.values()) < target:
        label = max(
            (label for label in label_counts if sizes[label] < label_counts[label]),
            key=lambda item: label_counts[item] - sizes[item],
            default=None,
        )
        if label is None:
            break
        sizes[label] += 1
    return sizes


def select_holdout(data_path, text_column, label_column, label_counts):
    """Reservoir-sample an exact, stratified holdout without storing all rows."""
    wanted = holdout_sizes(label_counts, TEST_SIZE)
    reservoirs = defaultdict(list)
    seen_by_label = Counter()
    rng = random.Random(RANDOM_STATE)

    for chunk in iter_clean_chunks(data_path, text_column, label_column):
        for text, label in zip(chunk[text_column], chunk[label_column]):
            label_seen = seen_by_label[label]
            seen_by_label[label] += 1
            if len(reservoirs[label]) < wanted[label]:
                reservoirs[label].append((text, label))
            else:
                replacement = rng.randrange(label_seen + 1)
                if replacement < wanted[label]:
                    reservoirs[label][replacement] = (text, label)

    rows = [row for reservoir in reservoirs.values() for row in reservoir]
    return pd.DataFrame(rows, columns=[text_column, label_column])


def prepare_features(chunk, text_column, label_column):
    data = chunk[[text_column, label_column]].rename(columns={text_column: TEXT_COLUMN})
    featured = LinguisticFeatureExtractor().transform(data)
    labels = featured.pop(label_column).to_numpy()
    return featured, labels


def main():
    data_path = resolve_data_file()
    header = pd.read_csv(data_path, nrows=0)
    text_column = find_column(
        header.columns,
        (TEXT_COLUMN, "content", "document", "essay"),
        "text",
    )
    label_column = get_label_column(header)

    print(f"Reading {data_path} in chunks of {CHUNK_SIZE:,} rows...")
    label_counts, total_unique = count_unique_rows(data_path, text_column, label_column)
    all_labels = np.array(sorted(label_counts, key=str))
    if len(all_labels) < 2:
        raise ValueError("The target column must contain at least two classes.")
    print(f"Unique rows discovered: {total_unique:,}")
    print(f"Labels: {dict(label_counts)}")

    holdout = select_holdout(data_path, text_column, label_column, label_counts)
    # Use exact text values for exclusion. The digest is used for efficient
    # corpus deduplication, while exact values make the holdout invariant clear.
    holdout_texts = set(holdout[text_column].tolist())
    print(f"Stratified holdout: {len(holdout):,} rows")

    # Pass 3: fit numeric scaling incrementally on training rows.
    transformer = HashingFeatureTransformer()
    scaler_fitted = False
    for chunk in iter_clean_chunks(data_path, text_column, label_column):
        train_chunk = chunk[~chunk[text_column].isin(holdout_texts)]
        if train_chunk.empty:
            continue
        featured, _ = prepare_features(train_chunk, text_column, label_column)
        transformer.numeric_scaler.partial_fit(featured[list(NUMERIC_COLUMNS)])
        scaler_fitted = True
    if not scaler_fitted:
        raise ValueError("No training rows remain after selecting the holdout.")

    classifier = SGDClassifier(
        loss="log_loss",
        max_iter=1000,
        random_state=RANDOM_STATE,
    )

    # Pass 4: stream every non-holdout row through partial_fit.
    processed = 0
    first_batch = True
    for chunk in iter_clean_chunks(data_path, text_column, label_column):
        train_chunk = chunk[~chunk[text_column].isin(holdout_texts)]
        if train_chunk.empty:
            continue
        featured, labels = prepare_features(train_chunk, text_column, label_column)
        matrix = transformer.transform(featured)
        if first_batch:
            classifier.partial_fit(matrix, labels, classes=all_labels)
            first_batch = False
        else:
            classifier.partial_fit(matrix, labels)
        processed += len(labels)
        print(
            f"Processed training rows: {processed:,}/{total_unique - len(holdout):,}",
            end="\r",
        )
    print()

    model = Pipeline(
        steps=[
            ("linguistic_features", LinguisticFeatureExtractor()),
            ("hashing_features", transformer),
            ("classifier", classifier),
        ]
    )
    test_features, test_labels = prepare_features(holdout, text_column, label_column)
    predictions = model.predict(test_features)
    print(f"Total unique rows processed: {total_unique:,}")
    print(f"Training rows processed: {processed:,}")
    print(f"Accuracy: {accuracy_score(test_labels, predictions):.4f}")
    print("\nClassification report:")
    print(classification_report(test_labels, predictions, zero_division=0))
    joblib.dump(model, OUTPUT_FILE)
    print(f"Saved fitted pipeline to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
