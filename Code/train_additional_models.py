import argparse
import csv
import math
import random
import re
import time
import zlib
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path


DATA_DIR = Path(__file__).resolve().parent
TRAIN_PATH = DATA_DIR / "final_train.csv"
VALID_PATH = DATA_DIR / "final_valid.csv"
TEST_PATH = DATA_DIR / "final_test.csv"
RESULT_PATH = DATA_DIR / "result.txt"
EXTRA_TXT_PATH = DATA_DIR / "additional_model_results.txt"
EXTRA_CSV_PATH = DATA_DIR / "additional_model_results.csv"

TOKEN_RE = re.compile(r"\w+", flags=re.UNICODE)


def read_dataset(path):
    encodings = ("utf-8-sig", "utf-8", "cp1258", "latin-1")

    for encoding in encodings:
        try:
            with open(path, "r", encoding=encoding, newline="") as f:
                rows = list(csv.DictReader(f))
            texts = [row["text"] for row in rows if row.get("text") and row.get("label")]
            labels = [
                int(row["label"])
                for row in rows
                if row.get("text") and row.get("label")
            ]
            return texts, labels
        except UnicodeDecodeError:
            continue

    raise ValueError(f"Cannot read {path} with supported encodings.")


def word_analyzer(text, ngram_range=(1, 2)):
    tokens = TOKEN_RE.findall(text.lower())
    features = []
    min_n, max_n = ngram_range

    for n in range(min_n, max_n + 1):
        if len(tokens) < n:
            continue
        for i in range(len(tokens) - n + 1):
            features.append(" ".join(tokens[i : i + n]))

    return features


def char_analyzer(text, ngram_range=(3, 5)):
    text = " " + re.sub(r"\s+", " ", text.lower()).strip() + " "
    features = []
    min_n, max_n = ngram_range

    for n in range(min_n, max_n + 1):
        if len(text) < n:
            continue
        for i in range(len(text) - n + 1):
            features.append(text[i : i + n])

    return features


def build_vocabulary(texts, analyzer, min_df=2, max_features=50000):
    doc_freq = Counter()

    for text in texts:
        doc_freq.update(set(analyzer(text)))

    features = [
        (feature, freq)
        for feature, freq in doc_freq.items()
        if freq >= min_df
    ]
    features.sort(key=lambda item: (-item[1], item[0]))

    if max_features:
        features = features[:max_features]

    vocab = {feature: idx for idx, (feature, _) in enumerate(features)}
    return vocab, doc_freq


def f1_for_label(y_true, y_pred, target):
    tp = sum(yt == target and yp == target for yt, yp in zip(y_true, y_pred))
    fp = sum(yt != target and yp == target for yt, yp in zip(y_true, y_pred))
    fn = sum(yt == target and yp != target for yt, yp in zip(y_true, y_pred))

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def compute_metrics(y_true, y_pred):
    accuracy = sum(yt == yp for yt, yp in zip(y_true, y_pred)) / len(y_true)
    f1_clean = f1_for_label(y_true, y_pred, 0)
    f1_toxic = f1_for_label(y_true, y_pred, 1)

    return {
        "accuracy": accuracy,
        "macro_f1": (f1_clean + f1_toxic) / 2,
        "f1_clean": f1_clean,
        "f1_toxic": f1_toxic,
    }


def predict_from_scores(scores, threshold):
    return [1 if score >= threshold else 0 for score in scores]


def find_best_threshold(y_true, scores):
    if not scores:
        return 0.0, 0.0

    candidates = sorted(set(scores))
    if len(candidates) > 250:
        step = max(1, len(candidates) // 250)
        candidates = candidates[::step] + [candidates[-1]]

    best_threshold = candidates[0]
    best_macro_f1 = -1.0

    for threshold in candidates:
        pred = predict_from_scores(scores, threshold)
        macro_f1 = compute_metrics(y_true, pred)["macro_f1"]

        if macro_f1 > best_macro_f1:
            best_macro_f1 = macro_f1
            best_threshold = threshold

    return best_threshold, best_macro_f1


class MultinomialNBText:
    def __init__(self, analyzer, min_df=2, max_features=50000, alpha=1.0):
        self.analyzer = analyzer
        self.min_df = min_df
        self.max_features = max_features
        self.alpha = alpha

    def fit(self, texts, labels):
        self.vocab, _ = build_vocabulary(
            texts,
            analyzer=self.analyzer,
            min_df=self.min_df,
            max_features=self.max_features,
        )

        class_doc_counts = Counter(labels)
        class_token_counts = {0: Counter(), 1: Counter()}
        total_tokens = {0: 0, 1: 0}

        for text, label in zip(texts, labels):
            counts = Counter(
                feature for feature in self.analyzer(text) if feature in self.vocab
            )
            class_token_counts[label].update(counts)
            total_tokens[label] += sum(counts.values())

        n_docs = len(labels)
        vocab_size = len(self.vocab)
        self.log_prior = {
            label: math.log(class_doc_counts[label] / n_docs)
            for label in (0, 1)
        }
        self.default_log_prob = {}
        self.feature_log_prob = {0: {}, 1: {}}

        for label in (0, 1):
            denom = total_tokens[label] + self.alpha * vocab_size
            self.default_log_prob[label] = math.log(self.alpha / denom)

            for feature, count in class_token_counts[label].items():
                self.feature_log_prob[label][feature] = math.log(
                    (count + self.alpha) / denom
                )

        return self

    def decision_function(self, texts):
        scores = []

        for text in texts:
            counts = Counter(
                feature for feature in self.analyzer(text) if feature in self.vocab
            )
            class_scores = {}

            for label in (0, 1):
                score = self.log_prior[label]
                default = self.default_log_prob[label]
                probs = self.feature_log_prob[label]

                for feature, count in counts.items():
                    score += count * probs.get(feature, default)

                class_scores[label] = score

            scores.append(class_scores[1] - class_scores[0])

        return scores


class BernoulliNBText:
    def __init__(self, analyzer, min_df=2, max_features=30000, alpha=1.0):
        self.analyzer = analyzer
        self.min_df = min_df
        self.max_features = max_features
        self.alpha = alpha

    def fit(self, texts, labels):
        self.vocab, _ = build_vocabulary(
            texts,
            analyzer=self.analyzer,
            min_df=self.min_df,
            max_features=self.max_features,
        )

        class_doc_counts = Counter(labels)
        feature_doc_counts = {0: Counter(), 1: Counter()}

        for text, label in zip(texts, labels):
            features = {
                feature for feature in self.analyzer(text) if feature in self.vocab
            }
            feature_doc_counts[label].update(features)

        n_docs = len(labels)
        self.base_score = {}
        self.present_delta = {0: {}, 1: {}}

        for label in (0, 1):
            class_docs = class_doc_counts[label]
            self.base_score[label] = math.log(class_docs / n_docs)

            for feature in self.vocab:
                prob = (
                    feature_doc_counts[label][feature] + self.alpha
                ) / (class_docs + 2 * self.alpha)
                log_absent = math.log(1 - prob)
                self.base_score[label] += log_absent
                self.present_delta[label][feature] = math.log(prob) - log_absent

        return self

    def decision_function(self, texts):
        scores = []

        for text in texts:
            features = {
                feature for feature in self.analyzer(text) if feature in self.vocab
            }
            class_scores = {}

            for label in (0, 1):
                score = self.base_score[label]

                for feature in features:
                    score += self.present_delta[label][feature]

                class_scores[label] = score

            scores.append(class_scores[1] - class_scores[0])

        return scores


class RocchioTFIDF:
    def __init__(self, analyzer, min_df=2, max_features=60000):
        self.analyzer = analyzer
        self.min_df = min_df
        self.max_features = max_features

    def fit(self, texts, labels):
        self.vocab, doc_freq = build_vocabulary(
            texts,
            analyzer=self.analyzer,
            min_df=self.min_df,
            max_features=self.max_features,
        )
        n_docs = len(texts)
        self.idf = {
            feature: math.log((1 + n_docs) / (1 + doc_freq[feature])) + 1
            for feature in self.vocab
        }

        sums = {0: defaultdict(float), 1: defaultdict(float)}
        class_doc_counts = Counter(labels)

        for text, label in zip(texts, labels):
            vector = self._vectorize(text)

            for feature, value in vector.items():
                sums[label][feature] += value

        self.centroids = {}

        for label in (0, 1):
            centroid = {
                feature: value / class_doc_counts[label]
                for feature, value in sums[label].items()
            }
            norm = math.sqrt(sum(value * value for value in centroid.values())) or 1.0
            self.centroids[label] = {
                feature: value / norm for feature, value in centroid.items()
            }

        return self

    def _vectorize(self, text):
        counts = Counter(
            feature for feature in self.analyzer(text) if feature in self.vocab
        )
        vector = {
            feature: (1 + math.log(count)) * self.idf[feature]
            for feature, count in counts.items()
        }
        norm = math.sqrt(sum(value * value for value in vector.values())) or 1.0
        return {feature: value / norm for feature, value in vector.items()}

    def decision_function(self, texts):
        scores = []

        for text in texts:
            vector = self._vectorize(text)
            score_0 = sum(
                value * self.centroids[0].get(feature, 0.0)
                for feature, value in vector.items()
            )
            score_1 = sum(
                value * self.centroids[1].get(feature, 0.0)
                for feature, value in vector.items()
            )
            scores.append(score_1 - score_0)

        return scores


class PassiveAggressiveHashing:
    def __init__(self, n_features=2**20, epochs=4, c=0.5, seed=42):
        self.n_features = n_features
        self.epochs = epochs
        self.c = c
        self.seed = seed
        self.weights = defaultdict(float)

    def fit(self, texts, labels):
        random.seed(self.seed)
        class_counts = Counter(labels)
        total = len(labels)
        class_weight = {
            0: total / (2 * class_counts[0]),
            1: total / (2 * class_counts[1]),
        }

        indices = list(range(total))

        for epoch in range(1, self.epochs + 1):
            random.shuffle(indices)
            updates = 0

            for idx in indices:
                x = self._vectorize(texts[idx])
                y = 1 if labels[idx] == 1 else -1
                margin = y * self._dot(x)
                loss = max(0.0, 1.0 - margin)

                if loss == 0.0:
                    continue

                norm = sum(value * value for value in x.values()) or 1.0
                tau = loss / (norm + 1.0 / (2 * self.c))
                tau *= class_weight[labels[idx]]
                update = tau * y

                for feature, value in x.items():
                    self.weights[feature] += update * value

                updates += 1

            print(f"  Passive-Aggressive epoch {epoch}: updates={updates:,}")

        return self

    def _hash(self, feature):
        return zlib.crc32(feature.encode("utf-8")) % self.n_features

    def _vectorize(self, text):
        counts = Counter(self._hash(feature) for feature in word_analyzer(text, (1, 2)))
        norm = math.sqrt(sum(count * count for count in counts.values())) or 1.0
        return {feature: count / norm for feature, count in counts.items()}

    def _dot(self, vector):
        return sum(self.weights.get(feature, 0.0) * value for feature, value in vector.items())

    def decision_function(self, texts):
        return [self._dot(self._vectorize(text)) for text in texts]


def evaluate_model(model_name, model, train_texts, train_y, valid_texts, valid_y, test_texts, test_y):
    print(f"\nTraining: {model_name}")
    start = time.time()
    model.fit(train_texts, train_y)
    train_seconds = time.time() - start

    print(f"Tuning threshold on valid: {model_name}")
    valid_scores = model.decision_function(valid_texts)
    threshold, valid_macro_f1 = find_best_threshold(valid_y, valid_scores)

    print(f"Evaluating on test: {model_name}")
    start = time.time()
    test_scores = model.decision_function(test_texts)
    predict_seconds = time.time() - start
    test_pred = predict_from_scores(test_scores, threshold)
    metrics = compute_metrics(test_y, test_pred)

    result = {
        "model": model_name,
        "threshold": threshold,
        "valid_macro_f1": valid_macro_f1,
        "train_seconds": train_seconds,
        "predict_seconds": predict_seconds,
        **metrics,
    }

    print(format_result_block(result))
    return result


def format_result_block(result):
    lines = [
        f"==================== KET QUA DANH GIA: {result['model']} ====================",
        f"Accuracy        : {result['accuracy']:.4f}",
        f"Macro Avg F1    : {result['macro_f1']:.4f}",
        f"F1 - Clean (0)  : {result['f1_clean']:.4f}",
        f"F1 - Toxic (1)  : {result['f1_toxic']:.4f}",
        f"Threshold Valid : {result['threshold']:.6f}",
        f"Valid Macro F1  : {result['valid_macro_f1']:.4f}",
        f"Train Time      : {result['train_seconds']:.2f}s",
        f"Predict Time    : {result['predict_seconds']:.2f}s",
        "=" * 60,
    ]
    return "\n".join(lines)


def write_outputs(results, append_result=False):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    blocks = [
        f"\n\n########## CAC MO HINH BO SUNG ({timestamp}) ##########",
        *[format_result_block(result) for result in results],
    ]
    output_text = "\n\n".join(blocks) + "\n"

    with open(EXTRA_TXT_PATH, "w", encoding="utf-8") as f:
        f.write(output_text)

    with open(EXTRA_CSV_PATH, "w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "model",
            "accuracy",
            "macro_f1",
            "f1_clean",
            "f1_toxic",
            "threshold",
            "valid_macro_f1",
            "train_seconds",
            "predict_seconds",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow(result)

    if append_result:
        with open(RESULT_PATH, "a", encoding="utf-8") as f:
            f.write(output_text)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--append-result",
        action="store_true",
        help="Append additional model results to result.txt.",
    )
    args = parser.parse_args()

    print("Loading data...")
    train_texts, train_y = read_dataset(TRAIN_PATH)
    valid_texts, valid_y = read_dataset(VALID_PATH)
    test_texts, test_y = read_dataset(TEST_PATH)

    print(f"Train: {len(train_texts):,}")
    print(f"Valid: {len(valid_texts):,}")
    print(f"Test : {len(test_texts):,}")

    models = [
        (
            "MultinomialNB + Word Count ngram(1,2)",
            MultinomialNBText(
                analyzer=lambda text: word_analyzer(text, (1, 2)),
                min_df=2,
                max_features=60000,
            ),
        ),
        (
            "MultinomialNB + Char Count ngram(3,5)",
            MultinomialNBText(
                analyzer=lambda text: char_analyzer(text, (3, 5)),
                min_df=3,
                max_features=80000,
            ),
        ),
        (
            "BernoulliNB + Binary Word ngram(1,2)",
            BernoulliNBText(
                analyzer=lambda text: word_analyzer(text, (1, 2)),
                min_df=2,
                max_features=40000,
            ),
        ),
        (
            "Rocchio + TF-IDF Word ngram(1,2)",
            RocchioTFIDF(
                analyzer=lambda text: word_analyzer(text, (1, 2)),
                min_df=2,
                max_features=60000,
            ),
        ),
        (
            "PassiveAggressive + Hashing Word ngram(1,2)",
            PassiveAggressiveHashing(n_features=2**20, epochs=4, c=0.5, seed=42),
        ),
    ]

    results = []

    for model_name, model in models:
        results.append(
            evaluate_model(
                model_name,
                model,
                train_texts,
                train_y,
                valid_texts,
                valid_y,
                test_texts,
                test_y,
            )
        )

    results.sort(key=lambda result: result["macro_f1"], reverse=True)
    write_outputs(results, append_result=args.append_result)

    print("\nRanking by Macro F1:")
    for idx, result in enumerate(results, start=1):
        print(
            f"{idx}. {result['model']} | "
            f"Macro F1={result['macro_f1']:.4f} | "
            f"Acc={result['accuracy']:.4f} | "
            f"F1 Toxic={result['f1_toxic']:.4f}"
        )

    print(f"\nSaved: Data&Result/{EXTRA_TXT_PATH.name}")
    print(f"Saved: Data&Result/{EXTRA_CSV_PATH.name}")
    if args.append_result:
        print(f"Appended: Data&Result/{RESULT_PATH.name}")


if __name__ == "__main__":
    main()
