import json
import re
import unicodedata
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DATA_DIR = Path("ViCTSD Dataset")
OUTPUT_DIR = Path("victsd_analysis_outputs")
PLOTS_DIR = OUTPUT_DIR / "plots"
HISTOGRAM_MAX_WORDS = 500
HISTOGRAM_BIN_WIDTH = 10

CSV_FILES = {
    "train": DATA_DIR / "ViCTSD_train.csv",
    "valid": DATA_DIR / "ViCTSD_valid.csv",
    "test": DATA_DIR / "ViCTSD_test.csv",
}

TEXT_COL = "Comment"
CONSTRUCT_COL = "Constructiveness"
TOXICITY_COL = "Toxicity"

CONSTRUCT_LABELS = [0, 1]
TOXICITY_LABELS = [0, 1, 2, 3]

CONSTRUCT_LABEL_NAME = {
    0: "Non-Constructive",
    1: "Constructive",
}

TOXICITY_LABEL_NAME = {
    0: "Non-toxic",
    1: "Quite toxic",
    2: "Toxic",
    3: "Very toxic",
}

URL_PATTERN = re.compile(r"(https?://\S+|www\.\S+)", re.IGNORECASE)
EMOJI_PATTERN = re.compile(
    "["
    "\U0001F600-\U0001F64F"
    "\U0001F300-\U0001F5FF"
    "\U0001F680-\U0001F6FF"
    "\U0001F1E0-\U0001F1FF"
    "\U00002700-\U000027BF"
    "\U0001F900-\U0001F9FF"
    "\U00002600-\U000026FF"
    "]+",
    flags=re.UNICODE,
)
TOKEN_PATTERN = re.compile(r"\b\w+\b", flags=re.UNICODE)

VI_STOPWORDS = {
    "và", "là", "của", "có", "cho", "với", "một", "những", "các", "được", "trong", "khi", "đã", "này", "đó",
    "thì", "mà", "rằng", "vì", "nên", "ra", "vào", "đi", "đến", "tôi", "tao", "mày", "bạn", "anh", "chị", "em",
    "nó", "họ", "chúng", "mình", "ở", "theo", "từ", "lại", "rất", "quá", "thật", "không", "ko", "k", "để",
    "hay", "cũng", "đang", "sẽ", "bị", "do", "vẫn", "chỉ", "thôi", "đây", "kia", "ấy", "à", "ừ", "ờ", "ơi",
}

SLANG_PATTERNS = [
    r"\bko\b", r"\bk\b", r"\bkhum\b", r"\bhok\b", r"\bj\b", r"\bz\b", r"\bvk\b", r"\bck\b",
    r"\bcmnr\b", r"\bvl\b", r"\bvcl\b", r"\blol\b", r"\blmao\b", r"\bwtf\b", r"\bdm\b", r"\bđm\b",
    r"\bclm\b", r"\bcc\b", r"\bkg\b", r"\bhem\b", r"\bhông\b", r"\bđc\b", r"\bdc\b", r"\bntn\b",
    r"\bib\b", r"\brep\b", r"\bad\b", r":\)+", r"=\)+", r":v",
]
SLANG_REGEXES = [re.compile(p, re.IGNORECASE) for p in SLANG_PATTERNS]


def strip_accents(text: str) -> str:
    normalized = unicodedata.normalize("NFD", text)
    return "".join(ch for ch in normalized if unicodedata.category(ch) != "Mn")


def is_non_diacritic_comment(text: str) -> bool:
    letters = "".join(ch for ch in text if ch.isalpha())
    if not letters:
        return False
    return strip_accents(letters).lower() == letters.lower()


def tokenize(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(text.lower())


def contains_slang(text: str) -> bool:
    return any(rx.search(text) for rx in SLANG_REGEXES)


def ensure_dirs() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def load_dataset() -> pd.DataFrame:
    frames = []
    required_cols = [TEXT_COL, CONSTRUCT_COL, TOXICITY_COL]

    for split_name, csv_path in CSV_FILES.items():
        if not csv_path.exists():
            raise FileNotFoundError(f"Missing file: {csv_path}")

        df = pd.read_csv(csv_path)
        if any(col not in df.columns for col in required_cols):
            raise ValueError(f"{csv_path} must contain columns: {required_cols}")

        df = df[required_cols].copy()
        df[TEXT_COL] = df[TEXT_COL].fillna("").astype(str)
        df[CONSTRUCT_COL] = pd.to_numeric(df[CONSTRUCT_COL], errors="coerce").fillna(-1).astype(int)
        df[TOXICITY_COL] = pd.to_numeric(df[TOXICITY_COL], errors="coerce").fillna(-1).astype(int)
        df["split"] = split_name
        frames.append(df)

    return pd.concat(frames, ignore_index=True)


def plot_construct_distribution(construct_counts: pd.Series) -> None:
    labels = [f"{label} ({CONSTRUCT_LABEL_NAME[label]})" for label in CONSTRUCT_LABELS]
    values = [int(construct_counts.get(label, 0)) for label in CONSTRUCT_LABELS]

    plt.figure(figsize=(8, 5))
    bars = plt.bar(labels, values, color=["#4C78A8", "#F58518"])
    plt.title("ViCTSD Constructiveness Distribution")
    plt.xlabel("Constructiveness label")
    plt.ylabel("Number of samples")

    for bar in bars:
        h = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, h, f"{int(h)}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "constructiveness_distribution_bar_chart.png", dpi=200)
    plt.close()


def plot_toxicity_distribution(toxicity_counts: pd.Series) -> None:
    labels = [f"{label} ({TOXICITY_LABEL_NAME[label]})" for label in TOXICITY_LABELS]
    values = [int(toxicity_counts.get(label, 0)) for label in TOXICITY_LABELS]

    plt.figure(figsize=(10, 5))
    bars = plt.bar(labels, values, color=["#54A24B", "#E45756", "#F58518", "#B279A2"])
    plt.title("ViCTSD Toxicity Distribution")
    plt.xlabel("Toxicity label")
    plt.ylabel("Number of samples")
    plt.xticks(rotation=10)

    for bar in bars:
        h = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, h, f"{int(h)}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "toxicity_distribution_bar_chart.png", dpi=200)
    plt.close()


def plot_length_histogram(word_lengths: pd.Series) -> None:
    plt.figure(figsize=(8, 5))
    bins = range(0, HISTOGRAM_MAX_WORDS + HISTOGRAM_BIN_WIDTH, HISTOGRAM_BIN_WIDTH)
    plt.hist(word_lengths, bins=bins, color="#4C78A8", edgecolor="black", alpha=0.85)
    plt.title("Comment Length Distribution")
    plt.xlabel("Number of words")
    plt.ylabel("Frequency")
    plt.xlim(0, HISTOGRAM_MAX_WORDS)
    plt.xticks(range(0, HISTOGRAM_MAX_WORDS + 1, 50))
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "comment_length_histogram.png", dpi=200)
    plt.close()


def plot_top_words(top_words: list[tuple[str, int]], top_n: int = 20) -> None:
    if not top_words:
        return
    words, freqs = zip(*top_words[:top_n])
    plt.figure(figsize=(10, 6))
    plt.barh(list(words)[::-1], list(freqs)[::-1], color="#F58518")
    plt.title(f"Top {top_n} Most Frequent Words")
    plt.xlabel("Frequency")
    plt.ylabel("Word")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "word_frequency_chart.png", dpi=200)
    plt.close()


def plot_label_cooccurrence_matrix(df: pd.DataFrame) -> None:
    matrix = np.zeros((len(CONSTRUCT_LABELS), len(TOXICITY_LABELS)), dtype=int)

    for i, c in enumerate(CONSTRUCT_LABELS):
        for j, t in enumerate(TOXICITY_LABELS):
            matrix[i, j] = int(((df[CONSTRUCT_COL] == c) & (df[TOXICITY_COL] == t)).sum())

    plt.figure(figsize=(9, 5))
    plt.imshow(matrix, cmap="Blues")
    plt.colorbar(label="Count")
    plt.title("Label Co-occurrence Matrix (Constructiveness x Toxicity)")
    plt.xlabel("Toxicity")
    plt.ylabel("Constructiveness")

    plt.xticks(range(len(TOXICITY_LABELS)), [f"{l}\n{TOXICITY_LABEL_NAME[l]}" for l in TOXICITY_LABELS])
    plt.yticks(range(len(CONSTRUCT_LABELS)), [f"{l}\n{CONSTRUCT_LABEL_NAME[l]}" for l in CONSTRUCT_LABELS])

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            plt.text(j, i, str(matrix[i, j]), ha="center", va="center", color="black", fontsize=10)

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "label_cooccurrence_matrix.png", dpi=200)
    plt.close()


def main() -> None:
    ensure_dirs()
    df = load_dataset()

    total_samples = int(len(df))

    construct_counts = df[CONSTRUCT_COL].value_counts().reindex(CONSTRUCT_LABELS, fill_value=0).sort_index()
    toxicity_counts = df[TOXICITY_COL].value_counts().reindex(TOXICITY_LABELS, fill_value=0).sort_index()

    tokens_series = df[TEXT_COL].apply(tokenize)
    token_lengths = tokens_series.apply(len)

    all_tokens = [token for tokens in tokens_series for token in tokens]
    token_counter = Counter(all_tokens)
    unique_words = len(token_counter)
    top_words = token_counter.most_common(50)

    if all_tokens:
        stopword_count = sum(1 for token in all_tokens if token in VI_STOPWORDS)
        stopword_ratio = stopword_count / len(all_tokens)
    else:
        stopword_count = 0
        stopword_ratio = 0.0

    non_diacritic_ratio = df[TEXT_COL].apply(is_non_diacritic_comment).mean() if total_samples else 0.0
    emoji_ratio = df[TEXT_COL].str.contains(EMOJI_PATTERN, regex=True).mean() if total_samples else 0.0
    url_ratio = df[TEXT_COL].str.contains(URL_PATTERN, regex=True).mean() if total_samples else 0.0
    slang_ratio = df[TEXT_COL].apply(contains_slang).mean() if total_samples else 0.0

    cooccurrence_counts = {}
    for c in CONSTRUCT_LABELS:
        for t in TOXICITY_LABELS:
            key = f"C{c}_T{t}"
            cooccurrence_counts[key] = int(((df[CONSTRUCT_COL] == c) & (df[TOXICITY_COL] == t)).sum())

    split_summary_rows = []
    for split_name, split_df in df.groupby("split"):
        row = {
            "split": split_name,
            "total_samples": int(len(split_df)),
        }
        for c in CONSTRUCT_LABELS:
            row[f"construct_{c}"] = int((split_df[CONSTRUCT_COL] == c).sum())
        for t in TOXICITY_LABELS:
            row[f"toxicity_{t}"] = int((split_df[TOXICITY_COL] == t).sum())
        split_summary_rows.append(row)

    split_counts_df = pd.DataFrame(split_summary_rows).sort_values("split")
    split_counts_df.to_csv(OUTPUT_DIR / "split_label_counts.csv", index=False, encoding="utf-8")

    pd.DataFrame(top_words, columns=["word", "count"]).to_csv(
        OUTPUT_DIR / "top_words.csv", index=False, encoding="utf-8"
    )

    unknown_construct_labels = sorted(set(df[CONSTRUCT_COL].unique()) - set(CONSTRUCT_LABELS))
    unknown_toxicity_labels = sorted(set(df[TOXICITY_COL].unique()) - set(TOXICITY_LABELS))

    stats = {
        "total_samples": total_samples,
        "label_schema": {
            "Constructiveness": {
                "labels": CONSTRUCT_LABELS,
                "label_names": {str(k): v for k, v in CONSTRUCT_LABEL_NAME.items()},
            },
            "Toxicity": {
                "labels": TOXICITY_LABELS,
                "label_names": {str(k): v for k, v in TOXICITY_LABEL_NAME.items()},
            },
        },
        "constructiveness_distribution": {
            "counts": {str(label): int(construct_counts.get(label, 0)) for label in CONSTRUCT_LABELS},
            "percent": {
                str(label): round((int(construct_counts.get(label, 0)) / total_samples * 100) if total_samples else 0.0, 4)
                for label in CONSTRUCT_LABELS
            },
        },
        "toxicity_distribution": {
            "counts": {str(label): int(toxicity_counts.get(label, 0)) for label in TOXICITY_LABELS},
            "percent": {
                str(label): round((int(toxicity_counts.get(label, 0)) / total_samples * 100) if total_samples else 0.0, 4)
                for label in TOXICITY_LABELS
            },
        },
        "label_cooccurrence_counts": cooccurrence_counts,
        "unknown_labels_detected": {
            "Constructiveness": unknown_construct_labels,
            "Toxicity": unknown_toxicity_labels,
        },
        "average_length_tokens": round(float(token_lengths.mean()) if total_samples else 0.0, 4),
        "max_length_tokens": int(token_lengths.max()) if total_samples else 0,
        "min_length_tokens": int(token_lengths.min()) if total_samples else 0,
        "vocabulary": {
            "unique_words": unique_words,
            "top_frequent_words": [{"word": w, "count": c} for w, c in top_words[:30]],
            "stopword_tokens": stopword_count,
            "stopword_ratio": round(stopword_ratio, 6),
            "slang_or_teen_code_ratio": round(slang_ratio, 6),
        },
        "comment_properties": {
            "percent_non_diacritic_comments": round(non_diacritic_ratio * 100, 4),
            "percent_with_emoji": round(emoji_ratio * 100, 4),
            "percent_with_url": round(url_ratio * 100, 4),
        },
    }

    with (OUTPUT_DIR / "analysis_summary.json").open("w", encoding="utf-8") as file:
        json.dump(stats, file, ensure_ascii=False, indent=2)

    text_lines = [
        "=== ViCTSD Dataset Analysis Summary ===",
        f"Total samples: {stats['total_samples']}",
        f"Constructiveness counts: {stats['constructiveness_distribution']['counts']}",
        f"Constructiveness (%): {stats['constructiveness_distribution']['percent']}",
        f"Toxicity counts: {stats['toxicity_distribution']['counts']}",
        f"Toxicity (%): {stats['toxicity_distribution']['percent']}",
        f"Label co-occurrence counts: {stats['label_cooccurrence_counts']}",
        "",
        f"Unknown Constructiveness labels found: {unknown_construct_labels}",
        f"Unknown Toxicity labels found: {unknown_toxicity_labels}",
        "",
        f"Average length (tokens): {stats['average_length_tokens']}",
        f"Max length (tokens): {stats['max_length_tokens']}",
        f"Min length (tokens): {stats['min_length_tokens']}",
        "",
        f"Unique words: {stats['vocabulary']['unique_words']}",
        f"Stopwords ratio: {stats['vocabulary']['stopword_ratio']:.6f}",
        f"Slang/teen-code ratio: {stats['vocabulary']['slang_or_teen_code_ratio']:.6f}",
        "",
        f"% comments without diacritics: {stats['comment_properties']['percent_non_diacritic_comments']:.4f}%",
        f"% comments with emoji: {stats['comment_properties']['percent_with_emoji']:.4f}%",
        f"% comments with URL: {stats['comment_properties']['percent_with_url']:.4f}%",
    ]
    (OUTPUT_DIR / "analysis_summary.txt").write_text("\n".join(text_lines), encoding="utf-8")

    plot_construct_distribution(construct_counts)
    plot_toxicity_distribution(toxicity_counts)
    plot_length_histogram(token_lengths)
    plot_top_words(top_words, top_n=20)
    plot_label_cooccurrence_matrix(df)

    print("Analysis done. Outputs saved to:")
    print(f"- {OUTPUT_DIR.resolve()}")
    print("Generated plots:")
    print(f"- {(PLOTS_DIR / 'constructiveness_distribution_bar_chart.png').resolve()}")
    print(f"- {(PLOTS_DIR / 'toxicity_distribution_bar_chart.png').resolve()}")
    print(f"- {(PLOTS_DIR / 'comment_length_histogram.png').resolve()}")
    print(f"- {(PLOTS_DIR / 'word_frequency_chart.png').resolve()}")
    print(f"- {(PLOTS_DIR / 'label_cooccurrence_matrix.png').resolve()}")


if __name__ == "__main__":
    main()
