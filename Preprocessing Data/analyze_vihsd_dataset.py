import json
import re
import unicodedata
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


DATA_DIR = Path("ViHSD Dataset")
OUTPUT_DIR = Path("vihsd_analysis_outputs")
PLOTS_DIR = OUTPUT_DIR / "plots"
HISTOGRAM_MAX_WORDS = 100
HISTOGRAM_BIN_WIDTH = 5

CSV_FILES = {
    "train": DATA_DIR / "vihsd_train.csv",
    "dev": DATA_DIR / "vihsd_dev.csv",
    "test": DATA_DIR / "vihsd_test.csv",
}

TEXT_COL = "free_text"
LABEL_COL = "label_id"
EXPECTED_LABELS = [0, 1, 2]

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
    for split_name, csv_path in CSV_FILES.items():
        if not csv_path.exists():
            raise FileNotFoundError(f"Missing file: {csv_path}")
        df = pd.read_csv(csv_path)
        if TEXT_COL not in df.columns or LABEL_COL not in df.columns:
            raise ValueError(f"{csv_path} must contain columns: {TEXT_COL}, {LABEL_COL}")
        df = df[[TEXT_COL, LABEL_COL]].copy()
        df[TEXT_COL] = df[TEXT_COL].fillna("").astype(str)
        df[LABEL_COL] = pd.to_numeric(df[LABEL_COL], errors="coerce").fillna(-1).astype(int)
        df["split"] = split_name
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def plot_class_distribution(class_counts: pd.Series) -> None:
    plt.figure(figsize=(7, 5))
    ax = class_counts.plot(kind="bar", color=["#4C78A8", "#F58518", "#54A24B"])
    plt.title("ViHSD Class Distribution")
    plt.xlabel("Label")
    plt.ylabel("Number of samples")
    plt.xticks(rotation=0)
    for patch in ax.patches:
        height = patch.get_height()
        ax.annotate(
            f"{int(height)}",
            (patch.get_x() + patch.get_width() / 2, height),
            ha="center",
            va="bottom",
            fontsize=9,
            xytext=(0, 3),
            textcoords="offset points",
        )
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "class_distribution_bar_chart.png", dpi=200)
    plt.close()


def plot_length_histogram(word_lengths: pd.Series) -> None:
    plt.figure(figsize=(8, 5))
    bins = range(0, HISTOGRAM_MAX_WORDS + HISTOGRAM_BIN_WIDTH, HISTOGRAM_BIN_WIDTH)
    plt.hist(word_lengths, bins=bins, color="#4C78A8", edgecolor="black", alpha=0.85)
    plt.title("Comment Length Distribution")
    plt.xlabel("Number of words")
    plt.ylabel("Frequency")
    plt.xlim(0, HISTOGRAM_MAX_WORDS)
    plt.xticks(range(0, HISTOGRAM_MAX_WORDS + 1, 10))
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


def save_note_if_not_multilabel() -> None:
    note = (
        "Dataset ViHSD is single-label (one label_id per comment), "
        "so label co-occurrence matrix is not applicable."
    )
    (OUTPUT_DIR / "label_cooccurrence_note.txt").write_text(note, encoding="utf-8")


def main() -> None:
    ensure_dirs()
    df = load_dataset()

    tokens_series = df[TEXT_COL].apply(tokenize)
    token_lengths = tokens_series.apply(len)

    class_counts = df[LABEL_COL].value_counts().reindex(EXPECTED_LABELS, fill_value=0).sort_index()
    total_samples = int(len(df))

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

    stats = {
        "total_samples": total_samples,
        "labels": EXPECTED_LABELS,
        "samples_per_label": {str(label): int(class_counts.get(label, 0)) for label in EXPECTED_LABELS},
        "class_distribution_percent": {
            str(label): round((int(class_counts.get(label, 0)) / total_samples * 100) if total_samples else 0.0, 4)
            for label in EXPECTED_LABELS
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

    split_counts = df.groupby(["split", LABEL_COL]).size().unstack(fill_value=0).reindex(columns=EXPECTED_LABELS, fill_value=0)
    split_counts.to_csv(OUTPUT_DIR / "split_label_counts.csv", encoding="utf-8")

    pd.DataFrame(top_words, columns=["word", "count"]).to_csv(
        OUTPUT_DIR / "top_words.csv", index=False, encoding="utf-8"
    )

    with (OUTPUT_DIR / "analysis_summary.json").open("w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    text_lines = [
        "=== ViHSD Dataset Analysis Summary ===",
        f"Total samples: {stats['total_samples']}",
        f"Label counts: {stats['samples_per_label']}",
        f"Class distribution (%): {stats['class_distribution_percent']}",
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

    plot_class_distribution(class_counts)
    plot_length_histogram(token_lengths)
    plot_top_words(top_words, top_n=20)
    save_note_if_not_multilabel()

    print("Analysis done. Outputs saved to:")
    print(f"- {OUTPUT_DIR.resolve()}")
    print("Generated plots:")
    print(f"- {(PLOTS_DIR / 'class_distribution_bar_chart.png').resolve()}")
    print(f"- {(PLOTS_DIR / 'comment_length_histogram.png').resolve()}")
    print(f"- {(PLOTS_DIR / 'word_frequency_chart.png').resolve()}")


if __name__ == "__main__":
    main()
