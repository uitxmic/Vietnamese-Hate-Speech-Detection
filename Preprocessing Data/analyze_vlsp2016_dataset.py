import csv
import json
import math
import re
import subprocess
import sys
import unicodedata
from collections import Counter
from html import escape
from pathlib import Path


DATA_DIR = Path("VLSP 2016 Dataset")
OUTPUT_DIR = Path("vlsp2016_analysis_outputs")
PLOTS_DIR = OUTPUT_DIR / "plots"

CSV_FILES = {
    "train": DATA_DIR / "train.csv",
    "dev": DATA_DIR / "dev.csv",
    "test": DATA_DIR / "test.csv",
}

TEXT_COL = "texts"
LABEL_COL = "labels"
EXPECTED_LABELS = ["NEG", "NEU", "POS"]
BINARY_NEGATIVE_MAPPING = {"NEG": 1, "NEU": 0, "POS": 0}

MERGED_CSV = OUTPUT_DIR / "vlsp2016_merged.csv"
HISTOGRAM_MAX_WORDS = 500
HISTOGRAM_BIN_WIDTH = 10

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

VI_STOPWORDS_ASCII = {
    "a",
    "ai",
    "anh",
    "ay",
    "bac",
    "ban",
    "bi",
    "cac",
    "cai",
    "chi",
    "cho",
    "chung",
    "co",
    "con",
    "cua",
    "cung",
    "da",
    "dang",
    "day",
    "de",
    "den",
    "di",
    "do",
    "duoc",
    "em",
    "hay",
    "ho",
    "k",
    "kia",
    "khi",
    "khong",
    "ko",
    "la",
    "lai",
    "ma",
    "may",
    "minh",
    "mot",
    "nay",
    "nen",
    "nha",
    "nhung",
    "no",
    "o",
    "oi",
    "qua",
    "ra",
    "rang",
    "rat",
    "se",
    "tao",
    "that",
    "theo",
    "thi",
    "thoi",
    "toi",
    "trong",
    "tu",
    "u",
    "va",
    "van",
    "vao",
    "vi",
    "voi",
}

SLANG_PATTERNS = [
    r"\bko\b",
    r"\bk\b",
    r"\bkhum\b",
    r"\bhok\b",
    r"\bj\b",
    r"\bz\b",
    r"\bvk\b",
    r"\bck\b",
    r"\bcmnr\b",
    r"\bvl\b",
    r"\bvcl\b",
    r"\blol\b",
    r"\blmao\b",
    r"\bwtf\b",
    r"\bdm\b",
    r"\bclm\b",
    r"\bcc\b",
    r"\bkg\b",
    r"\bhem\b",
    r"\bhong\b",
    r"\bdc\b",
    r"\bntn\b",
    r"\bib\b",
    r"\brep\b",
    r"\bad\b",
    r":\)+",
    r"=\)+",
    r":3",
    r":v",
    r"\bkaka+\b",
    r"\bhihi+\b",
    r"\bhe\s*he\b",
]
SLANG_REGEXES = [re.compile(pattern, re.IGNORECASE) for pattern in SLANG_PATTERNS]


def strip_accents(text: str) -> str:
    normalized = unicodedata.normalize("NFD", text)
    return "".join(ch for ch in normalized if unicodedata.category(ch) != "Mn")


def normalize_for_matching(text: str) -> str:
    text = strip_accents(text.lower())
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    return re.sub(r"\s+", " ", text).strip()


def is_non_diacritic_comment(text: str) -> bool:
    letters = "".join(ch for ch in text if ch.isalpha())
    if not letters:
        return False
    return strip_accents(letters).lower() == letters.lower()


def tokenize(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(text.lower())


def contains_slang(text: str) -> bool:
    normalized = normalize_for_matching(text)
    return any(regex.search(normalized) for regex in SLANG_REGEXES)


def percent(part: int | float, total: int | float) -> float:
    return round((part / total * 100) if total else 0.0, 4)


def ensure_dirs() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def load_dataset() -> list[dict[str, str | int]]:
    rows: list[dict[str, str | int]] = []

    for split_name, csv_path in CSV_FILES.items():
        if not csv_path.exists():
            raise FileNotFoundError(f"Missing file: {csv_path}")

        with csv_path.open("r", encoding="utf-8-sig", newline="") as file:
            reader = csv.DictReader(file)
            fieldnames = reader.fieldnames or []
            missing_cols = [col for col in (TEXT_COL, LABEL_COL) if col not in fieldnames]
            if missing_cols:
                raise ValueError(f"{csv_path} must contain columns: {TEXT_COL}, {LABEL_COL}")

            for row in reader:
                text = (row.get(TEXT_COL) or "").replace("\ufeff", "").strip()
                label = (row.get(LABEL_COL) or "").strip().upper()
                binary_label = BINARY_NEGATIVE_MAPPING.get(label, "")
                rows.append(
                    {
                        "split": split_name,
                        TEXT_COL: text,
                        LABEL_COL: label,
                        "binary_negative": binary_label,
                    }
                )

    return rows


def write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_merged_dataset(rows: list[dict[str, str | int]]) -> None:
    write_csv(MERGED_CSV, ["split", TEXT_COL, LABEL_COL, "binary_negative"], rows)


def write_bar_chart_svg(
    labels: list[str],
    values: list[int],
    title: str,
    path: Path,
    colors: list[str] | None = None,
) -> None:
    width, height = 860, 520
    margin_left, margin_right, margin_top, margin_bottom = 90, 40, 70, 95
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom
    max_value = max(values) if values else 1
    max_value = max(max_value, 1)
    colors = colors or ["#4C78A8", "#F58518", "#54A24B", "#E45756"]

    bar_gap = 30
    bar_width = (plot_width - bar_gap * (len(values) + 1)) / max(len(values), 1)
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width / 2}" y="35" text-anchor="middle" font-family="Arial" font-size="22" font-weight="700">{escape(title)}</text>',
        f'<line x1="{margin_left}" y1="{margin_top + plot_height}" x2="{margin_left + plot_width}" y2="{margin_top + plot_height}" stroke="#333"/>',
        f'<line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{margin_top + plot_height}" stroke="#333"/>',
    ]

    for tick in range(6):
        value = max_value * tick / 5
        y = margin_top + plot_height - (value / max_value * plot_height)
        svg.append(f'<line x1="{margin_left - 5}" y1="{y:.2f}" x2="{margin_left + plot_width}" y2="{y:.2f}" stroke="#ddd"/>')
        svg.append(
            f'<text x="{margin_left - 12}" y="{y + 4:.2f}" text-anchor="end" font-family="Arial" font-size="12" fill="#555">{int(round(value))}</text>'
        )

    for idx, (label, value) in enumerate(zip(labels, values)):
        x = margin_left + bar_gap + idx * (bar_width + bar_gap)
        bar_height = value / max_value * plot_height
        y = margin_top + plot_height - bar_height
        color = colors[idx % len(colors)]
        svg.extend(
            [
                f'<rect x="{x:.2f}" y="{y:.2f}" width="{bar_width:.2f}" height="{bar_height:.2f}" fill="{color}"/>',
                f'<text x="{x + bar_width / 2:.2f}" y="{y - 8:.2f}" text-anchor="middle" font-family="Arial" font-size="13" fill="#222">{value}</text>',
                f'<text x="{x + bar_width / 2:.2f}" y="{margin_top + plot_height + 32}" text-anchor="middle" font-family="Arial" font-size="13" fill="#222">{escape(label)}</text>',
            ]
        )

    svg.append("</svg>")
    path.write_text("\n".join(svg), encoding="utf-8")


def write_horizontal_bar_chart_svg(
    items: list[tuple[str, int]],
    title: str,
    path: Path,
    top_n: int = 20,
) -> None:
    items = items[:top_n]
    width = 980
    row_height = 26
    margin_left, margin_right, margin_top, margin_bottom = 170, 55, 70, 35
    height = margin_top + margin_bottom + max(1, len(items)) * row_height
    plot_width = width - margin_left - margin_right
    max_value = max((value for _, value in items), default=1)

    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width / 2}" y="35" text-anchor="middle" font-family="Arial" font-size="22" font-weight="700">{escape(title)}</text>',
    ]

    for idx, (word, value) in enumerate(items):
        y = margin_top + idx * row_height
        bar_width = value / max_value * plot_width if max_value else 0
        svg.extend(
            [
                f'<text x="{margin_left - 12}" y="{y + 17}" text-anchor="end" font-family="Arial" font-size="13" fill="#222">{escape(word)}</text>',
                f'<rect x="{margin_left}" y="{y + 3}" width="{bar_width:.2f}" height="18" fill="#F58518"/>',
                f'<text x="{margin_left + bar_width + 6:.2f}" y="{y + 17}" font-family="Arial" font-size="12" fill="#333">{value}</text>',
            ]
        )

    svg.append("</svg>")
    path.write_text("\n".join(svg), encoding="utf-8")


def make_histogram(
    values: list[int],
    bins: int = 40,
    min_value: int | None = None,
    max_value: int | None = None,
    bin_width: int | None = None,
) -> tuple[list[str], list[int]]:
    if not values:
        return ["0"], [0]

    if max_value is not None and bin_width is not None:
        start_value = 0 if min_value is None else min_value
        actual_bins = max(1, math.ceil((max_value - start_value) / bin_width))
        counts = [0] * actual_bins

        for value in values:
            if value < start_value or value > max_value:
                continue
            idx = int((value - start_value) / bin_width)
            idx = min(idx, actual_bins - 1)
            counts[idx] += 1

        labels = []
        for idx in range(actual_bins):
            start = start_value + idx * bin_width
            end = start + bin_width - 1
            if idx == actual_bins - 1:
                end = max_value
            labels.append(f"{start}-{end}" if start != end else str(start))

        return labels, counts

    min_value = min(values)
    max_value = max(values)
    if min_value == max_value:
        return [str(min_value)], [len(values)]

    bin_count = min(bins, max_value - min_value + 1)
    bin_width = (max_value - min_value + 1) / bin_count
    counts = [0] * bin_count

    for value in values:
        idx = int((value - min_value) / bin_width)
        idx = min(idx, bin_count - 1)
        counts[idx] += 1

    labels = []
    for idx in range(bin_count):
        start = math.floor(min_value + idx * bin_width)
        end = math.floor(min_value + (idx + 1) * bin_width - 1)
        if idx == bin_count - 1:
            end = max_value
        labels.append(f"{start}-{end}" if start != end else str(start))

    return labels, counts


def write_histogram_svg(
    values: list[int],
    title: str,
    path: Path,
    max_words: int | None = None,
    bin_width: int | None = None,
) -> None:
    labels, counts = make_histogram(values, min_value=0, max_value=max_words, bin_width=bin_width)
    omitted_count = sum(1 for value in values if max_words is not None and value > max_words)
    width, height = 980, 520
    margin_left, margin_right, margin_top, margin_bottom = 75, 35, 70, 85
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom
    max_count = max(counts) if counts else 1
    max_count = max(max_count, 1)
    bar_width = plot_width / max(len(counts), 1)

    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width / 2}" y="35" text-anchor="middle" font-family="Arial" font-size="22" font-weight="700">{escape(title)}</text>',
        f'<line x1="{margin_left}" y1="{margin_top + plot_height}" x2="{margin_left + plot_width}" y2="{margin_top + plot_height}" stroke="#333"/>',
        f'<line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{margin_top + plot_height}" stroke="#333"/>',
    ]
    if max_words is not None:
        note = f"0-{max_words} words shown; {omitted_count} texts >{max_words} omitted"
        svg.append(
            f'<text x="{margin_left + plot_width - 10}" y="58" text-anchor="end" font-family="Arial" font-size="12" fill="#666">{escape(note)}</text>'
        )

    for tick in range(6):
        value = max_count * tick / 5
        y = margin_top + plot_height - (value / max_count * plot_height)
        svg.append(f'<line x1="{margin_left - 5}" y1="{y:.2f}" x2="{margin_left + plot_width}" y2="{y:.2f}" stroke="#ddd"/>')
        svg.append(
            f'<text x="{margin_left - 12}" y="{y + 4:.2f}" text-anchor="end" font-family="Arial" font-size="12" fill="#555">{int(round(value))}</text>'
        )

    for idx, count in enumerate(counts):
        x = margin_left + idx * bar_width
        bar_height = count / max_count * plot_height
        y = margin_top + plot_height - bar_height
        svg.append(
            f'<rect x="{x + 1:.2f}" y="{y:.2f}" width="{max(bar_width - 2, 1):.2f}" height="{bar_height:.2f}" fill="#4C78A8" opacity="0.85"/>'
        )

    if labels:
        label_step = max(1, len(labels) // 8)
        label_indices = list(range(0, len(labels), label_step))
        if (len(labels) - 1) not in label_indices:
            label_indices = [idx for idx in label_indices if (len(labels) - 1 - idx) >= label_step]
            label_indices.append(len(labels) - 1)
        for idx in label_indices:
            x = margin_left + idx * bar_width + bar_width / 2
            svg.append(
                f'<text x="{x:.2f}" y="{margin_top + plot_height + 26}" text-anchor="middle" font-family="Arial" font-size="11" fill="#555">{escape(labels[idx])}</text>'
            )
    svg.append(
        f'<text x="{width / 2}" y="{height - 18}" text-anchor="middle" font-family="Arial" font-size="13" fill="#222">Number of words</text>'
    )

    svg.append("</svg>")
    path.write_text("\n".join(svg), encoding="utf-8")


def summarize(rows: list[dict[str, str | int]]) -> dict:
    total_samples = len(rows)
    texts = [str(row[TEXT_COL]) for row in rows]
    labels = [str(row[LABEL_COL]) for row in rows]
    binary_labels = [row["binary_negative"] for row in rows if row["binary_negative"] != ""]

    label_counts = Counter(labels)
    binary_counts = Counter(binary_labels)
    split_counts = Counter(str(row["split"]) for row in rows)
    unknown_labels = sorted(set(labels) - set(EXPECTED_LABELS))

    tokens_by_text = [tokenize(text) for text in texts]
    token_lengths = [len(tokens) for tokens in tokens_by_text]
    char_lengths = [len(text) for text in texts]
    all_tokens = [token for tokens in tokens_by_text for token in tokens]
    token_counter = Counter(all_tokens)
    top_words = token_counter.most_common(50)
    unique_words = len(token_counter)

    stopword_count = sum(
        1 for token in all_tokens if normalize_for_matching(token) in VI_STOPWORDS_ASCII
    )
    stopword_ratio = stopword_count / len(all_tokens) if all_tokens else 0.0

    non_diacritic_count = sum(1 for text in texts if is_non_diacritic_comment(text))
    emoji_count = sum(1 for text in texts if EMOJI_PATTERN.search(text))
    url_count = sum(1 for text in texts if URL_PATTERN.search(text))
    slang_count = sum(1 for text in texts if contains_slang(text))
    empty_text_count = sum(1 for text in texts if not text.strip())

    normalized_texts = [normalize_for_matching(text) for text in texts if text.strip()]
    duplicate_text_count = len(normalized_texts) - len(set(normalized_texts))

    label_length_stats = {}
    for label in EXPECTED_LABELS:
        label_lengths = [
            length for length, row_label in zip(token_lengths, labels) if row_label == label
        ]
        label_length_stats[label] = {
            "average_length_tokens": round(sum(label_lengths) / len(label_lengths), 4)
            if label_lengths
            else 0.0,
            "max_length_tokens": max(label_lengths) if label_lengths else 0,
            "min_length_tokens": min(label_lengths) if label_lengths else 0,
        }

    split_rows = []
    for split in CSV_FILES:
        split_rows_for_name = [row for row in rows if row["split"] == split]
        split_labels = Counter(str(row[LABEL_COL]) for row in split_rows_for_name)
        split_binary = Counter(
            row["binary_negative"]
            for row in split_rows_for_name
            if row["binary_negative"] != ""
        )
        split_token_lengths = [
            len(tokenize(str(row[TEXT_COL]))) for row in split_rows_for_name
        ]
        split_rows.append(
            {
                "split": split,
                "total_samples": len(split_rows_for_name),
                **{f"label_{label}": split_labels.get(label, 0) for label in EXPECTED_LABELS},
                "unknown_labels": sum(
                    count
                    for label, count in split_labels.items()
                    if label not in EXPECTED_LABELS
                ),
                "binary_0_non_negative": split_binary.get(0, 0),
                "binary_1_negative": split_binary.get(1, 0),
                "average_length_tokens": round(
                    sum(split_token_lengths) / len(split_token_lengths), 4
                )
                if split_token_lengths
                else 0.0,
            }
        )

    write_csv(
        OUTPUT_DIR / "split_label_counts.csv",
        [
            "split",
            "total_samples",
            "label_NEG",
            "label_NEU",
            "label_POS",
            "unknown_labels",
            "binary_0_non_negative",
            "binary_1_negative",
            "average_length_tokens",
        ],
        split_rows,
    )
    write_csv(
        OUTPUT_DIR / "top_words.csv",
        ["word", "count"],
        [{"word": word, "count": count} for word, count in top_words],
    )

    return {
        "total_samples": total_samples,
        "source_files": {split: str(path) for split, path in CSV_FILES.items()},
        "merged_csv": str(MERGED_CSV),
        "label_schema": {
            "original_sentiment_labels": EXPECTED_LABELS,
            "binary_negative_mapping": {
                label: BINARY_NEGATIVE_MAPPING[label] for label in EXPECTED_LABELS
            },
        },
        "split_counts": dict(split_counts),
        "label_distribution": {
            "counts": {label: label_counts.get(label, 0) for label in EXPECTED_LABELS},
            "percent": {
                label: percent(label_counts.get(label, 0), total_samples)
                for label in EXPECTED_LABELS
            },
        },
        "binary_negative_distribution": {
            "counts": {
                "0_non_negative_POS_or_NEU": binary_counts.get(0, 0),
                "1_negative_NEG": binary_counts.get(1, 0),
            },
            "percent": {
                "0_non_negative_POS_or_NEU": percent(binary_counts.get(0, 0), len(binary_labels)),
                "1_negative_NEG": percent(binary_counts.get(1, 0), len(binary_labels)),
            },
        },
        "unknown_labels_detected": unknown_labels,
        "average_length_tokens": round(sum(token_lengths) / total_samples, 4)
        if total_samples
        else 0.0,
        "max_length_tokens": max(token_lengths) if token_lengths else 0,
        "min_length_tokens": min(token_lengths) if token_lengths else 0,
        "average_length_characters": round(sum(char_lengths) / total_samples, 4)
        if total_samples
        else 0.0,
        "max_length_characters": max(char_lengths) if char_lengths else 0,
        "min_length_characters": min(char_lengths) if char_lengths else 0,
        "length_by_label": label_length_stats,
        "vocabulary": {
            "unique_words": unique_words,
            "top_frequent_words": [
                {"word": word, "count": count} for word, count in top_words[:30]
            ],
            "stopword_tokens": stopword_count,
            "stopword_ratio": round(stopword_ratio, 6),
            "slang_or_teen_code_ratio": round(slang_count / total_samples, 6)
            if total_samples
            else 0.0,
        },
        "text_properties": {
            "empty_text_count": empty_text_count,
            "duplicate_text_count_after_normalization": duplicate_text_count,
            "percent_non_diacritic_texts": percent(non_diacritic_count, total_samples),
            "percent_with_emoji": percent(emoji_count, total_samples),
            "percent_with_url": percent(url_count, total_samples),
        },
        "_plot_inputs": {
            "label_counts": [label_counts.get(label, 0) for label in EXPECTED_LABELS],
            "binary_counts": [binary_counts.get(0, 0), binary_counts.get(1, 0)],
            "token_lengths": token_lengths,
            "top_words": top_words,
        },
    }


def write_summary_files(stats: dict) -> None:
    plot_inputs = stats.pop("_plot_inputs")

    with (OUTPUT_DIR / "analysis_summary.json").open("w", encoding="utf-8") as file:
        json.dump(stats, file, ensure_ascii=False, indent=2)

    text_lines = [
        "=== VLSP 2016 Dataset Analysis Summary ===",
        f"Total samples: {stats['total_samples']}",
        f"Merged CSV: {stats['merged_csv']}",
        f"Split counts: {stats['split_counts']}",
        "",
        f"Sentiment label counts: {stats['label_distribution']['counts']}",
        f"Sentiment label distribution (%): {stats['label_distribution']['percent']}",
        f"Binary mapping: {stats['label_schema']['binary_negative_mapping']}",
        f"Binary label counts: {stats['binary_negative_distribution']['counts']}",
        f"Binary label distribution (%): {stats['binary_negative_distribution']['percent']}",
        "",
        f"Unknown labels found: {stats['unknown_labels_detected']}",
        "",
        f"Average length (tokens): {stats['average_length_tokens']}",
        f"Max length (tokens): {stats['max_length_tokens']}",
        f"Min length (tokens): {stats['min_length_tokens']}",
        f"Average length (characters): {stats['average_length_characters']}",
        f"Max length (characters): {stats['max_length_characters']}",
        f"Min length (characters): {stats['min_length_characters']}",
        "",
        f"Length by label: {stats['length_by_label']}",
        "",
        f"Unique words: {stats['vocabulary']['unique_words']}",
        f"Stopwords ratio: {stats['vocabulary']['stopword_ratio']:.6f}",
        f"Slang/teen-code ratio: {stats['vocabulary']['slang_or_teen_code_ratio']:.6f}",
        "",
        f"Empty texts: {stats['text_properties']['empty_text_count']}",
        f"Duplicate texts after normalization: {stats['text_properties']['duplicate_text_count_after_normalization']}",
        f"% texts without diacritics: {stats['text_properties']['percent_non_diacritic_texts']:.4f}%",
        f"% texts with emoji: {stats['text_properties']['percent_with_emoji']:.4f}%",
        f"% texts with URL: {stats['text_properties']['percent_with_url']:.4f}%",
    ]
    (OUTPUT_DIR / "analysis_summary.txt").write_text("\n".join(text_lines), encoding="utf-8")

    write_bar_chart_svg(
        EXPECTED_LABELS,
        plot_inputs["label_counts"],
        "VLSP 2016 Sentiment Label Distribution",
        PLOTS_DIR / "sentiment_label_distribution_bar_chart.svg",
        colors=["#E45756", "#4C78A8", "#54A24B"],
    )
    write_bar_chart_svg(
        ["0 non-negative", "1 negative"],
        plot_inputs["binary_counts"],
        "VLSP 2016 Binary Negative Distribution",
        PLOTS_DIR / "binary_negative_distribution_bar_chart.svg",
        colors=["#54A24B", "#E45756"],
    )
    write_histogram_svg(
        plot_inputs["token_lengths"],
        "VLSP 2016 Text Length Distribution",
        PLOTS_DIR / "text_length_histogram.svg",
        max_words=HISTOGRAM_MAX_WORDS,
        bin_width=HISTOGRAM_BIN_WIDTH,
    )
    write_horizontal_bar_chart_svg(
        plot_inputs["top_words"],
        "VLSP 2016 Top 20 Most Frequent Words",
        PLOTS_DIR / "word_frequency_chart.svg",
        top_n=20,
    )


def generate_png_plots_if_possible() -> bool:
    script_path = Path("generate_vlsp2016_png_plots.ps1")
    if not script_path.exists():
        print(f"Warning: PNG plot helper not found: {script_path}", file=sys.stderr)
        return False

    try:
        result = subprocess.run(
            [
                "powershell.exe",
                "-NoProfile",
                "-ExecutionPolicy",
                "Bypass",
                "-File",
                str(script_path),
                "-OutputDir",
                str(OUTPUT_DIR),
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
    except OSError as exc:
        print(f"Warning: could not generate PNG plots: {exc}", file=sys.stderr)
        return False

    if result.returncode != 0:
        message = result.stderr.strip() or result.stdout.strip()
        print(f"Warning: PNG plot generation failed: {message}", file=sys.stderr)
        return False

    return True


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    ensure_dirs()
    rows = load_dataset()
    write_merged_dataset(rows)
    stats = summarize(rows)
    write_summary_files(stats)
    png_generated = generate_png_plots_if_possible()

    print("Analysis done. Outputs saved to:")
    print(f"- {OUTPUT_DIR.resolve()}")
    print("Merged dataset:")
    print(f"- {MERGED_CSV.resolve()}")
    print("Generated plots:")
    print(f"- {(PLOTS_DIR / 'sentiment_label_distribution_bar_chart.svg').resolve()}")
    print(f"- {(PLOTS_DIR / 'binary_negative_distribution_bar_chart.svg').resolve()}")
    print(f"- {(PLOTS_DIR / 'text_length_histogram.svg').resolve()}")
    print(f"- {(PLOTS_DIR / 'word_frequency_chart.svg').resolve()}")
    if png_generated:
        print("Generated PNG plots:")
        print(f"- {(PLOTS_DIR / 'sentiment_label_distribution_bar_chart.png').resolve()}")
        print(f"- {(PLOTS_DIR / 'binary_negative_distribution_bar_chart.png').resolve()}")
        print(f"- {(PLOTS_DIR / 'text_length_histogram.png').resolve()}")
        print(f"- {(PLOTS_DIR / 'word_frequency_chart.png').resolve()}")


if __name__ == "__main__":
    main()
