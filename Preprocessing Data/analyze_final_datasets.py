import csv
from collections import Counter, defaultdict
from pathlib import Path


DATA_DIR = Path(__file__).resolve().parent

DATASETS = {
    "train": ["final_train.csv", "final_train.scv"],
    "valid": ["final_valid.csv"],
    "test": ["final_test.csv"],
}

TEXT_COLUMN = "text"
LABEL_COLUMN = "label"
MISSING_LABEL = "<missing>"


def find_dataset_file(candidates):
    for filename in candidates:
        path = DATA_DIR / filename
        if path.exists():
            return path
    raise FileNotFoundError(f"Khong tim thay file nao trong danh sach: {candidates}")


def read_csv(file_path):
    encodings = ("utf-8-sig", "utf-8", "cp1258", "latin-1")

    for encoding in encodings:
        try:
            with open(file_path, "r", encoding=encoding, newline="") as f:
                return list(csv.DictReader(f))
        except UnicodeDecodeError:
            continue

    raise ValueError(f"Khong doc duoc file {file_path} voi cac encoding: {encodings}")


def label_sort_key(label):
    try:
        return (0, float(label))
    except (TypeError, ValueError):
        return (1, str(label))


def analyze_dataset(split_name, file_path):
    rows = read_csv(file_path)
    columns = rows[0].keys() if rows else []

    if LABEL_COLUMN not in columns:
        raise ValueError(
            f"File {file_path.name} khong co cot nhan '{LABEL_COLUMN}'. "
            f"Cac cot hien co: {list(columns)}"
        )

    total_samples = len(rows)
    has_text_column = TEXT_COLUMN in columns
    missing_text = (
        sum(not row.get(TEXT_COLUMN, "").strip() for row in rows)
        if has_text_column
        else None
    )

    labels = [
        row.get(LABEL_COLUMN, "").strip() or MISSING_LABEL
        for row in rows
    ]
    label_counts = Counter(labels)
    missing_label = label_counts.get(MISSING_LABEL, 0)
    unique_labels = len([label for label in label_counts if label != MISSING_LABEL])

    print(f"\n{'=' * 60}")
    print(f"Tap du lieu: {split_name.upper()} ({file_path.name})")
    print(f"So luong mau: {total_samples:,}")
    print(f"So luong nhan khac nhau: {unique_labels:,}")

    if missing_text is not None:
        print(f"So dong thieu text: {missing_text:,}")
    print(f"So dong thieu label: {missing_label:,}")

    print("\nThong ke tung nhan:")
    print(f"{'label':>10} {'so_luong':>12} {'ty_le_%':>10}")
    for label in sorted(label_counts, key=label_sort_key):
        count = label_counts[label]
        percent = count / total_samples * 100 if total_samples else 0
        print(f"{label:>10} {count:>12,} {percent:>10.2f}")

    return {
        "split": split_name,
        "file": file_path.name,
        "samples": total_samples,
        "unique_labels": unique_labels,
        "missing_text": missing_text,
        "missing_label": missing_label,
        "label_counts": label_counts,
    }


def main():
    results = []

    for split_name, candidates in DATASETS.items():
        file_path = find_dataset_file(candidates)
        results.append(analyze_dataset(split_name, file_path))

    print(f"\n{'=' * 60}")
    print("Tong hop 3 tap")
    total_samples = sum(item["samples"] for item in results)
    print(f"Tong so mau: {total_samples:,}")

    all_labels = sorted(
        {label for item in results for label in item["label_counts"]},
        key=label_sort_key,
    )
    print("\nPhan bo nhan theo tung tap:")
    split_names = [item["split"] for item in results]
    counts_by_split = {
        item["split"]: defaultdict(int, item["label_counts"]) for item in results
    }

    header = f"{'label':>10} " + " ".join(f"{name:>10}" for name in split_names)
    header += f" {'tong':>10} {'ty_le_tong_%':>12}"
    print(header)

    for label in all_labels:
        row_counts = [counts_by_split[split][label] for split in split_names]
        label_total = sum(row_counts)
        percent = label_total / total_samples * 100 if total_samples else 0
        count_text = " ".join(f"{count:>10,}" for count in row_counts)
        print(f"{label:>10} {count_text} {label_total:>10,} {percent:>12.2f}")


if __name__ == "__main__":
    main()
