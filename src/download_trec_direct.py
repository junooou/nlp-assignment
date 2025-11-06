"""
Task 0 — Download TREC via code (no torchtext), split 80/20, save CSVs.
This follows the assignment’s allowed alternative (download from original site).
"""

import csv, random, os
from pathlib import Path
import requests

TRAIN_URL = "https://cogcomp.seas.upenn.edu/Data/QA/QC/train_5500.label"
TEST_URL  = "https://cogcomp.seas.upenn.edu/Data/QA/QC/TREC_10.label"

root = Path(__file__).resolve().parent
data_dir = root / "data"
data_dir.mkdir(exist_ok=True)

def fetch(url, out_path):
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    # Save raw text exactly
    out_path.write_bytes(r.content)

def read_lines_any_encoding(path):
    # Try UTF-8, fallback to latin-1 to avoid decoding issues
    try:
        return path.read_text(encoding="utf-8").splitlines()
    except UnicodeDecodeError:
        return path.read_text(encoding="latin-1").splitlines()

def parse_trec_lines(lines):
    """
    Each line looks like:
      DESC:manner How did serfdom develop in and then leave Russia ?
    We use the part before ':' as the coarse label (e.g., 'DESC') and the rest as text.
    """
    rows = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        # split once on first space
        if " " not in line:
            continue
        label_full, text = line.split(" ", 1)
        coarse = label_full.split(":", 1)[0]
        rows.append((text, coarse))
    return rows

def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["text", "label"])
        w.writerows(rows)

#downloading
train_label_path = data_dir / "train_5500.label"
test_label_path  = data_dir / "TREC_10.label"

print("Downloading training set...")
fetch(TRAIN_URL, train_label_path)
print("Downloading test set...")
fetch(TEST_URL, test_label_path)

#parsing
train_lines = read_lines_any_encoding(train_label_path)
test_lines  = read_lines_any_encoding(test_label_path)

train_data = parse_trec_lines(train_lines)
test_data  = parse_trec_lines(test_lines)

#splitting data
random.seed(42)
random.shuffle(train_data)
n_val = int(len(train_data) * 0.2)
val_data = train_data[:n_val]
final_train = train_data[n_val:]

#saving csvs
write_csv(data_dir / "trec_train.csv", final_train)
write_csv(data_dir / "trec_val.csv",   val_data)
write_csv(data_dir / "trec_test.csv",  test_data)

print(f"Done! CSVs saved in {data_dir.resolve()}")
print(f"Train: {len(final_train)} | Val: {len(val_data)} | Test: {len(test_data)}")
