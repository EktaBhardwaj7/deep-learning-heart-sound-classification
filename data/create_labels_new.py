from pathlib import Path
import csv

DATASET_DIR = Path("data/heart_sound_dataset")
OUTPUT_FILE = Path("data/labels.csv")

# 5-class mapping
CLASS_TO_LABEL = {
    "AS": 0,
    "MR": 1,
    "MS": 2,
    "MVP": 3,
    "N": 4,
}

rows = []

for class_name, label in CLASS_TO_LABEL.items():
    class_dir = DATASET_DIR / class_name

    for wav_file in sorted(class_dir.rglob("*.wav")):
        relative_path = wav_file.relative_to(DATASET_DIR)

        rows.append([
            relative_path.as_posix(),
            label,
            class_name
        ])

with OUTPUT_FILE.open("w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["filename", "label", "class"])
    writer.writerows(rows)

print(f"\nSaved {len(rows)} labels to {OUTPUT_FILE}")

print("\nClass distribution:")
for class_name, label in CLASS_TO_LABEL.items():
    count = sum(1 for row in rows if row[2] == class_name)
    print(f"{class_name}: {count}")
