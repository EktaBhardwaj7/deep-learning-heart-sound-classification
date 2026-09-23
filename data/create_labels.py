from pathlib import Path
import csv


def get_label(header_file):
    """
    Read the label from a PhysioNet .hea file.

    Normal    -> 0
    Abnormal  -> 1
    """

    with open(header_file, "r") as f:
        for line in f:
            line = line.strip().lower()

            if line == "# normal":
                return 0

            if line == "# abnormal":
                return 1

    return None


def create_label_csv(data_folder, output_file):
    data_folder = Path(data_folder)

    records = []

    for header_file in sorted(data_folder.glob("*.hea")):
        label = get_label(header_file)

        if label is not None:
            records.append([
                header_file.stem,
                label
            ])

    with open(output_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["recording", "label"])
        writer.writerows(records)

    print(f"Created {output_file}")
    print(f"Total recordings: {len(records)}")


if __name__ == "__main__":
    create_label_csv(
        "data/physionet2016/training-f",
        "data/labels.csv"
    )
