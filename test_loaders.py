import multiprocessing as mp
try:
    mp.set_start_method("fork", force=True)
except RuntimeError:
    pass

from pathlib import Path
from utils.multi_dataset import build_loaders


def main():
    loaders, info = build_loaders(
        yaseen_root=Path("data/heart_sound_dataset"),
        circor_root=Path("data/circor"),
        yaseen_csv=Path("data/labels_yaseen.csv"),
        circor_csv=Path("data/labels_circor.csv"),
    )
    print("Yaseen:", info["yaseen_counts"])
    print("CirCor:", info["circor_counts"])
    print("Input shape:", info["input_shape"])

    y_x, y_y = next(iter(loaders["yaseen_train"]))
    c_x, c_y = next(iter(loaders["circor_train"]))
    print("yaseen batch:", y_x.shape, y_y.shape, "mean=", float(y_x.mean().item()))
    print("circor batch:", c_x.shape, c_y.shape, "mean=", float(c_x.mean().item()))
    print("OK")


if __name__ == "__main__":
    main()
