import pandas as pd
import pytest

from hsd.config import load_config
from hsd.splits import assign_splits, check_no_leakage


def _index(n_patients=40):
    rows = []
    for i in range(n_patients):
        murmur = ["Present", "Absent"][i % 2]
        if i % 9 == 0:
            murmur = "Unknown"
        for site in ["AV", "MV"]:
            rows.append({"dataset": "circor", "path": f"c/{i}_{site}.wav", "rec_id": f"{i}_{site}",
                         "patient_id": str(i), "site": site, "label": int(murmur == "Present" and site == "AV"),
                         "class_name": murmur, "status": "unknown" if murmur == "Unknown" else "ok",
                         "duration": 6.0, "patient_murmur": murmur})
    for i in range(30):
        cls = i % 5
        rows.append({"dataset": "yaseen", "path": f"y/{i}.wav", "rec_id": f"y/{i}.wav", "patient_id": f"y/{i}.wav",
                     "site": "", "label": cls, "class_name": str(cls), "status": "ok", "duration": 4.0,
                     "patient_murmur": ""})
    return pd.DataFrame(rows)


def test_patient_level_split_has_no_leakage():
    cfg = load_config("configs/base.yaml")
    idx = assign_splits(_index(), cfg)
    check_no_leakage(idx)
    c = idx[idx.dataset == "circor"]
    assert set(c[c.status == "unknown"].split) == {"unknown"}
    trained = c[c.split.isin(["train", "val", "test"])]
    assert trained.groupby("patient_id").split.nunique().max() == 1      # one split per patient


def test_yaseen_split_is_by_recording_and_stratified():
    cfg = load_config("configs/base.yaml")
    idx = assign_splits(_index(), cfg)
    y = idx[idx.dataset == "yaseen"]
    assert set(y.split) <= {"train", "val", "test"}
    assert set(y[y.split == "train"].label) == {0, 1, 2, 3, 4}


def test_leakage_check_catches_overlap():
    """One recording of a patient in test while the patient's other recording stays in train must raise."""
    cfg = load_config("configs/base.yaml")
    idx = assign_splits(_index(), cfg)
    c = idx[(idx.dataset == "circor") & (idx.split == "train")]
    pid = c.patient_id.iloc[0]
    rows = idx[(idx.dataset == "circor") & (idx.patient_id == pid)]
    assert len(rows) >= 2
    idx.loc[rows.index[0], "split"] = "test"
    with pytest.raises(AssertionError):
        check_no_leakage(idx)
