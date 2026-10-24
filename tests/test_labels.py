"""Location-aware CirCor labelling and folder discovery, on a tiny dataset that mirrors the real layout."""
import numpy as np
import pandas as pd
import soundfile as sf
import pytest

from hsd.labels import build_circor_index, build_yaseen_index, expand_windows


def _wav(path, dur=6.0, sr=4000):
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, (0.1 * np.sin(np.arange(int(dur * sr)) / 10)).astype(np.float32), sr)


@pytest.fixture
def tiny(tmp_path):
    root = tmp_path / "data"
    circor = root / "circor"
    w = circor / "training_data" / "training_data"
    meta = [
        {"Patient ID": "1", "Murmur": "Present", "Murmur locations": "AV+MV"},
        {"Patient ID": "2", "Murmur": "Absent", "Murmur locations": "nan"},
        {"Patient ID": "3", "Murmur": "Unknown", "Murmur locations": "nan"},
        {"Patient ID": "4", "Murmur": "Present", "Murmur locations": "nan"},
    ]
    circor.mkdir(parents=True)
    pd.DataFrame(meta).to_csv(circor / "training_data.csv", index=False)
    for pid in ["1", "2", "3", "4"]:
        for site in ["AV", "PV", "TV", "MV"]:
            _wav(w / f"{pid}_{site}.wav")
    y = root / "heart_sound_dataset"
    for cls in ["AS", "N"]:
        _wav(y / cls / "sub" / f"{cls}_001.wav", dur=4.0, sr=8000)
    _wav(y / "source" / "not_a_class.wav", dur=4.0, sr=8000)   # must be ignored
    return root


def test_location_aware_labels(tiny):
    idx = build_circor_index(tiny, "circor", tiny / "circor" / "training_data.csv", "absent", 0.5)
    p1 = idx[idx.patient_id == "1"].set_index("site")
    assert p1.loc["AV", "label"] == 1 and p1.loc["MV", "label"] == 1          # listed sites
    assert p1.loc["PV", "label"] == 0 and p1.loc["PV", "status"] == "ok"      # other sites: absent (policy)
    assert set(idx[idx.patient_id == "3"].status) == {"unknown"}              # held out
    assert set(idx[idx.patient_id == "4"].status) == {"ambiguous_location"}   # no listed site


def test_nonsite_exclude_policy(tiny):
    idx = build_circor_index(tiny, "circor", tiny / "circor" / "training_data.csv", "exclude", 0.5)
    p1 = idx[idx.patient_id == "1"].set_index("site")
    assert p1.loc["PV", "status"] == "nonsite_excluded"


def test_yaseen_ignores_non_class_folders(tiny):
    idx = build_yaseen_index(tiny, "heart_sound_dataset", ["AS", "N"], 0.5)
    assert len(idx) == 2
    assert set(idx.class_name) == {"AS", "N"}
    assert not idx.path.str.contains("source").any()


def test_windows_cover_the_tail(tiny):
    idx = build_circor_index(tiny, "circor", tiny / "circor" / "training_data.csv", "absent", 0.5)
    w = expand_windows(idx.head(1), 5.0, 2.5)
    assert w.start.iloc[0] == 0.0
    assert w.start.iloc[-1] == pytest.approx(1.0)               # 6 s recording: last window starts at 1.0 s
    y = build_yaseen_index(tiny, "heart_sound_dataset", ["AS"], 0.5)
    c = expand_windows(y, 5.0, None)
    assert len(c) == 1 and c.start.iloc[0] == 0.0               # 4 s clip: one window, wrap-padded later
