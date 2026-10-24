import pytest
from hsd.config import load_config


def test_extends_and_overrides():
    cfg = load_config("configs/rq1_raw.yaml", ["train.epochs=3", "model.base_channels=8"])
    assert cfg["features"]["representation"] == "raw"          # from rq1_raw.yaml
    assert cfg["model"]["name"] == "resnet1d"
    assert cfg["train"]["epochs"] == 3                          # override
    assert cfg["train"]["freeze_epochs"] == 0                   # inherited from rq1_logmel.yaml
    assert cfg["model"]["base_channels"] == 8


def test_invalid_combinations_are_rejected():
    with pytest.raises(ValueError):
        load_config("configs/base.yaml", ["features.representation=raw"])   # raw needs a 1-D model
    with pytest.raises(ValueError):
        load_config("configs/base.yaml", ["model.pooling=max"])
    with pytest.raises(ValueError):
        load_config("configs/base.yaml", ["eval.aggregation=average"])
