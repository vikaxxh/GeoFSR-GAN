import os
import glob
import yaml
import pytest


def test_ablation_configs_exist_and_valid():
    ablation_dir = "configs/ablations"
    config_files = glob.glob(os.path.join(ablation_dir, "*.yaml"))

    assert len(config_files) >= 4, f"Expected at least 4 ablation config files, found {len(config_files)}"

    for cfg_file in config_files:
        with open(cfg_file, "r") as f:
            cfg = yaml.safe_load(f)

        assert "project" in cfg, f"Missing 'project' section in {cfg_file}"
        assert "dataset" in cfg, f"Missing 'dataset' section in {cfg_file}"
        assert "model" in cfg, f"Missing 'model' section in {cfg_file}"
        assert "losses" in cfg, f"Missing 'losses' section in {cfg_file}"
        assert "training" in cfg, f"Missing 'training' section in {cfg_file}"
