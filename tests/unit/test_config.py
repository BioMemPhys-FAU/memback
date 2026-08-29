import importlib

from memback import config


def test_bundled_data_files_are_all_present():
    """Regression guard: the packaged data/model directories must stay intact."""
    missing = config.check_data_files()
    assert missing == []


def test_memback_root_env_var_overrides_data_and_model_dirs(tmp_path, monkeypatch):
    monkeypatch.setenv("MEMBACK_ROOT", str(tmp_path))
    try:
        reloaded = importlib.reload(config)
        assert reloaded.DATA_DIR == tmp_path / "data"
        assert reloaded.MODEL_DIR == tmp_path / "model"
        assert reloaded.model_path == str(tmp_path / "model" / "memback_0.1.1_state_dict.pt")
        assert len(reloaded.check_data_files()) > 0
    finally:
        monkeypatch.delenv("MEMBACK_ROOT", raising=False)
        importlib.reload(config)