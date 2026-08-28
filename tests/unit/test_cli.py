from pathlib import Path

import pytest

from memback.cli import build_parser, main, resolve_device


def test_build_parser_defaults():
    args = build_parser().parse_args(["membrane.gro"])

    assert args.input == Path("membrane.gro")
    assert args.output is None
    assert args.extension is None
    assert args.model is None
    assert args.device == "auto"


def test_build_parser_accepts_all_options():
    args = build_parser().parse_args([
        "membrane.gro", "-o", "out_dir", "-e", "ext_dir", "-m", "ckpt.pt", "--device", "cpu",
    ])

    assert args.output == Path("out_dir")
    assert args.extension == Path("ext_dir")
    assert args.model == Path("ckpt.pt")
    assert args.device == "cpu"


def test_build_parser_version(capsys):
    with pytest.raises(SystemExit) as exc_info:
        build_parser().parse_args(["--version"])
    assert exc_info.value.code == 0
    assert "memback" in capsys.readouterr().out


def test_resolve_device_cpu():
    import torch

    assert resolve_device("cpu") == torch.device("cpu")


def test_resolve_device_cuda_raises_when_unavailable():
    import torch

    if torch.cuda.is_available():
        pytest.skip("CUDA is available on this machine; nothing to assert here")
    with pytest.raises(SystemExit, match="cuda"):
        resolve_device("cuda")


def test_resolve_device_auto_matches_cuda_availability():
    import torch

    expected = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    assert resolve_device("auto") == expected


def test_main_missing_input_file_exits_with_message(tmp_path):
    missing = tmp_path / "does_not_exist.gro"

    with pytest.raises(SystemExit, match="not found"):
        main([str(missing)])


def test_main_extension_path_not_a_directory_exits(tmp_path):
    input_gro = tmp_path / "membrane.gro"
    input_gro.write_text("dummy\n")
    not_a_dir = tmp_path / "ext.txt"
    not_a_dir.write_text("dummy\n")

    with pytest.raises(SystemExit, match="not a directory"):
        main([str(input_gro), "-e", str(not_a_dir)])


def test_main_missing_model_checkpoint_exits(tmp_path):
    input_gro = tmp_path / "membrane.gro"
    input_gro.write_text("dummy\n")
    missing_model = tmp_path / "missing.pt"

    with pytest.raises(SystemExit, match="checkpoint not found"):
        main([str(input_gro), "-m", str(missing_model)])
