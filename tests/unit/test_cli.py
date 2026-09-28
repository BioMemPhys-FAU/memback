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


def test_build_parser_skip_missing():
    assert build_parser().parse_args(["membrane.gro"]).skip_missing is False
    assert build_parser().parse_args(["membrane.gro", "--skip-missing"]).skip_missing is True


def _gro_with_unmapped_residue(path):
    """A water bead and one bead of FOO, a lipid without a mapping."""
    atoms = [(1, "W", "W", 1.0), (2, "FOO", "NC3", 2.0)]
    lines = ["test", f"{len(atoms):5d}"]
    lines += [f"{resid:5d}{resname:<5s}{name:>5s}{i:5d}{x:8.3f}{x:8.3f}{x:8.3f}"
              for i, (resid, resname, name, x) in enumerate(atoms, 1)]
    lines.append("   5.00000   5.00000   5.00000")
    path.write_text("\n".join(lines) + "\n")
    return path


def test_backmapping_stops_on_missing_mapping_before_writing(tmp_path):
    from memback.pipeline import MissingMappingError, backmapping

    output = tmp_path / "out"
    with pytest.raises(MissingMappingError) as exc_info:
        backmapping(str(_gro_with_unmapped_residue(tmp_path / "in.gro")), filename=str(output))
    assert exc_info.value.resnames == ["FOO"]
    assert not output.exists()


def test_find_missing_mappings_ignores_water_and_ions(make_universe):
    from memback.pipeline import find_missing_mappings

    u = make_universe(["W", "NA", "NC3", "NC3"], ["W", "ION", "POPC", "FOO"], [0, 1, 2, 3],
                      [[0, 0, 0]] * 4)
    assert find_missing_mappings(u, {"POPC": {}}) == ["FOO"]


def test_main_missing_mapping_suggests_create_map(tmp_path):
    gro = _gro_with_unmapped_residue(tmp_path / "in.gro")

    with pytest.raises(SystemExit) as exc_info:
        main([str(gro), "-o", str(tmp_path / "out"), "--device", "cpu"])
    message = str(exc_info.value.code)
    assert "no mapping for residue(s)" in message and "FOO" in message
    assert "memback create_map" in message and "--m3name FOO" in message
    assert "--skip-missing" in message


def test_missing_mapping_message_uses_shipped_topologies():
    from memback.cli import missing_mapping_message

    message = missing_mapping_message(["DXPC", "DOPC"], Path("membrane.gro"), Path("ext"))
    # DXPC: Martini 3 itp is shipped, CHARMM36 itp is not
    assert "martini_v3.0.0_phospholipids_PC_v2.itp --m3name DXPC" in message
    assert "cp <CHARMM36 DXPC.itp> ext/DXPC.itp" in message
    # DOPC: both shipped, so no copy step
    assert "charmm_lipid_itps/DOPC.itp -o ext/DOPC" in message
    assert "cp <CHARMM36 DOPC.itp>" not in message
    assert "memback membrane.gro -e ext" in message
