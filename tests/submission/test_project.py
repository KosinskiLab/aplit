from dataclasses import replace
from pathlib import Path

import pytest
import yaml

from aplit.submission.errors import ProjectCollisionError
from aplit.submission.models import InputMode, RunSpec, SiteConfig, UploadedFasta
from aplit.submission.project import prepare_project
from aplit.submission.registry import read_manifest, read_status


def test_prepare_direct_project_and_resume(fake_site: SiteConfig) -> None:
    spec = RunSpec(
        project_name="direct-run",
        input_mode=InputMode.DIRECT,
        backend="af3",
        direct_complexes=("uploaded+P01258",),
        uploads=(UploadedFasta("uploaded.fasta", b">uploaded\nAAAA\n"),),
        overrides={"batch_size": 2},
    )

    prepared = prepare_project(spec, fake_site)
    resumed = prepare_project(spec, fake_site)

    assert not prepared.resumed
    assert resumed.resumed
    assert prepared.path == resumed.path
    manifest = read_manifest(prepared.path)
    assert manifest["backend"] == "af3"
    assert manifest["project_dir"] == str(prepared.path)
    assert read_status(prepared.path)["state"] == "draft"
    sample = (prepared.path / "config" / "sample_sheet.csv").read_text()
    assert str(prepared.path / "input" / "uploads" / "uploaded.fasta") in sample
    config = yaml.safe_load((prepared.path / "config" / "config.yaml").read_text())
    assert config["batch_size"] == 2
    assert config["output_directory"] == str(prepared.path / "output")
    assert config["create_feature_arguments"]["--data_pipeline"] == "alphafold3"


def test_collision_with_changed_input_is_rejected(fake_site: SiteConfig) -> None:
    first = RunSpec(
        project_name="same-name",
        input_mode=InputMode.DIRECT,
        backend="af3",
        direct_complexes=("P1+P2",),
    )
    second = RunSpec(
        project_name="same-name",
        input_mode=InputMode.DIRECT,
        backend="af3",
        direct_complexes=("P1+P3",),
    )
    prepare_project(first, fake_site)

    with pytest.raises(ProjectCollisionError, match="Choose a new project name"):
        prepare_project(second, fake_site)


def test_bait_prey_generates_two_input_files(fake_site: SiteConfig) -> None:
    prepared = prepare_project(
        RunSpec(
            project_name="screen",
            input_mode=InputMode.BAIT_PREY,
            backend="af3",
            baits=("B1", "B2"),
            preys=("P1", "P2", "P3"),
        ),
        fake_site,
    )

    manifest = read_manifest(prepared.path)
    assert manifest["resolved_inputs"]["combination_count"] == 6
    config = yaml.safe_load((prepared.path / "config" / "config.yaml").read_text())
    assert [Path(path).name for path in config["input_files"]] == [
        "baits.txt",
        "preys.txt",
    ]


def test_trusted_site_workflow_config_is_applied(fake_site: SiteConfig) -> None:
    fake_site = replace(
        fake_site,
        workflow_config={"slurm_partition": "gpu-a,gpu-b", "slurm_qos": "normal"},
    )
    prepared = prepare_project(
        RunSpec(
            project_name="site-overlay",
            input_mode=InputMode.DIRECT,
            backend="af3",
            direct_complexes=("P1",),
        ),
        fake_site,
    )

    config = yaml.safe_load((prepared.path / "config" / "config.yaml").read_text())
    assert config["slurm_partition"] == "gpu-a,gpu-b"
    assert config["slurm_qos"] == "normal"


def test_project_config_uses_backend_specific_databases_and_features(
    fake_site: SiteConfig, tmp_path: Path
) -> None:
    backend_databases = tmp_path / "af3-databases"
    backend_features = tmp_path / "af3-features"
    backend_databases.mkdir()
    backend_features.mkdir()
    backend = replace(
        fake_site.backends["af3"],
        databases_directory=backend_databases,
        feature_directories=(backend_features,),
    )
    site = replace(fake_site, backends={"af3": backend})
    spec = RunSpec(
        project_name="backend-paths",
        input_mode=InputMode.DIRECT,
        backend="af3",
        direct_complexes=("P01258",),
    )

    project = prepare_project(spec, site)
    config = yaml.safe_load((project.path / "config" / "config.yaml").read_text())

    assert config["databases_directory"] == str(backend_databases)
    assert config["feature_directory"] == [str(backend_features)]
