from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from aplit.submission.models import SiteConfig


@pytest.fixture
def fake_site(tmp_path: Path) -> SiteConfig:
    workflow = tmp_path / "workflow-template"
    (workflow / "workflow" / "rules").mkdir(parents=True)
    (workflow / "workflow" / "Snakefile").write_text(
        'configfile: "config/config.yaml"\nrule all:\n    input: []\n',
        encoding="utf-8",
    )
    (workflow / "workflow" / "rules" / "common.smk").write_text("", encoding="utf-8")
    (workflow / "config" / "profiles" / "test").mkdir(parents=True)
    (workflow / "config" / "profiles" / "test" / "config.yaml").write_text(
        "executor: local\ncores: 1\n", encoding="utf-8"
    )
    base_config = {
        "input_files": ["config/sample_sheet.csv"],
        "output_directory": "output",
        "databases_directory": "/replace/db",
        "backend_weights_directory": "/replace/weights",
        "feature_directory": [],
        "create_feature_arguments": {"--data_pipeline": "alphafold2"},
        "structure_inference_arguments": {"--fold_backend": "alphafold2"},
        "prediction_container": "/replace/image.sif",
        "enable_structure_analysis": True,
    }
    (workflow / "config" / "config.yaml").write_text(
        yaml.safe_dump(base_config), encoding="utf-8"
    )

    executable = tmp_path / "bin" / "snakemake"
    executable.parent.mkdir()
    executable.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    executable.chmod(0o755)
    databases = tmp_path / "databases"
    weights = tmp_path / "weights"
    features = tmp_path / "features"
    for path in (databases, weights, features):
        path.mkdir()
    image = tmp_path / "af3.sif"
    image.write_bytes(b"fake-sif")

    return SiteConfig.from_mapping(
        {
            "schema_version": 1,
            "name": "Test",
            "deployment": {
                "projects_root": str(tmp_path / "projects"),
                "workflow_template": str(workflow),
                "snakemake_executable": str(executable),
                "project_mode": "0700",
                "access_token_required": False,
            },
            "allowed_input_roots": [str(tmp_path)],
            "databases_directory": str(databases),
            "feature_directories": [str(features)],
            "backends": {
                "af3": {
                    "data_pipeline": "alphafold3",
                    "fold_backend": "alphafold3",
                    "prediction_container": str(image),
                    "weights_directory": str(weights),
                }
            },
            "execution": {
                "executor": "local",
                "profile_template": "config/profiles/test",
                "max_parallel_jobs": 4,
                "retries": 1,
                "slurm_jobname_prefix": "test",
            },
        },
        username="tester",
    )


@pytest.fixture
def fake_site_file(fake_site: SiteConfig, tmp_path: Path) -> Path:
    site_path = tmp_path / "site.yaml"
    backend = fake_site.backends["af3"]
    value = {
        "schema_version": 1,
        "name": fake_site.name,
        "deployment": {
            "projects_root": str(fake_site.projects_root),
            "workflow_template": str(fake_site.workflow_template),
            "snakemake_executable": str(fake_site.snakemake_executable),
            "project_mode": "0700",
            "access_token_required": False,
        },
        "allowed_input_roots": [str(path) for path in fake_site.allowed_input_roots],
        "databases_directory": str(fake_site.databases_directory),
        "feature_directories": [str(path) for path in fake_site.feature_directories],
        "backends": {
            "af3": {
                "data_pipeline": backend.data_pipeline,
                "fold_backend": backend.fold_backend,
                "prediction_container": str(backend.prediction_container),
                "weights_directory": str(backend.weights_directory),
            }
        },
        "execution": {
            "executor": fake_site.executor,
            "profile_template": str(fake_site.profile_template),
            "max_parallel_jobs": fake_site.max_parallel_jobs,
            "retries": fake_site.retries,
            "slurm_jobname_prefix": fake_site.slurm_jobname_prefix,
        },
    }
    site_path.write_text(yaml.safe_dump(value), encoding="utf-8")
    return site_path
