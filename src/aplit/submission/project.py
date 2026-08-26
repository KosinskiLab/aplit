"""Generate immutable, self-contained AlphaPulldownSnakemake run projects."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from .command import build_snakemake_command
from .errors import ProjectCollisionError, ValidationError
from .inputs import StagedUpload, rewrite_upload_aliases, stage_uploads
from .models import InputMode, RunSpec, SiteConfig
from .registry import atomic_write_json, initialize_status, manifest_path, read_manifest
from .validator import parse_allowlisted_overrides, validate_run_spec


@dataclass(frozen=True)
class PreparedProject:
    path: Path
    manifest: dict[str, Any]
    resumed: bool
    warning: str | None = None


def _git_revision(path: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() or None


def _copy_workflow(site: SiteConfig, project_dir: Path) -> None:
    workflow_source = site.workflow_template / "workflow"
    if not (workflow_source / "Snakefile").is_file():
        raise ValidationError(
            f"Workflow template has no workflow/Snakefile: {site.workflow_template}"
        )
    shutil.copytree(workflow_source, project_dir / "workflow")

    profile_source = site.profile_template
    if not profile_source.is_absolute():
        profile_source = site.workflow_template / profile_source
    if not (profile_source / "config.yaml").is_file():
        raise ValidationError(f"Snakemake profile has no config.yaml: {profile_source}")
    shutil.copytree(profile_source, project_dir / "config" / "profile")


def _load_base_config(site: SiteConfig) -> dict[str, Any]:
    path = site.workflow_template / "config" / "config.yaml"
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except OSError as exc:
        raise ValidationError(
            f"Could not read workflow config template {path}: {exc}"
        ) from exc
    if not isinstance(value, dict):
        raise ValidationError(f"Workflow config template is not a mapping: {path}")
    return value


def _write_lines(path: Path, rows: tuple[str, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"{row}\n" for row in rows), encoding="utf-8")


def _replace_aliases(
    rows: tuple[str, ...], uploads: tuple[StagedUpload, ...]
) -> tuple[str, ...]:
    return tuple(rewrite_upload_aliases(row, uploads) for row in rows)


def _input_files(
    spec: RunSpec, uploads: tuple[StagedUpload, ...], project_dir: Path
) -> tuple[list[str], dict[str, Any]]:
    config_dir = project_dir / "config"
    if spec.input_mode == InputMode.DIRECT:
        rows = _replace_aliases(spec.direct_complexes, uploads)
        sample_sheet = config_dir / "sample_sheet.csv"
        _write_lines(sample_sheet, rows)
        return [str(sample_sheet)], {"direct_complexes": list(rows)}

    baits = _replace_aliases(spec.baits, uploads)
    preys = _replace_aliases(spec.preys, uploads)
    bait_file = config_dir / "baits.txt"
    prey_file = config_dir / "preys.txt"
    _write_lines(bait_file, baits)
    _write_lines(prey_file, preys)
    return [str(bait_file), str(prey_file)], {
        "baits": list(baits),
        "preys": list(preys),
        "combination_count": len(baits) * len(preys),
    }


def _resolved_workflow_config(
    spec: RunSpec,
    site: SiteConfig,
    project_dir: Path,
    input_files: list[str],
) -> dict[str, Any]:
    backend = site.backend(spec.backend)
    databases_directory = site.backend_databases_directory(spec.backend)
    feature_directories = site.backend_feature_directories(spec.backend)
    config = _load_base_config(site)
    config.update(site.workflow_config)
    config.update(parse_allowlisted_overrides(spec.overrides))
    config.update(
        {
            "input_files": input_files,
            "output_directory": str(project_dir / "output"),
            "databases_directory": str(databases_directory),
            "backend_weights_directory": str(backend.weights_directory),
            "feature_directory": [str(path) for path in feature_directories],
            "prediction_container": str(backend.prediction_container),
        }
    )
    config.setdefault("create_feature_arguments", {})["--data_pipeline"] = (
        backend.data_pipeline
    )
    config.setdefault("structure_inference_arguments", {})["--fold_backend"] = (
        backend.fold_backend
    )
    return config


def _workflow_fingerprint(site: SiteConfig) -> str:
    revision = _git_revision(site.workflow_template)
    if revision:
        return revision
    snakefile = site.workflow_template / "workflow" / "Snakefile"
    return hashlib.sha256(snakefile.read_bytes()).hexdigest()


def _scientific_fingerprint(spec: RunSpec, site: SiteConfig) -> str:
    # The copied workflow and site snapshot are frozen in the project. A later
    # site/workflow upgrade must not prevent resuming that snapshot, so collision
    # detection compares only the user's scientific request.
    return spec.fingerprint()


def _replace_temporary_prefix(config_dir: Path, temporary: Path, target: Path) -> None:
    for path in config_dir.iterdir():
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8")
        rendered = text.replace(str(temporary), str(target))
        if rendered != text:
            path.write_text(rendered, encoding="utf-8")


def prepare_project(spec: RunSpec, site: SiteConfig) -> PreparedProject:
    validation = validate_run_spec(spec, site)
    if not validation.valid:
        raise ValidationError("\n".join(validation.errors))

    target = site.projects_root / spec.project_name
    fingerprint = _scientific_fingerprint(spec, site)
    if target.exists():
        if not manifest_path(target).is_file():
            raise ProjectCollisionError(
                f"Project path already exists but is not an APLit project: {target}"
            )
        manifest = read_manifest(target)
        if manifest.get("scientific_fingerprint") != fingerprint:
            raise ProjectCollisionError(
                "A project with this name already exists, but its inputs, backend, "
                "or overrides differ. Choose a new project name."
            )
        return PreparedProject(
            target,
            manifest,
            resumed=True,
            warning="Existing project found. Its frozen configuration will be resumed.",
        )

    site.projects_root.mkdir(parents=True, exist_ok=True)
    os.chmod(site.projects_root, site.project_mode)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{spec.project_name}.", dir=site.projects_root)
    )
    os.chmod(temporary, site.project_mode)
    try:
        _copy_workflow(site, temporary)
        staged_uploads = stage_uploads(spec.uploads, temporary / "input" / "uploads")
        input_files, resolved_inputs = _input_files(spec, staged_uploads, temporary)
        config = _resolved_workflow_config(spec, site, temporary, input_files)
        config_path = temporary / "config" / "config.yaml"
        config_path.write_text(
            yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
        )

        run_id = str(uuid.uuid4())
        workflow_revision = _workflow_fingerprint(site)
        slurm_prefix = f"{site.slurm_jobname_prefix}_{run_id.replace('-', '')[:12]}"
        manifest = {
            "schema_version": 1,
            "run_id": run_id,
            "project_name": spec.project_name,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "owner_uid": os.getuid(),
            "input_mode": spec.input_mode.value,
            "backend": spec.backend,
            "scientific_fingerprint": fingerprint,
            "spec": spec.fingerprint_dict(),
            "resolved_inputs": resolved_inputs,
            "uploads": [
                {
                    "original_name": item.original_name,
                    "alias": item.alias,
                    "path": str(item.path),
                    "sha256": item.sha256,
                    "record_ids": [record.identifier for record in item.records],
                }
                for item in staged_uploads
            ],
            "workflow_revision": workflow_revision,
            "site": site.snapshot(),
            "project_dir": str(target),
            "predictions_dir": str(target / "output" / "predictions"),
            "slurm_jobname_prefix": slurm_prefix,
            "command": build_snakemake_command(site, target, slurm_prefix),
        }
        # Paths were generated under the temporary directory. Replace that prefix
        # before publishing the immutable project directory.
        serialized = json.dumps(manifest).replace(str(temporary), str(target))
        manifest = json.loads(serialized)
        _replace_temporary_prefix(temporary / "config", temporary, target)
        atomic_write_json(manifest_path(temporary), manifest)
        initialize_status(temporary)
        os.replace(temporary, target)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise

    return PreparedProject(target, manifest, resumed=False)
