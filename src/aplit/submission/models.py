"""Typed data models for submission, site configuration, and run state."""

from __future__ import annotations

import getpass
import hashlib
import json
import os
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import yaml

from .errors import ValidationError


class InputMode(str, Enum):
    DIRECT = "direct"
    BAIT_PREY = "bait_prey"


class RunState(str, Enum):
    DRAFT = "draft"
    VALIDATING = "validating"
    READY = "ready"
    SUBMITTING = "submitting"
    QUEUED = "queued"
    RUNNING = "running"
    CANCELLING = "cancelling"
    CANCELLED = "cancelled"
    COMPLETED = "completed"
    FAILED = "failed"
    UNKNOWN = "unknown"


TERMINAL_STATES = {
    RunState.CANCELLED,
    RunState.COMPLETED,
    RunState.FAILED,
}


@dataclass(frozen=True)
class UploadedFasta:
    """A browser upload retained in memory until the project is prepared."""

    name: str
    content: bytes = field(repr=False)

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.content).hexdigest()

    def fingerprint_dict(self) -> dict[str, Any]:
        return {"name": self.name, "sha256": self.sha256, "size": len(self.content)}


@dataclass(frozen=True)
class BackendPreset:
    key: str
    data_pipeline: str
    fold_backend: str
    prediction_container: Path
    weights_directory: Path

    @classmethod
    def from_mapping(cls, key: str, value: Mapping[str, Any]) -> BackendPreset:
        required = {
            "data_pipeline",
            "fold_backend",
            "prediction_container",
            "weights_directory",
        }
        missing = sorted(required - set(value))
        if missing:
            raise ValidationError(
                f"Backend preset '{key}' is missing: {', '.join(missing)}"
            )
        return cls(
            key=key,
            data_pipeline=str(value["data_pipeline"]),
            fold_backend=str(value["fold_backend"]),
            prediction_container=Path(str(value["prediction_container"])).expanduser(),
            weights_directory=Path(str(value["weights_directory"])).expanduser(),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "data_pipeline": self.data_pipeline,
            "fold_backend": self.fold_backend,
            "prediction_container": str(self.prediction_container),
            "weights_directory": str(self.weights_directory),
        }


@dataclass(frozen=True)
class SiteConfig:
    schema_version: int
    name: str
    projects_root: Path
    workflow_template: Path
    snakemake_executable: Path
    profile_template: Path
    allowed_input_roots: tuple[Path, ...]
    databases_directory: Path
    feature_directories: tuple[Path, ...]
    backends: Mapping[str, BackendPreset]
    executor: str = "slurm"
    max_parallel_jobs: int = 200
    retries: int = 3
    slurm_jobname_prefix: str = "aplit"
    squeue_executable: Path = Path("/usr/bin/squeue")
    scancel_executable: Path = Path("/usr/bin/scancel")
    sacct_executable: Path = Path("/usr/bin/sacct")
    project_mode: int = 0o700
    access_token_required: bool = True
    workflow_config: Mapping[str, Any] = field(default_factory=dict)

    @staticmethod
    def _expand(value: Any, username: str) -> Path:
        rendered = str(value).replace("{username}", username)
        return Path(os.path.expandvars(rendered)).expanduser()

    @classmethod
    def from_mapping(
        cls, value: Mapping[str, Any], *, username: str | None = None
    ) -> SiteConfig:
        username = username or getpass.getuser()
        schema_version = int(value.get("schema_version", 1))
        if schema_version != 1:
            raise ValidationError(
                f"Unsupported site schema_version {schema_version}; expected 1"
            )

        deployment = value.get("deployment") or {}
        execution = value.get("execution") or {}
        scheduler = value.get("scheduler") or {}
        backend_values = value.get("backends") or {}
        if not backend_values:
            raise ValidationError("Site configuration defines no backend presets")

        try:
            projects_root = deployment["projects_root"]
            workflow_template = deployment["workflow_template"]
            snakemake_executable = deployment["snakemake_executable"]
            profile_template = execution["profile_template"]
            databases_directory = value["databases_directory"]
        except KeyError as exc:
            raise ValidationError(
                f"Site configuration is missing required key: {exc.args[0]}"
            ) from exc

        mode_raw = deployment.get("project_mode", "0700")
        project_mode = int(mode_raw, 8) if isinstance(mode_raw, str) else int(mode_raw)
        if project_mode & 0o002:
            raise ValidationError("deployment.project_mode must not be world-writable")

        backends = {
            str(key): BackendPreset.from_mapping(str(key), item)
            for key, item in backend_values.items()
        }
        return cls(
            schema_version=schema_version,
            name=str(value.get("name") or "Unnamed site"),
            projects_root=cls._expand(projects_root, username),
            workflow_template=cls._expand(workflow_template, username),
            snakemake_executable=cls._expand(snakemake_executable, username),
            profile_template=cls._expand(profile_template, username),
            allowed_input_roots=tuple(
                cls._expand(path, username)
                for path in value.get("allowed_input_roots", ())
            ),
            databases_directory=cls._expand(databases_directory, username),
            feature_directories=tuple(
                cls._expand(path, username)
                for path in value.get("feature_directories", ())
            ),
            backends=backends,
            executor=str(execution.get("executor", "slurm")),
            max_parallel_jobs=int(execution.get("max_parallel_jobs", 200)),
            retries=int(execution.get("retries", 3)),
            slurm_jobname_prefix=str(execution.get("slurm_jobname_prefix", "aplit")),
            squeue_executable=cls._expand(
                scheduler.get("squeue_executable", "/usr/bin/squeue"), username
            ),
            scancel_executable=cls._expand(
                scheduler.get("scancel_executable", "/usr/bin/scancel"), username
            ),
            sacct_executable=cls._expand(
                scheduler.get("sacct_executable", "/usr/bin/sacct"), username
            ),
            project_mode=project_mode,
            access_token_required=bool(deployment.get("access_token_required", True)),
            workflow_config=dict(value.get("workflow_config") or {}),
        )

    @classmethod
    def load(cls, path: Path, *, username: str | None = None) -> SiteConfig:
        try:
            value = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        except OSError as exc:
            raise ValidationError(
                f"Could not read site configuration {path}: {exc}"
            ) from exc
        except yaml.YAMLError as exc:
            raise ValidationError(
                f"Invalid YAML in site configuration {path}: {exc}"
            ) from exc
        if not isinstance(value, Mapping):
            raise ValidationError("Site configuration root must be a mapping")
        return cls.from_mapping(value, username=username)

    def backend(self, key: str) -> BackendPreset:
        try:
            return self.backends[key]
        except KeyError as exc:
            raise ValidationError(
                f"Unknown backend '{key}'; choose one of {', '.join(sorted(self.backends))}"
            ) from exc

    def snapshot(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "name": self.name,
            "projects_root": str(self.projects_root),
            "workflow_template": str(self.workflow_template),
            "snakemake_executable": str(self.snakemake_executable),
            "profile_template": str(self.profile_template),
            "allowed_input_roots": [str(path) for path in self.allowed_input_roots],
            "databases_directory": str(self.databases_directory),
            "feature_directories": [str(path) for path in self.feature_directories],
            "backends": {
                key: preset.to_dict() for key, preset in self.backends.items()
            },
            "executor": self.executor,
            "max_parallel_jobs": self.max_parallel_jobs,
            "retries": self.retries,
            "slurm_jobname_prefix": self.slurm_jobname_prefix,
            "squeue_executable": str(self.squeue_executable),
            "scancel_executable": str(self.scancel_executable),
            "sacct_executable": str(self.sacct_executable),
            "workflow_config": dict(self.workflow_config),
        }


@dataclass(frozen=True)
class RunSpec:
    project_name: str
    input_mode: InputMode
    backend: str
    direct_complexes: tuple[str, ...] = ()
    baits: tuple[str, ...] = ()
    preys: tuple[str, ...] = ()
    uploads: tuple[UploadedFasta, ...] = field(default=(), repr=False)
    overrides: Mapping[str, Any] = field(default_factory=dict)

    def fingerprint_dict(self) -> dict[str, Any]:
        return {
            "project_name": self.project_name,
            "input_mode": self.input_mode.value,
            "backend": self.backend,
            "direct_complexes": list(self.direct_complexes),
            "baits": list(self.baits),
            "preys": list(self.preys),
            "uploads": [upload.fingerprint_dict() for upload in self.uploads],
            "overrides": dict(sorted(self.overrides.items())),
        }

    def fingerprint(self) -> str:
        encoded = json.dumps(
            self.fingerprint_dict(), sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class ValidationResult:
    errors: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    @property
    def valid(self) -> bool:
        return not self.errors


def dataclass_json(value: Any) -> dict[str, Any]:
    """Convert simple dataclasses to JSON-friendly dictionaries."""

    return asdict(value)
