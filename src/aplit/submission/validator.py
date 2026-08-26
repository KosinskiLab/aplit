"""Pure validation for run specifications and site preflight checks."""

from __future__ import annotations

import os
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .errors import ValidationError
from .models import InputMode, RunSpec, SiteConfig, ValidationResult

PROJECT_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,79}$")
SLURM_PREFIX_RE = re.compile(r"^[A-Za-z0-9_-]{1,30}$")
FOLD_ROW_RE = re.compile(r"^[A-Za-z0-9_./:+-]+$")

ALLOWED_OVERRIDE_KEYS = {
    "batch_max_tokens",
    "batch_size",
    "enable_structure_analysis",
    "generate_recursive_report",
    "length_filter_fetch_uniprot",
    "max_protein_length",
    "max_total_length_alphafold2",
    "max_total_length_alphafold3",
    "only_generate_features",
}


def normalize_nonempty_lines(value: str) -> tuple[str, ...]:
    return tuple(
        line.strip()
        for line in value.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    )


def parse_allowlisted_overrides(value: Mapping[str, Any]) -> dict[str, Any]:
    unknown = sorted(set(value) - ALLOWED_OVERRIDE_KEYS)
    if unknown:
        raise ValueError(
            "Unsupported advanced override(s): "
            + ", ".join(unknown)
            + ". Allowed keys: "
            + ", ".join(sorted(ALLOWED_OVERRIDE_KEYS))
        )

    parsed = dict(value)
    for key in ("batch_size", "batch_max_tokens", "max_protein_length"):
        if key in parsed:
            if isinstance(parsed[key], bool) or not isinstance(parsed[key], int):
                raise ValueError(f"{key} must be an integer")
            minimum = 1 if key == "batch_size" else 0
            if parsed[key] < minimum:
                raise ValueError(f"{key} must be at least {minimum}")
    for key in ("max_total_length_alphafold2", "max_total_length_alphafold3"):
        if key in parsed:
            if isinstance(parsed[key], bool) or not isinstance(parsed[key], int):
                raise ValueError(f"{key} must be an integer")
            if parsed[key] < 0:
                raise ValueError(f"{key} must be zero or greater")
    for key in (
        "enable_structure_analysis",
        "generate_recursive_report",
        "length_filter_fetch_uniprot",
        "only_generate_features",
    ):
        if key in parsed and not isinstance(parsed[key], bool):
            raise ValueError(f"{key} must be true or false")
    return parsed


def validate_run_spec(spec: RunSpec, site: SiteConfig) -> ValidationResult:
    errors: list[str] = []
    warnings: list[str] = []

    if not PROJECT_NAME_RE.fullmatch(spec.project_name):
        errors.append(
            "Project name must be 1-80 characters and contain only letters, "
            "numbers, dot, underscore, or hyphen."
        )
    elif spec.project_name in {".", ".."}:
        errors.append("Project name cannot be '.' or '..'.")

    if spec.backend not in site.backends:
        errors.append(f"Unknown backend: {spec.backend}")

    if spec.input_mode == InputMode.DIRECT:
        if not spec.direct_complexes:
            errors.append(
                "Direct-complex mode requires at least one fold specification."
            )
        if spec.baits or spec.preys:
            errors.append("Direct-complex mode cannot include bait or prey lists.")
    elif spec.input_mode == InputMode.BAIT_PREY:
        if not spec.baits or not spec.preys:
            errors.append("Bait-prey mode requires at least one bait and one prey.")
        if spec.direct_complexes:
            errors.append("Bait-prey mode cannot include direct complexes.")
        combinations = len(spec.baits) * len(spec.preys)
        if combinations > 1000:
            warnings.append(
                f"This screen expands to {combinations:,} bait-prey combinations."
            )
    else:
        errors.append(f"Unsupported input mode: {spec.input_mode}")

    all_rows = spec.direct_complexes + spec.baits + spec.preys
    for row in all_rows:
        if not row.strip():
            errors.append("Input rows cannot be empty.")
        if any(char in row for char in ("\n", "\r", "\0")):
            errors.append("Input rows cannot contain control characters.")
        if any(char.isspace() for char in row):
            errors.append(
                f"Whitespace is not supported in fold specifications: {row!r}"
            )
        if not FOLD_ROW_RE.fullmatch(row):
            errors.append(
                f"Fold specification contains unsupported or unsafe characters: {row!r}"
            )
        for token in row.split("+"):
            base = token.split(":", 1)[0]
            if "/" not in base:
                continue
            path = Path(base)
            if not is_within(path, site.allowed_input_roots):
                errors.append(
                    f"Server input path is missing or outside allowed roots: {base}"
                )
            else:
                # Imported lazily because the FASTA parser also reuses is_within.
                from .inputs import validate_server_fasta

                try:
                    validate_server_fasta(path, site.allowed_input_roots)
                except ValidationError as exc:
                    errors.append(str(exc))

    try:
        parse_allowlisted_overrides(spec.overrides)
    except ValueError as exc:
        errors.append(str(exc))

    upload_names: set[str] = set()
    for upload in spec.uploads:
        key = upload.name.casefold()
        if key in upload_names:
            errors.append(f"Duplicate uploaded FASTA filename: {upload.name}")
        upload_names.add(key)
        if not upload.content:
            errors.append(f"Uploaded FASTA is empty: {upload.name}")

    return ValidationResult(tuple(errors), tuple(warnings))


def is_within(path: Path, roots: tuple[Path, ...]) -> bool:
    try:
        resolved = path.expanduser().resolve(strict=True)
    except OSError:
        return False
    for root in roots:
        try:
            resolved.relative_to(root.expanduser().resolve(strict=True))
            return True
        except (OSError, ValueError):
            continue
    return False


def preflight_site(site: SiteConfig) -> ValidationResult:
    errors: list[str] = []
    warnings: list[str] = []

    if site.executor not in {"local", "slurm"}:
        errors.append("execution.executor must be 'local' or 'slurm'.")
    if not SLURM_PREFIX_RE.fullmatch(site.slurm_jobname_prefix):
        errors.append(
            "execution.slurm_jobname_prefix must contain 1-30 letters, numbers, "
            "underscores, or hyphens."
        )
    if site.max_parallel_jobs < 1:
        errors.append("execution.max_parallel_jobs must be at least 1.")
    if site.retries < 0:
        errors.append("execution.retries cannot be negative.")

    profile_template = site.profile_template
    if not profile_template.is_absolute():
        profile_template = site.workflow_template / profile_template
    checks = {
        "workflow template": site.workflow_template,
        "Snakemake executable": site.snakemake_executable,
        "profile template": profile_template,
        "databases directory": site.databases_directory,
    }
    for label, path in checks.items():
        if not path.exists():
            errors.append(f"Configured {label} does not exist: {path}")
    if site.snakemake_executable.exists() and not os.access(
        site.snakemake_executable, os.X_OK
    ):
        errors.append(
            f"Configured Snakemake executable is not executable: {site.snakemake_executable}"
        )

    project_parent = site.projects_root
    while not project_parent.exists() and project_parent != project_parent.parent:
        project_parent = project_parent.parent
    if project_parent.exists() and not os.access(project_parent, os.W_OK | os.X_OK):
        errors.append(
            f"Projects root cannot be created or written by this user: {site.projects_root}"
        )

    for feature_directory in site.feature_directories:
        if not feature_directory.is_dir():
            errors.append(
                f"Configured feature directory does not exist: {feature_directory}"
            )
        elif not os.access(feature_directory, os.R_OK | os.X_OK):
            errors.append(
                f"Configured feature directory is not readable: {feature_directory}"
            )

    for backend in site.backends.values():
        databases_directory = site.backend_databases_directory(backend.key)
        if not databases_directory.is_dir():
            errors.append(
                f"Backend {backend.key} databases directory does not exist: "
                f"{databases_directory}"
            )
        elif not os.access(databases_directory, os.R_OK | os.X_OK):
            errors.append(
                f"Backend {backend.key} databases directory is not readable: "
                f"{databases_directory}"
            )
        for feature_directory in site.backend_feature_directories(backend.key):
            if not feature_directory.is_dir():
                errors.append(
                    f"Backend {backend.key} feature directory does not exist: "
                    f"{feature_directory}"
                )
            elif not os.access(feature_directory, os.R_OK | os.X_OK):
                errors.append(
                    f"Backend {backend.key} feature directory is not readable: "
                    f"{feature_directory}"
                )
        if not backend.prediction_container.exists():
            errors.append(
                f"Backend {backend.key} container does not exist: "
                f"{backend.prediction_container}"
            )
        elif not backend.prediction_container.is_file() or not os.access(
            backend.prediction_container, os.R_OK
        ):
            errors.append(
                f"Backend {backend.key} container is not a readable file: "
                f"{backend.prediction_container}"
            )
        if not backend.weights_directory.exists():
            errors.append(
                f"Backend {backend.key} weights do not exist: "
                f"{backend.weights_directory}"
            )
        elif not os.access(backend.weights_directory, os.R_OK | os.X_OK):
            errors.append(
                f"Backend {backend.key} weights are not readable: "
                f"{backend.weights_directory}"
            )

    for root in site.allowed_input_roots:
        if not root.exists():
            warnings.append(f"Allowed input root does not exist: {root}")

    if site.executor == "slurm":
        for label, path in (
            ("squeue", site.squeue_executable),
            ("scancel", site.scancel_executable),
        ):
            if not path.exists():
                errors.append(f"Configured {label} executable does not exist: {path}")

    return ValidationResult(tuple(errors), tuple(warnings))
