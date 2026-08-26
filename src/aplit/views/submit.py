"""Streamlit submission page backed by the submission service layer."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import streamlit as st
import yaml

from aplit.submission.errors import SubmissionError
from aplit.submission.inputs import upload_aliases
from aplit.submission.launcher import launch_project
from aplit.submission.models import (
    InputMode,
    RunSpec,
    RunState,
    SiteConfig,
    UploadedFasta,
)
from aplit.submission.planner import plan_project
from aplit.submission.project import prepare_project
from aplit.submission.registry import read_status
from aplit.submission.validator import (
    normalize_nonempty_lines,
    parse_allowlisted_overrides,
    preflight_site,
    validate_run_spec,
)

COLLISION_BUSY_STATES = {
    RunState.VALIDATING.value,
    RunState.SUBMITTING.value,
    RunState.QUEUED.value,
    RunState.RUNNING.value,
    RunState.CANCELLING.value,
    RunState.UNKNOWN.value,
}


def _parse_overrides(text: str) -> dict[str, Any]:
    try:
        value = yaml.safe_load(text) if text.strip() else {}
    except yaml.YAMLError as exc:
        raise ValueError(f"Advanced overrides are not valid YAML: {exc}") from exc
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise TypeError("Advanced overrides must be a YAML mapping")
    return parse_allowlisted_overrides(value)


def _uploaded_fastas(files: list[Any] | None) -> tuple[UploadedFasta, ...]:
    return tuple(UploadedFasta(file.name, file.getvalue()) for file in files or ())


def _render_last_validation() -> None:
    validation = st.session_state.get("submission_validation")
    if not validation:
        return
    if validation.get("warning"):
        st.warning(validation["warning"])
    for warning in validation.get("warnings") or ():
        st.warning(warning)
    if validation.get("active"):
        st.info(
            "This project is already active or needs reconciliation. Open it from Runs."
        )
        return
    if validation.get("completed"):
        st.info("This existing project is already complete. Open it from Runs.")
        return
    if validation.get("valid"):
        st.success("Validation succeeded. The frozen project is ready to submit.")
        columns = st.columns(3)
        columns[0].metric("Requested folds", validation.get("requested_folds", 0))
        columns[1].metric("Skipped folds", validation.get("skipped_folds", 0))
        columns[2].metric(
            "DAG jobs",
            validation.get("dag_jobs")
            if validation.get("dag_jobs") is not None
            else "N/A",
        )
        st.code(" ".join(validation.get("command") or ()), language="bash")
        if st.button("Submit run", type="primary", width="stretch"):
            try:
                launch_project(Path(validation["project_dir"]))
            except SubmissionError as exc:
                st.error(str(exc))
            else:
                st.session_state.navigation_target = "Runs"
                st.rerun()


def render_submit_page(site: SiteConfig | None) -> None:
    st.title("Submit run")
    st.caption(
        "Create a reproducible AlphaPulldownSnakemake project, validate it, then submit it."
    )
    if site is None:
        st.info(
            "Submission is not configured. Start APLit with --site-config or set "
            "APLIT_SITE_CONFIG. The results browser remains available."
        )
        return

    preflight = preflight_site(site)
    for warning in preflight.warnings:
        st.warning(warning)
    if not preflight.valid:
        st.error("Site preflight failed:\n\n- " + "\n- ".join(preflight.errors))
        return

    st.caption(f"Site: {site.name} · Projects: {site.projects_root}")
    with st.form("submission_form"):
        project_name = st.text_input(
            "Project name",
            placeholder="my-screen-2026-08",
            help="An existing matching name is resumed; it is never overwritten.",
        )
        mode_label = st.radio(
            "Input mode",
            ["Direct complexes", "Bait–prey screen"],
            horizontal=True,
        )
        backend = st.selectbox("Backend", sorted(site.backends))
        uploads = st.file_uploader(
            "Upload FASTA files",
            type=["fa", "faa", "fasta", "fas"],
            accept_multiple_files=True,
            help="Use the sanitized filename stem as an alias in the input rows below.",
        )
        if uploads:
            try:
                aliases = upload_aliases(_uploaded_fastas(uploads))
            except SubmissionError as exc:
                st.error(str(exc))
            else:
                st.caption(f"Uploaded aliases: {', '.join(aliases)}")

        if mode_label == "Direct complexes":
            direct_text = st.text_area(
                "Complexes",
                placeholder="P01258+P01579\nP01258\nuploaded_protein+P01579",
                help="One AlphaPulldown fold specification per line.",
            )
            bait_text = ""
            prey_text = ""
        else:
            direct_text = ""
            left, right = st.columns(2)
            bait_text = left.text_area("Baits", placeholder="P01258\n/path/bait.fasta")
            prey_text = right.text_area("Preys", placeholder="P01579\nP61769")

        overrides_text = st.text_area(
            "Advanced overrides (allowlisted YAML)",
            placeholder="batch_size: 2\nenable_structure_analysis: true",
        )
        validate_clicked = st.form_submit_button(
            "Validate", type="primary", width="stretch"
        )

    if validate_clicked:
        try:
            mode = (
                InputMode.DIRECT
                if mode_label == "Direct complexes"
                else InputMode.BAIT_PREY
            )
            spec = RunSpec(
                project_name=project_name.strip(),
                input_mode=mode,
                backend=backend,
                direct_complexes=normalize_nonempty_lines(direct_text),
                baits=normalize_nonempty_lines(bait_text),
                preys=normalize_nonempty_lines(prey_text),
                uploads=_uploaded_fastas(uploads),
                overrides=_parse_overrides(overrides_text),
            )
            result = validate_run_spec(spec, site)
            if not result.valid:
                raise ValueError("\n".join(result.errors))
            prepared = prepare_project(spec, site)
            current = read_status(prepared.path)
            if prepared.resumed and current.get("state") == RunState.COMPLETED.value:
                st.session_state.submission_validation = {
                    "completed": True,
                    "project_dir": str(prepared.path),
                    "warning": prepared.warning,
                }
            elif prepared.resumed and current.get("state") in COLLISION_BUSY_STATES:
                st.session_state.submission_validation = {
                    "active": True,
                    "project_dir": str(prepared.path),
                    "warning": prepared.warning,
                }
            else:
                plan = plan_project(prepared.path)
                if not plan.valid:
                    raise ValueError(
                        "Snakemake dry-run failed. Inspect the validation output below.\n\n"
                        + plan.output[-4000:]
                    )
                st.session_state.submission_validation = {
                    "valid": True,
                    "project_dir": str(prepared.path),
                    "warning": prepared.warning,
                    "warnings": list(result.warnings),
                    "requested_folds": plan.requested_folds,
                    "skipped_folds": plan.skipped_folds,
                    "dag_jobs": plan.dag_jobs,
                    "command": list(plan.command),
                }
        except (SubmissionError, TypeError, ValueError) as exc:
            st.session_state.submission_validation = None
            st.error(str(exc))

    _render_last_validation()
