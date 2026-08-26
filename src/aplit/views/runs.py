"""Streamlit run registry, status, log, resume, and cancellation page."""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from aplit.submission.cancellation import cancel_project
from aplit.submission.errors import SubmissionError
from aplit.submission.launcher import launch_project
from aplit.submission.models import RunState, SiteConfig
from aplit.submission.reconciler import reconcile_project
from aplit.submission.registry import (
    list_projects,
    read_manifest,
    read_status,
    tail_log,
)

ACTIVE_STATES = {
    RunState.SUBMITTING.value,
    RunState.QUEUED.value,
    RunState.RUNNING.value,
    RunState.CANCELLING.value,
}
RESUMABLE_STATES = {
    RunState.FAILED.value,
    RunState.CANCELLED.value,
    RunState.READY.value,
}


def render_runs_page(site: SiteConfig | None) -> None:
    st.title("Runs")
    if site is None:
        st.info("Run tracking is unavailable until a site configuration is selected.")
        return

    if st.button("Refresh statuses"):
        st.rerun()
    projects = list_projects(site.projects_root)
    if not projects:
        st.info("No APLit projects found. Validate a project on the Submit page first.")
        return

    rows = []
    statuses: dict[Path, dict] = {}
    for project in projects:
        try:
            status = reconcile_project(project)
        except SubmissionError:
            status = read_status(project)
        manifest = read_manifest(project)
        statuses[project] = status
        plan = status.get("plan") or {}
        rows.append(
            {
                "Project": manifest.get("project_name", project.name),
                "State": status.get("state"),
                "Backend": manifest.get("backend"),
                "Mode": manifest.get("input_mode"),
                "Folds": plan.get("requested_folds"),
                "DAG jobs": plan.get("dag_jobs"),
                "Updated": status.get("updated_at"),
            }
        )
    st.dataframe(rows, width="stretch", hide_index=True)

    selected_name = st.selectbox("Run details", [project.name for project in projects])
    project = next(item for item in projects if item.name == selected_name)
    manifest = read_manifest(project)
    status = statuses[project]
    state = str(status.get("state"))

    columns = st.columns(4)
    columns[0].metric("State", state)
    columns[1].metric("Backend", manifest.get("backend", "N/A"))
    columns[2].metric("SLURM jobs", len(status.get("scheduler_jobs") or ()))
    columns[3].metric(
        "Exit code",
        status.get("exit_code") if status.get("exit_code") is not None else "N/A",
    )
    if status.get("message"):
        st.warning(status["message"])

    action_columns = st.columns(3)
    if action_columns[0].button(
        "View results", disabled=not Path(manifest["predictions_dir"]).exists()
    ):
        st.session_state.predictions_directory = manifest["predictions_dir"]
        st.session_state.current_page = "overview"
        st.session_state.navigation_target = "Results"
        st.rerun()

    if action_columns[1].button("Resume", disabled=state not in RESUMABLE_STATES):
        try:
            launch_project(project)
        except SubmissionError as exc:
            st.error(str(exc))
        else:
            st.rerun()

    confirm_cancel = st.checkbox(
        "Confirm cancellation of all controller and SLURM jobs for this run",
        disabled=state not in ACTIVE_STATES,
    )
    if action_columns[2].button(
        "Cancel run", disabled=state not in ACTIVE_STATES or not confirm_cancel
    ):
        try:
            cancel_project(project)
        except SubmissionError as exc:
            st.error(str(exc))
        else:
            st.rerun()

    st.subheader("Log tail")
    log = tail_log(project)
    st.code(log or "No controller log yet.", language="text")
