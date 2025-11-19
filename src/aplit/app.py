#!/usr/bin/env python3
"""
AlphaPulldown Structure Viewer - Web Application
"""

import argparse
import os
import time
from pathlib import Path

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from utils import (
    AlphaPulldownAnalyzer,
    plot_pae_heatmap,
    plot_model_comparison,
    create_3dmol_view,
    get_pae_file_for_model,
    load_interfaces_csv,
    get_pae_plot_image,
)

# Set page config
st.set_page_config(
    page_title="AlphaPulldown Viewer",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown(
    """
<style>
    .main-title {
        font-size: 2.8rem;
        font-weight: 700;
        color: #1f77b4;
        margin-bottom: 0.2rem;
        letter-spacing: -0.5px;
    }
    .subtitle {
        font-size: 1.1rem;
        color: #666;
        margin-bottom: 2rem;
        font-weight: 400;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 12px 24px;
        font-size: 1rem;
    }
</style>
""",
    unsafe_allow_html=True,
)


def initialize_session_state():
    """Initialize session state variables"""
    if "current_page" not in st.session_state:
        st.session_state.current_page = "overview"
    if "selected_job" not in st.session_state:
        st.session_state.selected_job = None
    if "results_df" not in st.session_state:
        st.session_state.results_df = None


def get_default_directory() -> str:
    """Derive default directory from CLI arguments or environment variables."""
    env_dir = os.environ.get("APLIT_DEFAULT_DIRECTORY", "")
    if env_dir:
        return env_dir

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--directory", type=str, default="")
    try:
        args, _ = parser.parse_known_args()
        return args.directory
    except SystemExit:
        return ""


def navigate_to_viewer(job_name: str):
    """Navigate to the viewer page with a specific job selected"""
    st.session_state.current_page = "viewer"
    st.session_state.selected_job = job_name
    st.rerun()


def render_sidebar(output_dir: str):
    """Render sidebar configuration"""
    st.sidebar.title("Configuration")

    # Directory input
    directory = st.sidebar.text_input(
        "Predictions Directory",
        value=output_dir,
        help="Path to directory containing AlphaPulldown predictions",
    )

    st.sidebar.divider()

    # Navigation
    st.sidebar.subheader("Navigation")
    if st.sidebar.button(
        "← Overview",
        width="stretch",
    ):
        st.session_state.current_page = "overview"
        st.rerun()

    if st.sidebar.button(
        "Structure Viewer →",
        width="stretch",
        disabled=st.session_state.selected_job is None,
    ):
        st.session_state.current_page = "viewer"
        st.rerun()

    st.sidebar.divider()

    # Auto-refresh
    st.sidebar.subheader("Auto-Refresh")
    auto_refresh = st.sidebar.checkbox("Enable", value=False)
    refresh_interval = st.sidebar.slider(
        "Interval (seconds)", 10, 300, 60, 10, disabled=not auto_refresh
    )

    if st.sidebar.button(
        "Refresh Now",
        width="stretch",
    ):
        st.cache_data.clear()
        st.rerun()

    return directory, auto_refresh, refresh_interval


def render_overview_page(results_df: pd.DataFrame, min_iptm: float, max_pae: float):
    """Render the main overview page with predictions table"""

    if results_df.empty:
        st.error("No valid multimer predictions found in the specified directory.")
        return

    # Apply filters
    filtered_df = results_df.copy()
    filtered_df = filtered_df[filtered_df["iptm"] >= min_iptm]
    if "mean_pae" in filtered_df.columns:
        filtered_df = filtered_df[
            (filtered_df["mean_pae"].isna()) | (filtered_df["mean_pae"] <= max_pae)
        ]

    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Predictions", len(filtered_df))
    with col2:
        st.metric("Best ipTM", f"{filtered_df['iptm'].max():.3f}")
    with col3:
        st.metric("Average ipTM", f"{filtered_df['iptm'].mean():.3f}")
    with col4:
        if "mean_pae" in filtered_df.columns and filtered_df["mean_pae"].notna().any():
            st.metric("Mean PAE (avg)", f"{filtered_df['mean_pae'].mean():.1f} Å")
        else:
            st.metric("Mean PAE (avg)", "N/A")

    st.divider()

    base_columns_excluded = {
        "job",
        "iptm",
        "iptm_ptm",
        "mean_pae",
        "best_model",
        "path",
        "n_models",
        "job_type",
        "ptm",
        "confidence_score",
        "interface_csv",
        "interface_summary_model",
    }
    alphajudge_numeric_cols = []
    for col in results_df.columns:
        if col in base_columns_excluded:
            continue
        if not pd.api.types.is_numeric_dtype(results_df[col]):
            continue
        if not results_df[col].notna().any():
            continue
        alphajudge_numeric_cols.append(col)
    alphajudge_available = len(alphajudge_numeric_cols) > 0

    # Filters and search in one row
    col1, col2, col3, col4 = st.columns([3, 1.5, 1.5, 1])
    with col1:
        search_term = st.text_input(
            "Search predictions", "", placeholder="Filter by job name..."
        )
    with col2:
        min_iptm = st.slider(
            "Minimum ipTM", 0.0, 1.0, min_iptm, 0.05, key="iptm_filter"
        )
    with col3:
        max_pae = st.slider(
            "Maximum mean PAE (Å)", 0.0, 30.0, max_pae, 0.5, key="pae_filter"
        )
    sort_options = ["ipTM", "ipTM+pTM", "Job name"]
    sort_options.extend(alphajudge_numeric_cols)
    with col4:
        sort_by = st.selectbox("Sort by", sort_options)

    # Re-apply filters with updated values
    filtered_df = results_df.copy()
    filtered_df = filtered_df[filtered_df["iptm"] >= min_iptm]
    if "mean_pae" in filtered_df.columns:
        filtered_df = filtered_df[
            (filtered_df["mean_pae"].isna()) | (filtered_df["mean_pae"] <= max_pae)
        ]

    # Apply search filter
    if search_term:
        filtered_df = filtered_df[
            filtered_df["job"].str.contains(search_term, case=False)
        ]

    # AlphaJudge-specific filters
    alphajudge_filter_ranges: Dict[str, Tuple[float, float]] = {}
    if alphajudge_available:
        with st.expander("AlphaJudge filters", expanded=False):
            filter_columns = st.columns(2)
            for idx, col in enumerate(alphajudge_numeric_cols):
                series = results_df[col].dropna()
                if series.empty:
                    continue
                min_val = float(series.min())
                max_val = float(series.max())
                target_col = filter_columns[idx % 2]
                with target_col:
                    if min_val == max_val:
                        st.caption(f"{col}: {min_val:.3f}")
                    else:
                        step = max((max_val - min_val) / 100, 0.0001)
                        alphajudge_filter_ranges[col] = st.slider(
                            col,
                            min_val,
                            max_val,
                            (min_val, max_val),
                            step=step,
                            key=f"alphajudge_filter_{col}",
                        )

    # Apply sorting
    if sort_by == "ipTM":
        filtered_df = filtered_df.sort_values("iptm", ascending=False)
    elif sort_by == "ipTM+pTM":
        filtered_df = filtered_df.sort_values("iptm_ptm", ascending=False)
    elif sort_by in alphajudge_numeric_cols:
        filtered_df = filtered_df.sort_values(
            sort_by, ascending=False, na_position="last"
        )
    else:
        filtered_df = filtered_df.sort_values("job")

    # Apply AlphaJudge filters
    for col, value_range in alphajudge_filter_ranges.items():
        filtered_df = filtered_df[
            (filtered_df[col].isna())
            | (
                (filtered_df[col] >= value_range[0])
                & (filtered_df[col] <= value_range[1])
            )
        ]

    # Display table
    st.subheader(f"Predictions ({len(filtered_df)} results)")

    # Prepare display dataframe
    column_specs = [
        {"key": "job", "label": "Job", "width": 3},
        {"key": "iptm", "label": "ipTM", "width": 1},
        {"key": "iptm_ptm", "label": "ipTM+pTM", "width": 1},
    ]
    if "mean_pae" in filtered_df.columns:
        column_specs.append({"key": "mean_pae", "label": "Mean PAE (Å)", "width": 1})
    for col in ["global_dockq", "best_interface_ipsae", "best_interface_lis"]:
        if col in filtered_df.columns:
            column_specs.append({"key": col, "label": col, "width": 1})
    column_specs.append({"key": "n_models", "label": "Models", "width": 1})

    display_keys = [spec["key"] for spec in column_specs]
    display_df = filtered_df[display_keys].copy()

    # CSS for table borders and styling
    st.markdown(
        """
    <style>
    .table-container {
        border: 1px solid #e0e0e0;
        border-radius: 5px;
        overflow: hidden;
    }
    .table-header {
        background-color: #f8f9fa;
        padding: 12px;
        font-weight: 600;
        border-bottom: 2px solid #dee2e6;
    }
    .table-row {
        border-bottom: 1px solid #e9ecef;
        padding: 8px 12px;
    }
    .table-row:last-child {
        border-bottom: none;
    }
    .table-row:hover {
        background-color: #f8f9fa;
    }
    div[data-testid="column"] {
        border-right: 1px solid #e9ecef;
        padding: 8px !important;
    }
    div[data-testid="column"]:last-child {
        border-right: none;
    }
    </style>
    """,
        unsafe_allow_html=True,
    )

    # Create clickable table header
    header_cols = [spec["label"] for spec in column_specs]
    col_widths = [spec["width"] for spec in column_specs]

    cols = st.columns(col_widths)
    for col, col_name in zip(cols, header_cols):
        with col:
            st.markdown(
                f'<div class="table-header">{col_name}</div>', unsafe_allow_html=True
            )

    # Display rows as clickable buttons
    container = st.container(height=400)
    with container:
        for idx, row in display_df.iterrows():
            job_name = row["job"]
            row_cols = st.columns(col_widths)

            for col, spec in zip(row_cols, column_specs):
                key = spec["key"]
                with col:
                    if key == "job":
                        if st.button(
                            job_name, key=f"job_{job_name}_{idx}", width="stretch"
                        ):
                            navigate_to_viewer(job_name)
                    elif key == "iptm":
                        iptm_val = row["iptm"]
                        color = (
                            "green"
                            if iptm_val > 0.7
                            else "orange"
                            if iptm_val > 0.5
                            else "red"
                        )
                        st.markdown(f":{color}[**{iptm_val:.3f}**]")
                    elif key == "iptm_ptm":
                        st.write(f"{row['iptm_ptm']:.3f}")
                    elif key == "mean_pae":
                        if pd.notna(row["mean_pae"]):
                            st.write(f"{row['mean_pae']:.1f}")
                        else:
                            st.write("N/A")
                    elif key == "global_dockq":
                        if pd.notna(row["global_dockq"]):
                            st.write(f"{row['global_dockq']:.3f}")
                        else:
                            st.write("N/A")
                    elif key == "best_interface_ipsae":
                        if pd.notna(row["best_interface_ipsae"]):
                            st.write(f"{row['best_interface_ipsae']:.3f}")
                        else:
                            st.write("N/A")
                    elif key == "best_interface_lis":
                        if pd.notna(row["best_interface_lis"]):
                            st.write(f"{row['best_interface_lis']:.3f}")
                        else:
                            st.write("N/A")
                    elif key == "n_models":
                        st.write(f"{row['n_models']}")

            # Add horizontal line between rows
            if idx < display_df.index[-1]:
                st.markdown(
                    '<hr style="margin: 0; border-color: #e9ecef;">',
                    unsafe_allow_html=True,
                )

    # Download button at the bottom
    st.divider()
    col1, col2, col3 = st.columns([2, 1, 1])
    with col3:
        display_df_export = display_df.copy()
        display_df_export.columns = header_cols
        csv = display_df_export.to_csv(index=False)
        st.download_button(
            "📥 Download CSV",
            csv,
            file_name="predictions.csv",
            mime="text/csv",
            width="stretch",
        )


def render_viewer_page(results_df: pd.DataFrame):
    """Render the detailed structure viewer page"""

    if results_df.empty:
        st.error("No predictions available")
        return

    # Job selection
    job_list = results_df["job"].tolist()

    # Set default selection
    default_idx = 0
    if st.session_state.selected_job and st.session_state.selected_job in job_list:
        default_idx = job_list.index(st.session_state.selected_job)

    st.markdown(
        """
        <style>
        .next-pred-btn button {
            background-color: #b5f5b0 !important;
            color: #0f3d0f !important;
            font-weight: 600 !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    select_col, next_col = st.columns([4, 1])
    with select_col:
        selected_job = st.selectbox("Select prediction", job_list, index=default_idx)
    with next_col:
        st.markdown('<div class="next-pred-btn">', unsafe_allow_html=True)
        if st.button("Next prediction →", key="next_prediction_button", width="stretch"):
            next_idx = (job_list.index(selected_job) + 1) % len(job_list)
            st.session_state.selected_job = job_list[next_idx]
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

    # Update session state
    st.session_state.selected_job = selected_job

    # Get job info
    job_row = results_df[results_df["job"] == selected_job].iloc[0]
    job_path = Path(job_row["path"])

    # Get all models for this job
    analyzer = AlphaPulldownAnalyzer(str(job_path.parent))
    models = analyzer.get_all_models(job_path)

    if not models:
        st.error(f"Could not load models for {selected_job}")
        return

    # Model selection
    model_options = [f"Rank {m['rank']} (Model {m['model_name']})" for m in models]
    selected_model_idx = st.selectbox(
        "Select model", range(len(models)), format_func=lambda i: model_options[i]
    )

    selected_model = models[selected_model_idx]
    pae_file = get_pae_file_for_model(job_path, selected_model["model_name"])
    pae_image = get_pae_plot_image(
        job_path, selected_model["model_name"], selected_model["rank"]
    )
    interfaces_df = load_interfaces_csv(job_path)
    model_interfaces = (
        interfaces_df[interfaces_df["model_used"] == selected_model["model_name"]]
        if interfaces_df is not None and "model_used" in interfaces_df.columns
        else None
    )

    st.divider()

    # Display metrics for selected model
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Rank", f"#{selected_model['rank']}")
    with col2:
        st.metric("ipTM", f"{selected_model['iptm']:.3f}")
    with col3:
        st.metric("ipTM+pTM", f"{selected_model['iptm_ptm']:.3f}")
    with col4:
        st.metric("Total Models", len(models))

    if model_interfaces is not None and not model_interfaces.empty:
        iface_numeric = model_interfaces.copy()
        for col in [
            "interface_pDockQ2",
            "interface_score",
            "average_interface_pae",
            "pDockQ/mpDockQ",
        ]:
            if col in iface_numeric.columns:
                iface_numeric[col] = pd.to_numeric(
                    iface_numeric[col], errors="coerce"
                )
        sort_columns = [
            col
            for col in [
                "interface_pDockQ2",
                "interface_score",
                "average_interface_pae",
            ]
            if col in iface_numeric.columns
        ]
        if sort_columns:
            ascending_flags = [
                True if col == "average_interface_pae" else False
                for col in sort_columns
            ]
            best_iface = iface_numeric.sort_values(
                by=sort_columns,
                ascending=ascending_flags,
                na_position="last",
            ).iloc[0]
        else:
            best_iface = iface_numeric.iloc[0]

        st.success("AlphaJudge interface scores available for this model.")
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            if "pDockQ/mpDockQ" in best_iface and pd.notna(best_iface["pDockQ/mpDockQ"]):
                st.metric("Global DockQ", f"{float(best_iface['pDockQ/mpDockQ']):.3f}")
        with col_b:
            if (
                "interface_pDockQ2" in best_iface
                and pd.notna(best_iface["interface_pDockQ2"])
            ):
                st.metric(
                    "Best interface pDockQ2",
                    f"{float(best_iface['interface_pDockQ2']):.3f}",
                )
        with col_c:
            if (
                "average_interface_pae" in best_iface
                and pd.notna(best_iface["average_interface_pae"])
            ):
                st.metric(
                    "Avg interface PAE (Å)",
                    f"{float(best_iface['average_interface_pae']):.2f}",
                )

    # Model comparison
    with st.expander("📊 Compare all models", expanded=False):
        fig = plot_model_comparison(models)
        if fig:
            st.pyplot(fig)

    st.divider()

    # Tabs for different views
    tab1, tab2, tab3 = st.tabs(["Structure (pLDDT)", "Structure (Chains)", "PAE Plot"])

    structure_file = selected_model.get("structure_file") or selected_model.get("pdb_file")
    structure_format = selected_model.get("structure_format", "pdb")

    with tab1:
        if structure_file and structure_file.exists():
            html = create_3dmol_view(
                structure_file,
                color_by="plddt",
                structure_format=structure_format,
                height=600,
            )
            components.html(html, height=650)
        else:
            st.warning("Structure file not found")

    with tab2:
        if structure_file and structure_file.exists():
            html = create_3dmol_view(
                structure_file,
                color_by="chain",
                structure_format=structure_format,
                height=600,
            )
            components.html(html, height=650)
        else:
            st.warning("Structure file not found")

    with tab3:
        if pae_image:
            st.image(str(pae_image), width="stretch")
        elif pae_file:
            fig = plot_pae_heatmap(pae_file, figsize=(10, 10))
            if fig:
                st.pyplot(fig)
        else:
            st.warning("PAE file not found for this model")

    # Download section
    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        if structure_file and structure_file.exists():
            with open(structure_file, "rb") as f:
                st.download_button(
                    "Download structure",
                    f,
                    file_name=f"{selected_job}_rank_{selected_model['rank']}{structure_file.suffix}",
                    mime="chemical/x-pdb" if structure_format == "pdb" else "chemical/x-mmcif",
                    width="stretch",
                )
    with col2:
        if pae_file:
            with open(pae_file, "rb") as f:
                st.download_button(
                    "Download PAE JSON",
                    f,
                    file_name=f"{selected_job}_rank_{selected_model['rank']}_pae.json",
                    mime="application/json",
                    width="stretch",
                )

    if (
        interfaces_df is not None
        and not interfaces_df.empty
        and model_interfaces is not None
    ):
        st.divider()
        with st.expander("Interface details (AlphaJudge)", expanded=False):
            download_df = model_interfaces.reset_index(drop=True)
            st.dataframe(download_df, width="stretch")
            csv_data = download_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download interfaces CSV (model)",
                csv_data,
                file_name=f"{selected_job}_{selected_model['model_name']}_interfaces.csv",
                mime="text/csv",
            )


def main():
    """Main application"""
    initialize_session_state()

    # Default directory (can be overridden by sidebar)
    default_dir = get_default_directory()

    # Render sidebar and get configuration
    directory, auto_refresh, refresh_interval = render_sidebar(default_dir)

    # Check if directory exists
    if not directory or not Path(directory).exists():
        st.warning("Please enter a valid predictions directory in the sidebar")
        st.info(
            """
        **Expected Directory Structure:**
        ```
        /path/to/predictions/
        ├── protein1_and_protein2/
        │   ├── ranking_debug.json
        │   ├── ranked_0.pdb
        │   ├── pae_model_1_ptm_pred_0.json
        │   └── ...
        ```
        """
        )
        return

    # Load predictions
    @st.cache_data(ttl=refresh_interval if auto_refresh else None)
    def load_predictions(directory_path):
        analyzer = AlphaPulldownAnalyzer(directory_path)
        return analyzer.analyze_directory()

    with st.spinner("Analyzing predictions..."):
        results_df = load_predictions(directory)

    st.session_state.results_df = results_df

    # Render appropriate page with default filter values
    if st.session_state.current_page == "overview":
        render_overview_page(
            results_df, 0.0, 30.0
        )  # Default min_iptm=0.0, max_pae=30.0
    elif st.session_state.current_page == "viewer":
        render_viewer_page(results_df)

    # Auto-refresh logic
    if auto_refresh and st.session_state.current_page == "overview":
        time.sleep(refresh_interval)
        st.rerun()


if __name__ == "__main__":
    main()
