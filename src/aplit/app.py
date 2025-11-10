#!/usr/bin/env python3
"""
AlphaPulldown Structure Viewer - Web Application
"""

import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
from pathlib import Path
import time

from utils import (
    AlphaPulldownAnalyzer,
    plot_pae_heatmap,
    plot_model_comparison,
    create_3dmol_view,
    get_pae_file_for_model,
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
    with col4:
        sort_by = st.selectbox("Sort by", ["ipTM", "ipTM+pTM", "Job name"])

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

    # Apply sorting
    if sort_by == "ipTM":
        filtered_df = filtered_df.sort_values("iptm", ascending=False)
    elif sort_by == "ipTM+pTM":
        filtered_df = filtered_df.sort_values("iptm_ptm", ascending=False)
    else:
        filtered_df = filtered_df.sort_values("job")

    # Display table
    st.subheader(f"Predictions ({len(filtered_df)} results)")

    # Prepare display dataframe
    display_cols = ["job", "iptm", "iptm_ptm", "n_models"]
    if "mean_pae" in filtered_df.columns:
        display_cols.insert(3, "mean_pae")

    display_df = filtered_df[display_cols].copy()

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
    header_cols = ["Job", "ipTM", "ipTM+pTM"]
    if "mean_pae" in display_cols:
        header_cols.append("Mean PAE (Å)")
    header_cols.append("Models")

    # Header with borders
    cols = st.columns([3, 1, 1] + ([1] if "mean_pae" in display_cols else []) + [1])
    for i, col_name in enumerate(header_cols):
        with cols[i]:
            st.markdown(
                f'<div class="table-header">{col_name}</div>', unsafe_allow_html=True
            )

    # Display rows as clickable buttons
    container = st.container(height=400)
    with container:
        for idx, row in display_df.iterrows():
            job_name = row["job"]

            # Create columns for each row
            if "mean_pae" in display_cols:
                col1, col2, col3, col4, col5 = st.columns([3, 1, 1, 1, 1])
            else:
                col1, col2, col3, col4 = st.columns([3, 1, 1, 1])

            # Job name as button - clicking navigates to viewer
            with col1:
                if st.button(job_name, key=f"job_{idx}", width="stretch"):
                    navigate_to_viewer(job_name)

            # Metrics with color coding for ipTM
            with col2:
                iptm_val = row["iptm"]
                color = (
                    "green" if iptm_val > 0.7 else "orange" if iptm_val > 0.5 else "red"
                )
                st.markdown(f":{color}[**{iptm_val:.3f}**]")

            with col3:
                st.write(f"{row['iptm_ptm']:.3f}")

            if "mean_pae" in display_cols:
                with col4:
                    if pd.notna(row["mean_pae"]):
                        st.write(f"{row['mean_pae']:.1f}")
                    else:
                        st.write("N/A")
                with col5:
                    st.write(f"{row['n_models']}")
            else:
                with col4:
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

    selected_job = st.selectbox("Select prediction", job_list, index=default_idx)

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

    # Model comparison
    with st.expander("📊 Compare all models", expanded=False):
        fig = plot_model_comparison(models)
        if fig:
            st.pyplot(fig)

    st.divider()

    # Tabs for different views
    tab1, tab2, tab3 = st.tabs(["PAE Plot", "Structure (pLDDT)", "Structure (Chains)"])

    with tab1:
        pae_file = get_pae_file_for_model(job_path, selected_model["model_name"])
        if pae_file:
            # Make PAE plot responsive - use container width
            fig = plot_pae_heatmap(pae_file, figsize=(10, 10))
            if fig:
                st.pyplot(
                    fig,
                    # width="stretch",
                )
        else:
            st.warning("PAE file not found for this model")

    with tab2:
        pdb_file = selected_model["pdb_file"]
        if pdb_file.exists():
            html = create_3dmol_view(pdb_file, color_by="plddt", height=600)
            components.html(html, height=650)
        else:
            st.warning("PDB file not found")

    with tab3:
        pdb_file = selected_model["pdb_file"]
        if pdb_file.exists():
            html = create_3dmol_view(pdb_file, color_by="chain", height=600)
            components.html(html, height=650)
        else:
            st.warning("PDB file not found")

    # Download section
    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        if pdb_file.exists():
            with open(pdb_file, "rb") as f:
                st.download_button(
                    "Download PDB",
                    f,
                    file_name=f"{selected_job}_rank_{selected_model['rank']}.pdb",
                    mime="chemical/x-pdb",
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


def main():
    """Main application"""
    initialize_session_state()

    # Default directory (can be overridden by sidebar)
    default_dir = ""

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
