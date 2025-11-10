"""
Utility functions and data classes for AlphaPulldown analysis
"""

import json

from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

import streamlit as st
import matplotlib.pyplot as plt


class AlphaPulldownAnalyzer:
    """Handles analysis of AlphaPulldown prediction directories"""

    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)

    def obtain_seq_lengths(self, result_dir: Path) -> List[int]:
        """Extract sequence lengths from PDB file by counting chains"""
        try:
            # Get the best ranked PDB file
            pdb_file = result_dir / "ranked_0.pdb"
            if not pdb_file.exists():
                return []

            # Parse PDB to get chain lengths
            chain_lengths = {}
            with open(pdb_file, "r") as f:
                for line in f:
                    if line.startswith("ATOM"):
                        chain_id = line[21]
                        res_num = int(line[22:26].strip())
                        if chain_id not in chain_lengths:
                            chain_lengths[chain_id] = 0
                        chain_lengths[chain_id] = max(chain_lengths[chain_id], res_num)

            # Return list of chain lengths
            return list(chain_lengths.values()) if chain_lengths else []

        except Exception as e:
            st.warning(f"Could not extract sequence lengths from {result_dir}: {e}")
            return []

    def obtain_pae_and_iptm(
        self, result_dir: Path, model_name: str
    ) -> Tuple[Optional[np.ndarray], Optional[float]]:
        """Extract PAE matrix and ipTM score for a specific model"""
        try:
            # Load PAE data
            pae_file = result_dir / f"pae_{model_name}.json"
            if not pae_file.exists():
                model_num = (
                    model_name.split("_")[1] if "_" in model_name else model_name
                )
                pae_file = result_dir / f"pae_model_{model_num}_ptm_pred_0.json"

            if pae_file.exists():

                with open(pae_file, "r") as f:
                    pae_data = json.load(f)
                pae_mtx = np.array(pae_data[0]["predicted_aligned_error"])
            else:
                pae_mtx = None

            # Load ranking data for ipTM
            ranking_file = result_dir / "ranking_debug.json"
            with open(ranking_file, "r") as f:
                ranking_data = json.load(f)

            iptm_score = ranking_data.get("iptm", {}).get(model_name)

            return pae_mtx, iptm_score

        except Exception as e:
            st.warning(f"Could not load PAE/ipTM data from {result_dir}: {e}")
            return None, None

    def get_all_models(self, result_dir: Path) -> List[Dict]:
        """Get information about all models in a prediction directory"""
        try:
            ranking_file = result_dir / "ranking_debug.json"
            if not ranking_file.exists():
                return []

            with open(ranking_file, "r") as f:
                ranking_data = json.load(f)

            models = []
            for rank_idx, model_name in enumerate(ranking_data["order"]):
                model_info = {
                    "rank": rank_idx,
                    "model_name": model_name,
                    "iptm": ranking_data.get("iptm", {}).get(model_name, 0.0),
                    "iptm_ptm": ranking_data.get("iptm+ptm", {}).get(model_name, 0.0),
                    "pdb_file": result_dir / f"ranked_{rank_idx}.pdb",
                }
                models.append(model_info)

            return models

        except Exception as e:
            st.warning(f"Could not load models from {result_dir}: {e}")
            return []

    def analyze_directory(self) -> pd.DataFrame:
        """Analyze all prediction jobs in the directory"""
        jobs = [d for d in self.output_dir.iterdir() if d.is_dir()]

        results = []
        progress_bar = st.progress(0)
        status_text = st.empty()

        for idx, job_dir in enumerate(jobs):
            status_text.text(f"Processing {job_dir.name} ({idx + 1}/{len(jobs)})")

            ranking_file = job_dir / "ranking_debug.json"
            if not ranking_file.exists():
                continue

            try:
                with open(ranking_file, "r") as f:
                    ranking_data = json.load(f)

                # Check if it's a multimer job
                if "iptm+ptm" not in ranking_data:
                    continue

                best_model = ranking_data["order"][0]
                iptm_ptm_score = ranking_data["iptm+ptm"][best_model]
                iptm_score = ranking_data.get("iptm", {}).get(best_model, 0.0)

                # Get PAE matrix for inter-chain PAE calculation
                pae_mtx, _ = self.obtain_pae_and_iptm(job_dir, best_model)

                # Calculate mean inter-chain PAE if available
                mean_pae = None
                if pae_mtx is not None:
                    seq_lengths = self.obtain_seq_lengths(job_dir)
                    if seq_lengths:
                        mean_pae = self.calculate_mean_inter_pae(pae_mtx, seq_lengths)

                results.append(
                    {
                        "job": job_dir.name,
                        "iptm_ptm": iptm_ptm_score,
                        "iptm": iptm_score,
                        "mean_pae": mean_pae,
                        "best_model": best_model,
                        "path": str(job_dir),
                        "n_models": len(ranking_data["order"]),
                    }
                )

            except Exception as e:
                st.warning(f"Error processing {job_dir.name}: {e}")

            progress_bar.progress((idx + 1) / len(jobs))

        progress_bar.empty()
        status_text.empty()

        if results:
            df = pd.DataFrame(results)
            df = df.sort_values("iptm", ascending=False).reset_index(drop=True)
            return df
        else:
            return pd.DataFrame()

    def calculate_mean_inter_pae(
        self, pae_mtx: np.ndarray, seq_lengths: List[int]
    ) -> float:
        """Calculate mean PAE for inter-chain regions"""
        try:
            pae_copy = pae_mtx.copy()
            old_length = 0
            for length in seq_lengths:
                new_length = old_length + length
                pae_copy[old_length:new_length, old_length:new_length] = np.nan
                old_length = new_length

            return float(np.nanmean(pae_copy))
        except Exception:
            return None


def plot_pae_heatmap(pae_file: Path, figsize: Tuple[int, int] = (10, 10)) -> plt.Figure:
    """Create PAE heatmap plot"""
    try:
        with open(pae_file, "r") as f:
            pae_data = json.load(f)

        pae_mtx = np.array(pae_data[0]["predicted_aligned_error"])

        fig, ax = plt.subplots(figsize=figsize)
        im = ax.imshow(pae_mtx, cmap="Greens_r", vmin=0, vmax=30)
        ax.set_xlabel("Scored residue", fontsize=12)
        ax.set_ylabel("Aligned residue", fontsize=12)
        ax.set_title("Predicted Aligned Error (PAE)", fontsize=14, fontweight="bold")

        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label("Expected position error (Å)", rotation=270, labelpad=20)

        return fig
    except Exception as e:
        st.error(f"Could not create PAE plot: {e}")
        return None


def plot_model_comparison(models: List[Dict]) -> plt.Figure:
    """Create bar plot comparing all models"""
    try:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        ranks = [m["rank"] for m in models]
        iptm_scores = [m["iptm"] for m in models]
        iptm_ptm_scores = [m["iptm_ptm"] for m in models]

        # ipTM scores
        ax1.bar(ranks, iptm_scores, color="steelblue", alpha=0.8)
        ax1.set_xlabel("Model Rank", fontsize=11)
        ax1.set_ylabel("ipTM Score", fontsize=11)
        ax1.set_title("ipTM Scores", fontsize=12, fontweight="bold")
        ax1.set_xticks(ranks)
        ax1.set_ylim([0, 1])
        ax1.grid(axis="y", alpha=0.3)

        # ipTM+pTM scores
        ax2.bar(ranks, iptm_ptm_scores, color="coral", alpha=0.8)
        ax2.set_xlabel("Model Rank", fontsize=11)
        ax2.set_ylabel("ipTM+pTM Score", fontsize=11)
        ax2.set_title("ipTM+pTM Scores", fontsize=12, fontweight="bold")
        ax2.set_xticks(ranks)
        ax2.set_ylim([0, 1])
        ax2.grid(axis="y", alpha=0.3)

        plt.tight_layout()
        return fig

    except Exception as e:
        st.error(f"Could not create comparison plot: {e}")
        return None


def create_3dmol_view(
    pdb_file: Path, color_by: str = "plddt", width: int = 800, height: int = 600
) -> str:
    """Create HTML for 3Dmol.js viewer"""
    try:
        with open(pdb_file, "r") as f:
            pdb_data = f.read()

        # Color schemes
        if color_by == "plddt":
            color_scheme = """
                viewer.setStyle({}, {
                    cartoon: {colorscheme: {
                        prop: 'b',
                        gradient: 'roygb',
                        min: 50,
                        max: 90
                    }}
                });
            """
        else:  # color by chain
            color_scheme = """
                viewer.setStyle({}, {cartoon: {colorscheme: 'chain'}});
            """

        html = f"""
        <div id="container" style="width: 100%; height: {height}px; position: relative;"></div>
        <script src="https://3Dmol.org/build/3Dmol-min.js"></script>
        <script>
            let viewer = $3Dmol.createViewer("container", {{
                backgroundColor: 'white'
            }});

            let pdbData = `{pdb_data}`;

            viewer.addModel(pdbData, "pdb");
            {color_scheme}
            viewer.zoomTo();
            viewer.render();
            viewer.zoom(0.8, 1000);
        </script>
        """

        return html
    except Exception as e:
        st.error(f"Could not create 3D view: {e}")
        return ""


def get_pae_file_for_model(job_path: Path, model_name: str) -> Optional[Path]:
    """Find the PAE file for a specific model"""
    # Try direct match
    pae_file = job_path / f"pae_{model_name}.json"
    if pae_file.exists():
        return pae_file

    # Try extracting model number
    model_num = model_name.split("_")[1] if "_" in model_name else model_name
    pae_file = job_path / f"pae_model_{model_num}.json"
    if pae_file.exists():
        return pae_file

    return None
