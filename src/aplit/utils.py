"""
Utility functions and data classes for AlphaPulldown analysis
"""

import csv
import json
import math

from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from pandas.errors import EmptyDataError

import streamlit as st
import matplotlib.pyplot as plt


class AlphaPulldownAnalyzer:
    """Handles analysis of AlphaPulldown prediction directories"""

    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self._job_cache: Dict[Path, Dict[str, Any]] = {}

    @staticmethod
    def _safe_float(value: Any, default: Optional[float] = None) -> Optional[float]:
        try:
            if value is None:
                return default
            val = float(value)
            if math.isnan(val):
                return default
            return val
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _select_best_interface_row(df: pd.DataFrame) -> Optional[pd.Series]:
        """Return the interface row with the highest ipTM (fallback to ipTM+PTM)."""
        if df is None or df.empty:
            return None

        working_df = df.copy()
        for col in ["iptm", "iptm_ptm"]:
            if col in working_df.columns:
                working_df[col] = pd.to_numeric(working_df[col], errors="coerce")

        sort_cols = [col for col in ["iptm", "iptm_ptm"] if col in working_df.columns]
        if sort_cols:
            working_df = working_df.sort_values(
                sort_cols, ascending=False, na_position="last"
            )

        try:
            best_idx = working_df.index[0]
        except IndexError:
            return None
        return df.loc[best_idx]

    def _detect_job_type(self, job_dir: Path) -> Optional[str]:
        if (job_dir / "ranking_debug.json").exists():
            return "af2"
        if (job_dir / "ranking_scores.csv").exists():
            return "af3"
        return None

    def _load_af2_models(self, job_dir: Path) -> List[Dict[str, Any]]:
        ranking_file = job_dir / "ranking_debug.json"
        if not ranking_file.exists():
            return []

        with ranking_file.open() as f:
            ranking_data = json.load(f)

        order = ranking_data.get("order", [])
        models: List[Dict[str, Any]] = []
        for rank_idx, model_name in enumerate(order):
            structure_path = job_dir / f"ranked_{rank_idx}.pdb"
            models.append(
                {
                    "rank": rank_idx,
                    "model_name": model_name,
                    "iptm": ranking_data.get("iptm", {}).get(model_name, 0.0),
                    "iptm_ptm": ranking_data.get("iptm+ptm", {}).get(model_name, 0.0),
                    "ptm": ranking_data.get("ptm", {}).get(model_name),
                    "structure_file": structure_path,
                    "structure_format": "pdb",
                    "pdb_file": structure_path,
                    "job_type": "af2",
                }
            )
        return models

    def _load_af3_models(self, job_dir: Path) -> List[Dict[str, Any]]:
        ranking_path = job_dir / "ranking_scores.csv"
        if not ranking_path.exists():
            return []

        with ranking_path.open(newline="") as f:
            rows = [row for row in csv.DictReader(f) if row]

        def score(row: Dict[str, Any]) -> float:
            val = self._safe_float(row.get("ranking_score"), float("-inf"))
            return val if val is not None else float("-inf")

        rows.sort(key=score, reverse=True)

        models: List[Dict[str, Any]] = []
        for rank_idx, row in enumerate(rows):
            seed = row.get("seed")
            sample = row.get("sample")
            if seed is None or sample is None:
                continue

            model_name = f"seed-{seed}_sample-{sample}"
            model_dir = job_dir / model_name
            summary_path = model_dir / "summary_confidences.json"
            summary: Dict[str, Any] = {}

            if not summary_path.exists():
                fallback_summary = job_dir / f"ranked_{rank_idx}_summary_confidences.json"
                if fallback_summary.exists():
                    summary_path = fallback_summary

            if summary_path.exists():
                try:
                    with summary_path.open() as f:
                        summary = json.load(f)
                except Exception as exc:
                    st.warning(f"Could not read {summary_path}: {exc}")

            iptm = self._safe_float(summary.get("iptm"), 0.0)
            ptm = self._safe_float(summary.get("ptm"))
            iptm_ptm = self._safe_float(
                summary.get("ranking_score")
                or summary.get("iptm+ptm")
                or row.get("ranking_score"),
                0.0,
            )

            structure_path = model_dir / "model.cif"
            if not structure_path.exists():
                ranked_alt = job_dir / f"ranked_{rank_idx}_model.cif"
                if ranked_alt.exists():
                    structure_path = ranked_alt
                else:
                    pdb_alt = job_dir / f"ranked_{rank_idx}.pdb"
                    if pdb_alt.exists():
                        structure_path = pdb_alt
            suffix = structure_path.suffix.lower()
            structure_format = "mmcif" if suffix in {".cif", ".mmcif"} else "pdb"

            models.append(
                {
                    "rank": rank_idx,
                    "model_name": model_name,
                    "iptm": iptm if iptm is not None else 0.0,
                    "iptm_ptm": iptm_ptm if iptm_ptm is not None else 0.0,
                    "ptm": ptm,
                    "structure_file": structure_path,
                    "structure_format": structure_format,
                    "pdb_file": structure_path,
                    "job_type": "af3",
                    "confidence_score": self._safe_float(summary.get("confidence_score")),
                }
            )

        return models

    def _get_job_models(self, job_dir: Path) -> Optional[Dict[str, Any]]:
        job_dir = job_dir.resolve()
        if job_dir in self._job_cache:
            return self._job_cache[job_dir]

        job_type = self._detect_job_type(job_dir)
        if job_type == "af2":
            models = self._load_af2_models(job_dir)
        elif job_type == "af3":
            models = self._load_af3_models(job_dir)
        else:
            models = []

        info = {"job_type": job_type, "models": models}
        self._job_cache[job_dir] = info
        return info if job_type else None

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
            info = self._get_job_models(result_dir)
            if not info:
                return []
            return info.get("models", [])
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

            try:
                job_info = self._get_job_models(job_dir)
                if not job_info or not job_info.get("models"):
                    continue

                models = job_info["models"]
                job_type = job_info["job_type"]

                best_model_info = models[0]
                best_model = best_model_info["model_name"]
                iptm_ptm_score = best_model_info.get("iptm_ptm", 0.0) or 0.0
                iptm_score = best_model_info.get("iptm", 0.0) or 0.0

                mean_pae = None
                if job_type == "af2":
                    pae_mtx, _ = self.obtain_pae_and_iptm(job_dir, best_model)
                    if pae_mtx is not None:
                        seq_lengths = self.obtain_seq_lengths(job_dir)
                        if seq_lengths:
                            mean_pae = self.calculate_mean_inter_pae(
                                pae_mtx, seq_lengths
                            )

                interface_df = load_interfaces_csv(job_dir)
                interface_summary = (
                    self._select_best_interface_row(interface_df)
                    if interface_df is not None
                    else None
                )

                if interface_summary is not None:
                    iptm_score = (
                        float(interface_summary.get("iptm"))
                        if pd.notna(interface_summary.get("iptm"))
                        else iptm_score
                    )
                    iptm_ptm_score = (
                        float(interface_summary.get("iptm_ptm"))
                        if pd.notna(interface_summary.get("iptm_ptm"))
                        else iptm_ptm_score
                    )
                    mean_pae = (
                        float(interface_summary.get("average_interface_pae"))
                        if pd.notna(interface_summary.get("average_interface_pae"))
                        else mean_pae
                    )

                results.append(
                    {
                        "job": job_dir.name,
                        "iptm_ptm": iptm_ptm_score,
                        "iptm": iptm_score,
                        "mean_pae": mean_pae,
                        "best_model": best_model,
                        "path": str(job_dir),
                        "n_models": len(models),
                        "job_type": job_type,
                        "ptm": float(interface_summary.get("ptm"))
                        if interface_summary is not None
                        and pd.notna(interface_summary.get("ptm"))
                        else None,
                        "confidence_score": float(
                            interface_summary.get("confidence_score")
                        )
                        if interface_summary is not None
                        and pd.notna(interface_summary.get("confidence_score"))
                        else None,
                        "global_dockq": float(
                            interface_summary.get("pDockQ/mpDockQ")
                        )
                        if interface_summary is not None
                        and pd.notna(interface_summary.get("pDockQ/mpDockQ"))
                        else None,
                        "interface_csv": str(job_dir / "interfaces.csv")
                        if interface_df is not None
                        else None,
                        "interface_summary_model": interface_summary.get("model_used")
                        if interface_summary is not None
                        else None,
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
    structure_file: Path,
    color_by: str = "plddt",
    structure_format: str = "pdb",
    width: int = 800,
    height: int = 600,
) -> str:
    """Create HTML for 3Dmol.js viewer"""
    try:
        with open(structure_file, "r") as f:
            structure_data = f.read()

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

        data_literal = json.dumps(structure_data)
        model_format = (
            "mmcif"
            if structure_format.lower() in {"mmcif", "cif"}
            else structure_format.lower()
        )

        html = f"""
        <div id="container" style="width: 100%; height: {height}px; position: relative;"></div>
        <script src="https://3Dmol.org/build/3Dmol-min.js"></script>
        <script>
            let viewer = $3Dmol.createViewer("container", {{
                backgroundColor: 'white'
            }});

            let pdbData = {data_literal};

            viewer.addModel(pdbData, "{model_format}");
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


def load_interfaces_csv(job_path: Path) -> Optional[pd.DataFrame]:
    """Load AlphaJudge interfaces.csv if it exists and is non-empty."""
    csv_path = job_path / "interfaces.csv"
    if not csv_path.exists() or csv_path.stat().st_size == 0:
        return None

    try:
        df = pd.read_csv(csv_path)
        if df.empty:
            return None
        return df
    except EmptyDataError:
        return None
    except Exception as e:
        st.warning(f"Could not load interfaces.csv from {job_path}: {e}")
        return None


def get_pae_plot_image(job_path: Path, model_name: str, rank: int) -> Optional[Path]:
    """Locate a precomputed PAE PNG image."""

    def _candidate_exists(path: Path) -> Optional[Path]:
        return path if path.exists() else None

    candidates = [
        job_path / f"pae_{model_name}.png",
        job_path / f"pae_plot_ranked_{rank}.png",
        job_path / f"pae_plot_ranked_{rank}.PNG",
    ]

    # Try a simplified model identifier if possible
    if "_" in model_name:
        model_num = model_name.split("_")[1]
        candidates.extend(
            [
                job_path / f"pae_model_{model_num}.png",
                job_path / f"pae_model_{model_num}_ptm_pred_0.png",
            ]
        )

    for candidate in candidates:
        existing = _candidate_exists(candidate)
        if existing:
            return existing

    return None
