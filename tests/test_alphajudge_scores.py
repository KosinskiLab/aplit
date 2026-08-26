import pandas as pd

from aplit.utils import AlphaPulldownAnalyzer


def test_all_current_alphajudge_numeric_scores_are_extracted() -> None:
    row = pd.Series(
        {
            "jobs": "complex",
            "model_used": "model_1",
            "interface": "A_B",
            "interface_contact_prob_source": "distogram",
            "iptm_ptm": 0.82,
            "iptm": 0.76,
            "ptm": 0.71,
            "confidence_score": 0.81,
            "pDockQ/mpDockQ": 0.63,
            "average_interface_pae": 4.2,
            "interface_average_plddt": 88.0,
            "interface_num_intf_residues": 42,
            "interface_polar": 0.3,
            "interface_hydrophobic": 0.5,
            "interface_charged": 0.2,
            "interface_contact_pairs": 125,
            "interface_contact_prob_max": 0.96,
            "interface_contact_prob_top10_mean": 0.91,
            "interface_expected_contacts": 31.4,
            "interface_ccc": 0.78,
            "interface_score": 184.6,
            "interface_pDockQ2": 0.72,
            "interface_ipSAE": 0.69,
            "interface_LIS": 0.74,
            "interface_cLIS": 0.66,
            "interface_iLIS": 0.70,
            "interface_hb": 8,
            "interface_sb": 2,
            "interface_ss": 1,
            "interface_sc": 0.67,
            "interface_area": 1450.0,
            "interface_solv_en": -18.5,
            "interface_meta_score": 0.84,
        }
    )

    metrics = AlphaPulldownAnalyzer._extract_interface_metrics(row)

    assert "jobs" not in metrics
    assert "interface_contact_prob_source" not in metrics
    assert metrics["global_dockq"] == 0.63
    assert metrics["mean_pae"] == 4.2
    assert metrics["interface_meta_score"] == 0.84
    assert metrics["interface_cLIS"] == 0.66
    assert metrics["interface_iLIS"] == 0.70
    assert metrics["interface_contact_prob_top10_mean"] == 0.91
    assert metrics["interface_ccc"] == 0.78
    assert metrics["interface_hb"] == 8
    assert metrics["interface_sc"] == 0.67


def test_best_interface_prefers_alphajudge_meta_score() -> None:
    interfaces = pd.DataFrame(
        [
            {"interface": "A_B", "interface_meta_score": 0.4, "iptm": 0.9},
            {"interface": "A_C", "interface_meta_score": 0.8, "iptm": 0.6},
        ]
    )

    selected = AlphaPulldownAnalyzer._select_best_interface_row(interfaces)

    assert selected is not None
    assert selected["interface"] == "A_C"
