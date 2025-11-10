# APLit

A web interface for analyzing AlphaPulldown predictions.

- 🎨 Clean, responsive interface with real-time filtering
- 📊 Interactive PAE heatmaps and 3D structure viewer
- 🔄 Optional auto-refresh for monitoring running predictions
- ⚡ Fast performance with large prediction datasets
- 📥 Export results and download structures

## Installation

**Requirements:** Python 3.8+
```bash
pip install git+https://github.com/KosinskiLab/aplit.git
```

## Usage

Launch the server locally:
```bash
aplit
```

This will open your web browser. In **Configuration > Predictions Directory**, enter the path to an AlphaPulldown run (currently only tested with AF2 backend).

```
/path/to/predictions/
├── protein1_and_protein2/
│   ├── ranking_debug.json
│   ├── ranked_0.pdb
│   ├── pae_model_1_ptm_pred_0.json
│   └── ...
├── protein1_and_protein3/
│   └── ...
```

## Links

- [AlphaPulldown](https://github.com/KosinskiLab/AlphaPulldown)
