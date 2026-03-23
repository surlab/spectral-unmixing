# Figure 5 Implementation Notes

## Overview
- **Data**: All simulated (fluorophore spectra are real data from FPbase)
- **Channels**: Start with all 42 combinations (6 excitations × 7 filters)
- **File**: Separate `figure5.py`, import from helpers and figure1 when possible

## Row Dictionary
```python
fig_5_row_dict = {
    "name": "fig_5",
    "Fluorophores": ["EBFP", "tagBFP", "mTFP1", "GCamp", "LSSmOrange", "TdTomato", "mCherry", "LSSmKAte", "mNeptune"],
    "Excitation wavelengths": [750, 800, 870, 1040, 1180, 1240],
    "emission filters": [[400,440], [445,475], [475,495], [500,550], [550,580], [590,620], [645,695]]
}
```

## Subpanel Details

### Subpanel 1: 2P Excitation Spectra
- Plot 2P excitation spectra for all 9 fluorophores
- Overlay excitation wavelengths as vertical lines (750, 800, 870, 1040, 1180, 1240)
- Range: 950-1250 (default, can override)
- **Special case**: LSSmKate has no 2P spectra - copy the column from mTFP1
- Use/modify existing `plot.ex_em_spectra` function (create new copy to avoid breaking old code)
- Download updated spectra from FPbase

### Subpanel 2: 1P Emission Spectra
- Plot 1P emission spectra for all 9 fluorophores
- Overlay emission filters as shaded regions
- Filter ranges: [400,440], [445,475], [475,495], [500,550], [550,580], [590,620], [645,695]

### Subpanel 3: Predicted Unmixing Ratios (Future)
- Show all 42 bars (all channel combinations)
- Sort by FP preference: BFP first, then tagBFP, then TFP, etc.
- Similar to Figure 1 subpanel 4 (bar charts)

### Subpanel 4: t-SNE (Future)
- All 42 channels (t-SNE reduces dimensions)
- 1000 simulated pixels
- Low perplexity
- Goal: star shape with arm for each fluorophore

### Subpanel 5: Angle Plots (Future)
- 9 plots total (1 per target FP)
- Angle relative to target FP mean vector
- Stacked histogram: all non-target FPs
- Overlap with target FP histogram
- Use same plotting code as fig 234 subpanel 7

## Fluorophore Colors
- EBFP: dark blue
- tagBFP: lighter blue
- mTFP1: teal
- GCamp: green
- LSSmOrange: orange
- TdTomato: yellow (same as fig 2)
- mCherry: red (same as fig 1)
- LSSmKAte: dark red
- mNeptune: purple (same as fig 1)

## Excitation Strength Optimization (Future)
- Options: 10, 20, 30, 40 mW
- Start all at 20 mW
- Simple optimization: try wiggling each up/down to see if it helps maximize angles

## Filter Data
- Filter CSVs from Chroma are available
- Use existing filter system from Figure 1

## Priority
1. **Start with Subpanels 1 and 2** (avoid simulation and optimization for now)
2. Get spectra downloading/loading working
3. Create basic plotting structure
4. Then move to simulation and optimization later



