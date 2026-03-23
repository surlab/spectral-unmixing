# Figure 1.5 Notes and Questions

## Main Figure Description
Based on dual_domain_subpanel5 scatterplot with labels and arrows added.

### Requirements:
1. **No shading for classifiers** - remove classification zone shading
2. **No predicted dashed line** - remove the predicted vector (dashed arrow)
3. **Keep data vector** (solid arrow) - this should remain

### Labels and Arrows to Add:

1. **Y-axis scaling arrow** (Channel 2 scaling)
   - Location: Near Y-axis, from ~1500 to ~3000
   - Type: Thick double-ended arrow
   - Label: "scaling channel 2 (laser power, PMT amplification, filter collection efficiency)"

2. **mNeptune vector scaling arrow** (Pixel brightness scaling)
   - Location: Near end of mNeptune vector, centered around (1500, 1500)
   - Type: Thick double-ended arrow along mNeptune vector direction
   - Label: "scaling pixel brightness (FP concentration, FP brightness, objective collection efficiency, net dwell, ROI size)"

3. **Perpendicular variance arrow** (Variance around mean angle)
   - Location: Perpendicular to mNeptune vector, further out, maybe around (2500, 2000)
   - Type: Double-headed arrow perpendicular to mNeptune vector
   - Label: "variance around mean angle (total collection fraction, angle of vector)"

4. **Angle label update**
   - Current: Shows actual angle measurement in degrees
   - New: Label with "angle of separation (emission filter and excitation wavelengths)"
   - Location: Same as current angle arc/label

5. **Noise arrows near origin**
   - Location: Near origin (0, 0)
   - Type: Multiple arrows of different lengths and directions
   - Can be negative (pointing toward negative values)
   - Label: "detector noise, background and dark noise"

## Supplement Panels

1. **Power comparison scatterplots**
   - Same Ch2 filter set with different power vs same Ch1
   - Two scatter plots side by side (one for each Ch2 power level)
   - Plot all valid paired scatterplots, save in subdirectory (2 subplots per figure)
   - Data source: May need fig2 data (same setup, only different pockels)

2. **Variance vs angle scatterplot**
   - X-axis: Angle (FP angle, not separation angle)
   - Y-axis: Variance around mean
   - Color: By FP
   - Variance taken at consistent distance from origin
   - Plot for many pairs

3. **Variance vs distance line plots**
   - X-axis: Distance from origin along vector
   - Y-axis: (Relative?) variance around mean
   - Line plots (not scatter)

4. **Percent correct vs ROI size**
   - X-axis: ROI size
   - Y-axis: Percent correct
   - Subselected within cells

5. **Dwell time vs percent correct**
   - X-axis: Dwell time
   - Y-axis: Percent correct
   - Data source: fig3/change_Averaging

6. **Example scatterplots**
   - Two scatterplots: 800 and 1080 red vs 800 and 1080 broad red
   - For neptune and cherry
   - Ideally same power in mW (may need Fig 2 data)

## Answers:

1. **Main figure base**: Use dual_domain_subpanel5 as base, remove classification zones and predicted dashed vector, keep data vector.

2. **Arrow styling**: 8-10x thicker than current vector lines. Filled shapes with nice triangle heads (like Google Slides style), not lines cobbled together.

3. **Arrow positioning**: Position automatically based on data (the coordinates mentioned were approximate from visual inspection).

4. **Noise arrows**: 5 arrows total. Random directions, random lengths (max 500 pixel values), reasonably short.

5. **Supplement panel 1**: 
   - Data source: `data/fig2_3color_inh_spatial_control_2p3_10072025`
   - Note: All fluorophores present in same image (no need to plot multiple separately)
   - Generate all valid paired scatterplots (Ch1 same, Ch2 different power), save in subdirectory (2 subplots per figure) for selection

6. **Supplement panel 2**: Use multiple distances: 500, 1500, and 2500 pixel values from origin.

7. **Supplement panel 3**: Create two plots:
   - One with raw variance
   - One normalized to its own mean variance

8. **Data sources**: 
   - Supplement panel 1: `data/fig2_3color_inh_spatial_control_2p3_10072025`
   - Supplement panel 5 (dwell time): `data/fig3_3color_inh_mixed/change_averaging_resonant`

9. **ROI size plots**: Skip for now (supplement panel 4).

10. **Code organization**: See recommendation below.

## Code Organization Recommendation:

**Recommendation: Single file (`figure1_5.py`) with clear function organization**

**Rationale:**
- Figure 1.5 is a cohesive unit (main figure + supplements that demonstrate the same concepts)
- All panels share similar data sources and plotting needs
- Single file is easier to navigate and understand for scientific code
- Clear function names and docstrings provide organization
- If it grows too large later, we can refactor

**Structure:**
```python
# figure1_5.py
# - Main figure function
# - Helper functions for arrows, labels, etc.
# - Supplement panel functions (1, 2, 3, 5, 6)
# - Shared helper functions (variance calculations, data loading, etc.)
```

**Alternative (if it gets too large):**
- `figure1_5_main.py` - main figure
- `figure1_5_supplements.py` - supplement panels
- `figure_helpers.py` - shared utilities (if needed by multiple figures)

For now, I recommend starting with a single file and refactoring later if needed.

