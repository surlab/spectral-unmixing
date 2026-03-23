# Figure 1.5 Implementation Plan

## Code Organization

**Recommendation: Single file (`figure1_5.py`) with clear function organization**

**Rationale:**
- Figure 1.5 is a cohesive unit (main figure + supplements demonstrating the same concepts)
- All panels share similar data sources and plotting needs  
- Single file is easier to navigate and understand for scientific code
- Clear function names and docstrings provide organization
- If it grows too large later, we can refactor

**Structure:**
```python
# figure1_5.py
# - Main figure function (based on dual_domain_subpanel5)
# - Helper functions for arrows, labels, etc.
# - Supplement panel functions (1, 2, 3, 5, 6)
# - Shared helper functions (variance calculations, data loading, etc.)
```

## Implementation Details

### Main Figure

**Base**: Use `subpanel_5` from `Row3_dict` (dual domain) as starting point

**Modifications**:
- Remove classification zone shading (wedges)
- Remove predicted vector (dashed arrow)
- Keep data vector (solid arrow)
- Keep scatterplot with subsampled points

**New Elements**:

1. **Y-axis scaling arrow** (Channel 2)
   - Position: Near Y-axis, from ~1500 to ~3000 (auto-position based on data)
   - Style: Double-ended arrow, 8-10x thicker than current vectors (linewidth=16-20)
   - Label: "scaling channel 2 (laser power, PMT amplification, filter collection efficiency)"

2. **mNeptune vector scaling arrow** (Pixel brightness)
   - Position: Along mNeptune vector, near end (auto-position based on data)
   - Style: Double-ended arrow along vector direction, 8-10x thicker
   - Label: "scaling pixel brightness (FP concentration, FP brightness, objective collection efficiency, net dwell, ROI size)"

3. **Perpendicular variance arrow**
   - Position: Perpendicular to mNeptune vector, further out (auto-position)
   - Style: Double-ended arrow perpendicular to vector, 8-10x thicker
   - Label: "variance around mean angle (total collection fraction, angle of vector)"

4. **Angle label update**
   - Replace current angle measurement with: "angle of separation (emission filter and excitation wavelengths)"
   - Keep same location (near arc between vectors)

5. **Noise arrows** (5 total)
   - Position: Near origin (0, 0)
   - Style: Random directions, random lengths (max 500 pixel values)
   - Can point in negative directions
   - Label: "detector noise, background and dark noise"

**Arrow Implementation**:
- Current vectors: `linewidth=2`, `mutation_scale=18`
- Thick arrows: `linewidth=16-20`, proportionally larger `mutation_scale` (144-180)
- Use `FancyArrowPatch` with `arrowstyle='<->'` for double-ended
- For filled triangle heads, may need custom `Polygon` patches or `FancyBboxPatch`

### Supplement Panels

**Panel 1: Power comparison scatterplots**
- Data: `data/fig2_3color_inh_spatial_control_2p3_10072025`
- Note: All fluorophores in same image (no separate plotting needed)
- Generate all valid pairs: same Ch1, different Ch2 power (different Pockels)
- Save in subdirectory: 2 subplots per figure
- User will select which ones to use

**Panel 2: Variance vs angle scatterplot**
- X-axis: Angle (FP angle, not separation angle)
- Y-axis: Variance around mean
- Color: By FP
- Variance at distances: 500, 1500, 2500 from origin
- Plot for many pairs

**Panel 3: Variance vs distance line plots**
- Two plots:
  - Raw variance vs distance from origin along vector
  - Normalized variance (to its own mean) vs distance from origin along vector
- X-axis: Distance from origin along vector
- Y-axis: Variance around mean (raw or normalized)

**Panel 4: Percent correct vs ROI size** - SKIP FOR NOW

**Panel 5: Dwell time vs percent correct**
- Data: `data/fig3_3color_inh_mixed/change_averaging_resonant`
- X-axis: Dwell time
- Y-axis: Percent correct

**Panel 6: Example scatterplots**
- Two scatterplots: 800 and 1080 red vs 800 and 1080 broad red
- For neptune and cherry
- Ideally same power in mW (may need Fig 2 data)

## Key Functions Needed

1. `figure1_5_main()` - Main figure with labeled arrows
2. `_add_scaling_arrow()` - Helper for adding double-ended scaling arrows
3. `_add_noise_arrows()` - Helper for adding noise arrows near origin
4. `supplement_panel_1()` - Power comparison scatterplots
5. `supplement_panel_2()` - Variance vs angle scatterplot
6. `supplement_panel_3()` - Variance vs distance line plots
7. `supplement_panel_5()` - Dwell time vs percent correct
8. `supplement_panel_6()` - Example scatterplots
9. Shared helpers: variance calculations, data loading from fig2/fig3 directories




