"""
New Figure 1 generation for spectral unmixing methods paper.

This module generates the new Figure 1 (Pipeline graphic) using the refactored
generic plotting functions and config-driven approach.

Subpanels:
- 1a (Function A): TODO placeholder panel
- 1b (Function B): Excitation spectra
- 1c (Function C): Emission spectra
- 1d (Function D): Unmixing vectors as bar chart
- 1e (Function E): Predicted angle between FPs
- Additional subpanels will be added as generic functions are implemented
"""

import os
import copy
import json
import hashlib
import pickle
import numpy as np
import matplotlib.pyplot as plt
import sys

from src import figure_1_config as fig_cfg
from src import config as cfg
from src.figure_plotting import (
    # Panel letters shifted forward by one (new Function A placeholder).
    A_excitation_spectra_plot as B_excitation_spectra_plot,
    B_emission_spectra_plot as C_emission_spectra_plot,
    C_unmixing_vectors_bar_chart_plot as D_unmixing_vectors_bar_chart_plot,
    D_predicted_angle_with_nearest_linear_combo_plot as E_predicted_angle_with_nearest_linear_combo_plot,
    F_two_channel_scatterplot_with_vectors_and_cones_plot as G_two_channel_scatterplot_with_vectors_and_cones_plot,
    H_three_channel_3d_scatterplot_with_vectors_and_cones_plot as H_three_channel_3d_scatterplot_with_vectors_and_cones_plot,
    I_three_channel_triangle_projection_plot as I_three_channel_triangle_projection_plot,
    J_angle_histogram_plot as J_angle_histogram_plot,
    # Additional functions will be imported as they are implemented
    # C_unmixing_vectors_bar_chart_plot,
    # etc.
)


class _TeeStream:
    """Write to both the original stream (console) and a log file."""

    def __init__(self, original_stream, file_handle):
        self._original = original_stream
        self._file = file_handle

    def write(self, data):
        try:
            self._original.write(data)
        except Exception:
            pass
        try:
            self._file.write(data)
        except Exception:
            pass

    def flush(self):
        try:
            self._original.flush()
        except Exception:
            pass
        try:
            self._file.flush()
        except Exception:
            pass


def generate_subpanel_1a(params_dict=None, ax=None, figure_params=None):
    """
    Generate subpanel 1a: Panel A placeholder (Function A).
    
    Parameters
    ----------
    params_dict : dict, optional
        Complete configuration dictionary. If None, uses from figure_params.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, creates new figure.
    figure_params : dict, optional
        Complete figure configuration dictionary. If None, uses figure_1_params from config.
        
    Returns
    -------
    fig : matplotlib.figure.Figure
    ax : matplotlib.axes.Axes
    """
    if params_dict is None:
        if figure_params is None:
            figure_params = fig_cfg.figure_1_params_presentation
        
        params_dict = figure_params.get("1a", {})

    placeholder_text = params_dict.get("placeholder_text", "TODO: add placeholder description for Panel A")
    if ax is None:
        fig, ax = plt.subplots(figsize=params_dict.get("figsize", (8, 5)))
    else:
        fig = ax.figure

    ax.axis("off")
    ax.text(
        0.5,
        0.5,
        placeholder_text,
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=12,
    )
    return fig, ax


def generate_subpanel_1b(params_dict=None, ax=None, figure_params=None):
    """
    Generate subpanel 1b: Excitation spectra (Function B).
    """
    if params_dict is None:
        if figure_params is None:
            figure_params = fig_cfg.figure_1_params_presentation

        params_dict = figure_params.get("1b", {})

    return B_excitation_spectra_plot(params_dict, ax=ax)


def generate_subpanel_1c(params_dict=None, ax=None, figure_params=None):
    """
    Generate subpanel 1c: Emission spectra with emission filters (Function C).
    
    Parameters
    ----------
    params_dict : dict, optional
        Complete configuration dictionary. If None, uses from figure_params.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, creates new figure.
    figure_params : dict, optional
        Complete figure configuration dictionary. If None, uses figure_1_params from config.
        
    Returns
    -------
    fig : matplotlib.figure.Figure
    ax : matplotlib.axes.Axes
    """
    if params_dict is None:
        if figure_params is None:
            figure_params = fig_cfg.figure_1_params_presentation
        
        params_dict = figure_params.get("1c", {})
    
    return C_emission_spectra_plot(params_dict, ax=ax)


def generate_subpanel_1d(params_dict=None, ax=None, figure_params=None):
    """
    Generate subpanel 1d: Unmixing vectors as bar chart (Function D).
    """
    if params_dict is None:
        if figure_params is None:
            figure_params = fig_cfg.figure_1_params_presentation
        params_dict = figure_params.get("1d", {})
    return D_unmixing_vectors_bar_chart_plot(params_dict, ax=ax)


def generate_subpanel_1e(params_dict=None, ax=None, figure_params=None):
    """
    Generate subpanel 1e: Predicted angle between FPs (Function E).
    """
    if params_dict is None:
        if figure_params is None:
            figure_params = fig_cfg.figure_1_params_presentation
        params_dict = figure_params.get("1e", {})
    return E_predicted_angle_with_nearest_linear_combo_plot(params_dict, ax=ax)


def generate_subpanel_1g(params_dict=None, ax=None, figure_params=None):
    """
    Generate subpanel 1g: 2-channel 2D scatterplot with vectors + classification wedges (legacy Figure 1 subpanel 5).
    """
    if params_dict is None:
        if figure_params is None:
            figure_params = fig_cfg.figure_1_params_presentation
        params_dict = figure_params.get("1g", {})
    return G_two_channel_scatterplot_with_vectors_and_cones_plot(params_dict, ax=ax)


def generate_subpanel_1h(params_dict=None, ax=None, figure_params=None):
    """
    Generate subpanel 1h: 3-channel 3D scatterplot with classification cones (legacy figure2.subpanel_5).
    """
    if params_dict is None:
        if figure_params is None:
            figure_params = fig_cfg.figure_1_params_presentation
        params_dict = figure_params.get("1h", {})
    return H_three_channel_3d_scatterplot_with_vectors_and_cones_plot(params_dict, ax=ax)


def generate_subpanel_1i(params_dict=None, ax=None, figure_params=None):
    """
    Generate subpanel 1i: 3-channel triangle projection (legacy figure2.subpanel_6).
    """
    if params_dict is None:
        if figure_params is None:
            figure_params = fig_cfg.figure_1_params_presentation
        params_dict = figure_params.get("1i", {})
    return I_three_channel_triangle_projection_plot(params_dict, ax=ax)


def generate_subpanel_1j(params_dict=None, ax=None, figure_params=None):
    """
    Generate subpanel 1j: stacked angle histogram (Function J).
    """
    if params_dict is None:
        if figure_params is None:
            figure_params = fig_cfg.figure_1_params_presentation
        params_dict = figure_params.get("1j", {})
    return J_angle_histogram_plot(params_dict, ax=ax)


def _build_hi_shared_data_cache_path(output_dir, h_params, i_params):
    """
    Build a deterministic cache path for shared H/I classification data.
    """
    # Allow user/config override with explicit path.
    explicit = None
    if isinstance(h_params, dict):
        explicit = h_params.get("shared_data_cache_path")
    if explicit is None and isinstance(i_params, dict):
        explicit = i_params.get("shared_data_cache_path")
    if explicit:
        return explicit

    cache_dir = os.path.join(output_dir, "_cache")
    os.makedirs(cache_dir, exist_ok=True)

    key_payload = {
        "h_params": h_params if isinstance(h_params, dict) else {},
        "i_params": i_params if isinstance(i_params, dict) else {},
        "classification_zone_percentile": getattr(cfg, "classification_zone_percentile", None),
        "classification_zone_min_distance": getattr(cfg, "classification_zone_min_distance", None),
    }
    key_json = json.dumps(key_payload, sort_keys=True, default=str)
    key_hash = hashlib.md5(key_json.encode("utf-8")).hexdigest()[:12]
    return os.path.join(cache_dir, f"figure1_hi_shared_data_{key_hash}.pkl")


def _subselect_balanced_pixels(shared_data, selection_params):
    """
    Subselect pixels in a way that balances the number of points across
    angular quadrants around each fluorophore vector, per radial (amplitude) bin.

    This is applied *before* the legacy plotting functions, which themselves
    do amplitude binning and subsample up to `samples_per_bin`.
    If we keep <= that many points per bin, legacy subsampling becomes a no-op,
    preserving our intended balance.
    """
    if not isinstance(shared_data, dict):
        return shared_data

    ch1 = np.asarray(shared_data.get("ch1_valid", []), dtype=float)
    ch2 = np.asarray(shared_data.get("ch2_valid", []), dtype=float)
    ch3 = np.asarray(shared_data.get("ch3_valid", []), dtype=float)
    labels = np.asarray(shared_data.get("pixel_labels", []), dtype=object)
    fluorophores = list(shared_data.get("fluorophores", []))
    data_vectors_3d = shared_data.get("data_vectors_3d", {})

    if ch1.size == 0 or ch1.shape != ch2.shape or ch1.shape != ch3.shape or ch1.shape[0] != labels.shape[0]:
        return shared_data

    bin_width = float(selection_params.get("balanced_bin_width", 100))
    samples_per_bin = int(selection_params.get("balanced_samples_per_bin", 300))
    if samples_per_bin <= 0:
        return shared_data

    n_quadrants = max(1, len(fluorophores))
    base = samples_per_bin // n_quadrants
    remainder = samples_per_bin % n_quadrants
    targets = [base + (1 if qi < remainder else 0) for qi in range(n_quadrants)]

    max_value = float(shared_data.get("max_value", 3000.0))
    max_distance = max_value * np.sqrt(3.0)
    n_bins = int(np.ceil(max_distance / bin_width))

    # Radial distance per point (matches legacy figure2 subsampling)
    distances = np.sqrt(ch1 ** 2 + ch2 ** 2 + ch3 ** 2)

    # Project points to triangle 2D coordinates (same linear transform as figure2.subpanel_6)
    cos_30 = float(np.cos(np.radians(30.0)))
    sin_30 = float(np.sin(np.radians(30.0)))
    x_proj = cos_30 * (ch2 - ch3)
    y_proj = ch1 - sin_30 * (ch2 + ch3)

    angles_pts = np.arctan2(y_proj, x_proj)
    two_pi = 2.0 * np.pi
    ang_pts = (angles_pts + two_pi) % two_pi

    # Compute vector angles in the same 2D triangle coordinates
    vec_angles = np.zeros((n_quadrants,), dtype=float)
    for qi, fp in enumerate(fluorophores):
        vec = np.asarray(data_vectors_3d.get(fp), dtype=float)
        if vec.size < 3:
            vec_angles[qi] = 0.0
            continue
        x_vec = cos_30 * (vec[1] - vec[2])
        y_vec = vec[0] - sin_30 * (vec[1] + vec[2])
        vec_angles[qi] = float((np.arctan2(y_vec, x_vec) + two_pi) % two_pi)

    # Assign each point to nearest vector direction in circular angle space.
    # This yields sectors whose boundaries are halfway between vectors.
    # Shape: (M, n_quadrants)
    diffs = np.abs(ang_pts[:, None] - vec_angles[None, :])
    circ_diffs = np.minimum(diffs, two_pi - diffs)
    quadrant_idx = np.argmin(circ_diffs, axis=1)

    keep_indices = []
    for bin_idx in range(n_bins):
        bin_max = (bin_idx + 1) * bin_width
        prev_bin_max = bin_idx * bin_width

        if bin_idx == 0:
            bin_mask = distances < bin_max
        else:
            bin_mask = (distances >= prev_bin_max) & (distances < bin_max)

        if not np.any(bin_mask):
            continue

        idx_bin = np.nonzero(bin_mask)[0]
        q_bin = quadrant_idx[idx_bin]

        # For each quadrant, select up to its target count
        for qi in range(n_quadrants):
            idx_q = idx_bin[q_bin == qi]
            if idx_q.size == 0:
                continue
            target = targets[qi]
            if idx_q.size <= target:
                keep_indices.append(idx_q)
            else:
                chosen = np.random.choice(idx_q.size, size=target, replace=False)
                keep_indices.append(idx_q[chosen])

    if len(keep_indices) == 0:
        return shared_data

    keep_idx = np.concatenate(keep_indices)
    keep_idx = np.unique(keep_idx)  # avoid duplicates if any

    shared_data["ch1_valid"] = ch1[keep_idx]
    shared_data["ch2_valid"] = ch2[keep_idx]
    shared_data["ch3_valid"] = ch3[keep_idx]
    shared_data["pixel_labels"] = labels[keep_idx]

    shared_data["balanced_selected_count"] = int(keep_idx.size)
    return shared_data


def generate_figure_1(figure_params=None, output_dir=None, filename_prefix=""):
    """
    Generate all subpanels for new Figure 1.
    
    This is the main function that loops through all wrapper functions,
    supplying the correct arguments from the input dict (which defaults
    to what is in the config dict for this figure if called with no input).
    
    Parameters
    ----------
    figure_params : dict, optional
        Complete figure configuration dictionary. If None, uses figure_1_params from config.
    output_dir : str, optional
        Directory to save figures. If None, uses figure_1_output_dir from config.
        
    Returns
    -------
    None (saves figures to output directory)
    """
    if figure_params is None:
        figure_params = fig_cfg.figure_1_params_presentation
    
    if output_dir is None:
        output_dir = fig_cfg.figure_1_output_dir
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n" + "="*60)
    print("Generating New Figure 1 Subpanels")
    print("="*60)
    
    # Subpanel 1a: Placeholder panel A
    print("\nSubpanel 1a: Panel A placeholder...")
    try:
        fig, ax = generate_subpanel_1a(figure_params=figure_params)
        filepath = os.path.join(output_dir, f"{filename_prefix}1a.png")
        fig.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved: {filepath}")
    except Exception as e:
        print(f"  Error in subpanel 1a: {e}")
        import traceback
        traceback.print_exc()

    # Subpanel 1b: Excitation spectra (Function B)
    print("\nSubpanel 1b: Excitation spectra...")
    try:
        fig, ax = generate_subpanel_1b(figure_params=figure_params)
        filepath = os.path.join(output_dir, f"{filename_prefix}1b.png")
        fig.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved: {filepath}")
    except Exception as e:
        print(f"  Error in subpanel 1b: {e}")
        import traceback
        traceback.print_exc()

    # Subpanel 1c: Emission spectra with filters (Function C)
    print("\nSubpanel 1c: Emission spectra with filters...")
    try:
        fig, ax = generate_subpanel_1c(figure_params=figure_params)
        filepath = os.path.join(output_dir, f"{filename_prefix}1c.png")
        fig.savefig(filepath, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {filepath}")
    except Exception as e:
        print(f"  Error in subpanel 1c: {e}")
        import traceback
        traceback.print_exc()

    # Subpanel 1d: Unmixing vectors bar chart (Function D)
    print("\nSubpanel 1d: Unmixing vectors bar chart...")
    try:
        fig, ax = generate_subpanel_1d(figure_params=figure_params)
        filepath = os.path.join(output_dir, f"{filename_prefix}1d.png")
        fig.savefig(filepath, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {filepath}")
    except Exception as e:
        print(f"  Error in subpanel 1d: {e}")
        import traceback
        traceback.print_exc()

    # Subpanel 1e: Predicted angle (Function E)
    print("\nSubpanel 1e: Predicted angle between FPs...")
    try:
        fig, ax = generate_subpanel_1e(figure_params=figure_params)
        filepath = os.path.join(output_dir, f"{filename_prefix}1e.png")
        fig.savefig(filepath, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {filepath}")
    except Exception as e:
        print(f"  Error in subpanel 1e: {e}")
        import traceback
        traceback.print_exc()

    # Shared classification state for 1g/1h/1i
    shared_data_h_i = None
    h_params = figure_params.get("1h", {})
    i_params = figure_params.get("1i", {})
    g_params = figure_params.get("1g", {})
    # Only compute if at least one of the panels is configured
    compute_shared = (
        (isinstance(h_params, dict) and bool(h_params))
        or (isinstance(i_params, dict) and bool(i_params))
    )

    if compute_shared:
        try:
            from src.figure2 import _compute_fig2_classifications

            use_cached = True
            if isinstance(h_params, dict) and "use_cached_shared_data" in h_params:
                use_cached = bool(h_params.get("use_cached_shared_data"))
            elif isinstance(i_params, dict) and "use_cached_shared_data" in i_params:
                use_cached = bool(i_params.get("use_cached_shared_data"))

            data_dir_shared = None
            single_fp_data_dir_shared = None
            row_dict_shared = None

            if isinstance(h_params, dict) and h_params.get("data_dir"):
                data_dir_shared = h_params.get("data_dir")
            elif isinstance(i_params, dict):
                data_dir_shared = i_params.get("data_dir")

            if isinstance(h_params, dict) and h_params.get("single_fp_data_dir"):
                single_fp_data_dir_shared = h_params.get("single_fp_data_dir")
            elif isinstance(i_params, dict):
                single_fp_data_dir_shared = i_params.get("single_fp_data_dir")

            if isinstance(h_params, dict) and h_params.get("row_dict") is not None:
                row_dict_shared = h_params.get("row_dict")
            elif isinstance(i_params, dict) and i_params.get("row_dict") is not None:
                row_dict_shared = i_params.get("row_dict")

            cache_path = _build_hi_shared_data_cache_path(output_dir, h_params, i_params)

            if use_cached and os.path.exists(cache_path):
                try:
                    with open(cache_path, "rb") as f:
                        shared_data_h_i = pickle.load(f)
                    print(f"  Loaded cached shared_data for 1h/1i: {cache_path}")
                except Exception as cache_load_error:
                    print(f"  Warning: failed loading shared_data cache ({cache_load_error}), recomputing...")
                    shared_data_h_i = _compute_fig2_classifications(
                        row_dict=row_dict_shared,
                        data_dir=data_dir_shared,
                        single_fp_data_dir=single_fp_data_dir_shared,
                    )
                    if use_cached:
                        with open(cache_path, "wb") as f:
                            pickle.dump(shared_data_h_i, f)
                        print(f"  Saved shared_data cache for 1h/1i: {cache_path}")
            else:
                shared_data_h_i = _compute_fig2_classifications(
                    row_dict=row_dict_shared,
                    data_dir=data_dir_shared,
                    single_fp_data_dir=single_fp_data_dir_shared,
                )
                if use_cached:
                    with open(cache_path, "wb") as f:
                        pickle.dump(shared_data_h_i, f)
                    print(f"  Saved shared_data cache for 1h/1i: {cache_path}")

            # Limit points to axis maxima for both H and I plots.
            if isinstance(shared_data_h_i, dict):
                try:
                    max_value = float(shared_data_h_i.get("max_value", 3000.0))
                    ch1 = np.asarray(shared_data_h_i.get("ch1_valid", []), dtype=float)
                    ch2 = np.asarray(shared_data_h_i.get("ch2_valid", []), dtype=float)
                    ch3 = np.asarray(shared_data_h_i.get("ch3_valid", []), dtype=float)
                    labels = np.asarray(shared_data_h_i.get("pixel_labels", []), dtype=object)

                    if len(ch1) == len(ch2) == len(ch3) == len(labels) and len(ch1) > 0:
                        keep_mask = (
                            (ch1 >= 0) & (ch1 <= max_value)
                            & (ch2 >= 0) & (ch2 <= max_value)
                            & (ch3 >= 0) & (ch3 <= max_value)
                        )
                        shared_data_h_i["ch1_valid"] = ch1[keep_mask]
                        shared_data_h_i["ch2_valid"] = ch2[keep_mask]
                        shared_data_h_i["ch3_valid"] = ch3[keep_mask]
                        shared_data_h_i["pixel_labels"] = labels[keep_mask]
                except Exception as limit_error:
                    print(f"  Warning: failed to apply axis-max filtering to shared_data: {limit_error}")

            # Optional: balanced quadrant selection to avoid uneven apparent densities.
            if isinstance(shared_data_h_i, dict):
                try:
                    apply_balanced = False
                    if isinstance(h_params, dict):
                        apply_balanced = bool(h_params.get("apply_balanced_pixel_selection", False))
                    if not apply_balanced and isinstance(i_params, dict):
                        apply_balanced = bool(i_params.get("apply_balanced_pixel_selection", False))

                    if apply_balanced:
                        selection_params = h_params if isinstance(h_params, dict) and h_params else i_params
                        shared_data_h_i = _subselect_balanced_pixels(shared_data_h_i, selection_params)
                        print(f"  Balanced pixel selection applied: kept {shared_data_h_i.get('balanced_selected_count', 'unknown')} points")
                except Exception as sel_error:
                    print(f"  Warning: failed to apply balanced pixel selection: {sel_error}")
        except Exception as e:
            print(f"  Warning: failed to compute shared_data for 1h/1i: {e}")
            shared_data_h_i = None

    # Subpanel 1g: 2-channel 2D scatterplot (Function G)
    print("\nSubpanel 1g: 2-channel 2D scatterplot with vectors + cones...")
    try:
        params_1g = dict(figure_params.get("1g", {}))
        if shared_data_h_i is not None:
            params_1g["shared_data"] = shared_data_h_i
        fig, ax = generate_subpanel_1g(params_dict=params_1g, figure_params=figure_params)
        filepath = os.path.join(output_dir, f"{filename_prefix}1g.png")
        fig.savefig(filepath, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {filepath}")
    except Exception as e:
        print(f"  Error in subpanel 1g: {e}")
        import traceback
        traceback.print_exc()

    # Subpanel 1h: 3-channel 3D scatterplot (Function H)
    print("\nSubpanel 1h: 3-channel 3D scatterplot with classification cones...")
    try:
        params_1h = dict(figure_params.get("1h", {}))
        if shared_data_h_i is not None:
            params_1h["shared_data"] = shared_data_h_i
        fig, ax = generate_subpanel_1h(params_dict=params_1h, figure_params=figure_params)
        filepath = os.path.join(output_dir, f"{filename_prefix}1h.png")
        fig.savefig(filepath, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {filepath}")
    except Exception as e:
        print(f"  Error in subpanel 1h: {e}")
        import traceback
        traceback.print_exc()

    # Subpanel 1i: 3-channel triangle projection (Function I)
    print("\nSubpanel 1i: 3-channel triangle projection...")
    try:
        params_1i = dict(figure_params.get("1i", {}))
        if shared_data_h_i is not None:
            params_1i["shared_data"] = shared_data_h_i
        fig, ax = generate_subpanel_1i(params_dict=params_1i, figure_params=figure_params)
        filepath = os.path.join(output_dir, f"{filename_prefix}1i.png")
        fig.savefig(filepath, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {filepath}")
    except Exception as e:
        print(f"  Error in subpanel 1i: {e}")
        import traceback
        traceback.print_exc()

    # Subpanel 1j: angle histogram (Function J)
    print("\nSubpanel 1j: balanced angle histogram...")
    try:
        params_1j = dict(figure_params.get("1j", {}))
        if shared_data_h_i is not None:
            params_1j["shared_data"] = shared_data_h_i
        fig, ax = generate_subpanel_1j(params_dict=params_1j, figure_params=figure_params)
        filepath = os.path.join(output_dir, f"{filename_prefix}1j.png")
        fig.savefig(filepath, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {filepath}")
    except Exception as e:
        print(f"  Error in subpanel 1j: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*60)
    print("New Figure 1 generation complete!")
    print("="*60)


if __name__ == "__main__":
    # Default behavior: save both presentation-sized and manuscript-sized panels,
    # each with a filename prefix.
    output_dir_master = fig_cfg.figure_1_output_dir
    os.makedirs(output_dir_master, exist_ok=True)

    # Overwrite the log file on each run.
    log_path = os.path.join(output_dir_master, "new_figure_1_run.log")
    with open(log_path, "w", encoding="utf-8") as f_log:
        original_stdout = sys.stdout
        original_stderr = sys.stderr
        sys.stdout = _TeeStream(original_stdout, f_log)
        sys.stderr = _TeeStream(original_stderr, f_log)
        try:
            generate_figure_1(
                figure_params=fig_cfg.figure_1_params_presentation,
                output_dir=fig_cfg.figure_1_output_dir,
                filename_prefix="presentation_",
            )
            generate_figure_1(
                figure_params=fig_cfg.figure_1_params_manuscript,
                output_dir=fig_cfg.figure_1_output_dir,
                filename_prefix="manuscript_",
            )
        finally:
            sys.stdout = original_stdout
            sys.stderr = original_stderr

