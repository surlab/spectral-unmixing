"""
DEPRECATED: use per-figure config modules.

This module remains as a compatibility shim while the codebase transitions to:
  - `src/figure_1_config.py`
  - `src/figure_2_config.py`
  - ...
"""

from src.figure_1_config import (  # noqa: F401
    figure_1_params_presentation as figure_1_params,
    figure_1_data_dir,
    figure_1_output_dir,
)
