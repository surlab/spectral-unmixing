"""
Figure 5 generation for spectral unmixing methods paper.

This module generates Figure 5, which shows spectral unmixing for many fluorophores
across multiple excitation wavelengths and emission filters.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from src import config as cfg
from src.figure1 import (
    load_2p_spectra,
    plot_2p_excitation_spectra,
    plot_1p_emission_spectra,
    apply_smoothing_to_spectrum,
    load_filter_transmission
)
from src.figure5 import load_figure5_2p_spectra

# ... existing code ...
