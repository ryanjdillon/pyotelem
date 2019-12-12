"""
This module sets configuration values that can be used between all plotting
modules for consistency.
"""
import matplotlib.pyplot as _plt
import seaborn as _seaborn

# Use specified style (e.g. 'ggplot')
_plt.style.use("seaborn-whitegrid")

# Use specified color palette
_colors = _seaborn.color_palette()

# Global axis properties
_linewidth = 0.5
