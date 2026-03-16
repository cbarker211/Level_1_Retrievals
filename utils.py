"""
Utilities used by example notebooks
"""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np


def plot_image(
    image: np.ndarray, factor: float = 1.0, clip_range: tuple[float, float] | None = None, **kwargs: Any
) -> None:
    """Utility function for plotting RGB images."""
    _, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 8))
    if clip_range is not None:
        im = ax.imshow(np.clip(image * factor, *clip_range), cmap='jet', **kwargs)
    else:
        im = ax.imshow(image * factor, cmap='jet', **kwargs)
    cbar_ax = _.add_axes([0.91,0.15,0.018,0.7])
    cb = _.colorbar(im, cax=cbar_ax, label='1e-3 mol/m2')
    ax.set_xticks([])
    ax.set_yticks([])
    cb.ax.tick_params(direction='in')
