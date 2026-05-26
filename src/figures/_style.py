"""
Shared matplotlib styling for the Nature-spec figure scripts.

Implements the Springer Nature "Guide to Preparing Final Artwork" requirements:

- vector PDF backend
- sans-serif Helvetica / Arial, 6-7 pt body
- RGB colour mode
- TrueType embedded fonts (``pdf.fonttype = 42``) so reviewers can edit text
- column widths: 88 mm (single) = 3.46 in, 180 mm (double) = 7.09 in

Call :func:`apply_nature_style` once at the top of each figure module before
creating a Figure. Use :data:`COL_SINGLE_IN` / :data:`COL_DOUBLE_IN` for
``figsize``.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("pdf")
import matplotlib.pyplot as plt

# Column widths in inches (88 mm and 180 mm at 25.4 mm/in).
COL_SINGLE_IN = 88.0 / 25.4
COL_DOUBLE_IN = 180.0 / 25.4

_NATURE_RC = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 6,
    "axes.labelsize": 7,
    "axes.titlesize": 7,
    "axes.linewidth": 0.5,
    "xtick.labelsize": 6,
    "ytick.labelsize": 6,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    "xtick.major.size": 2.0,
    "ytick.major.size": 2.0,
    "legend.fontsize": 6,
    "legend.frameon": False,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.format": "pdf",
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
    "pdf.fonttype": 42,  # TrueType: text remains editable
    "ps.fonttype": 42,
    "axes.spines.top": False,
    "axes.spines.right": False,
}


def apply_nature_style() -> None:
    """Install the Nature rcParams. Idempotent."""
    plt.rcParams.update(_NATURE_RC)


__all__ = ["COL_DOUBLE_IN", "COL_SINGLE_IN", "apply_nature_style"]
