"""Part labels for objects without annotations."""
import numpy as np


def axis_part_labels(points, wide_ratio=1.6, n_bins=24):
    """Handle / head split of an elongated tool along its principal axis.
    A slice is `head` when its cross-section is wider than wide_ratio x the median slice width, else `handle`.
    Returns labels (M,) with 0 = handle, 1 = head."""
    p = np.asarray(points, dtype=np.float64)
    c = p - p.mean(0)
    axes = np.linalg.svd(c, full_matrices=False)[2]
    t, r = c @ axes[0], c @ axes[1:].T
    edges = np.linspace(t.min(), t.max() + 1e-9, n_bins + 1)
    b = np.clip(np.digitize(t, edges) - 1, 0, n_bins - 1)
    width = np.array([np.ptp(r[b == i], 0).max() if (b == i).sum() > 3 else 0.0 for i in range(n_bins)])
    wide = width > wide_ratio * np.median(width[width > 0])
    return wide[b].astype(int)
