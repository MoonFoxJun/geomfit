"""Data preprocessing: PCA coordinate rotation (decorrelating strongly
correlated dimensions).

Background
----------
Tensor-product bases implicitly assume that data lie on a Cartesian product
domain (a hypercube) with orthogonal coordinates. When data are concentrated
along an oblique direction (e.g. a narrow band y ≈ x):

- basis functions become almost linearly dependent on the data, which makes
  the Gram matrix ill-conditioned and the coefficients huge;
- most basis functions waste degrees of freedom on regions without data.

PCA rotates the coordinates onto the orthogonal principal axes of the data
variance, making the principal components mutually uncorrelated:

- after the rotation, near-collinearity of the basis columns disappears;
- dropping near-zero-variance components (n_components < d) reduces dimension.
"""

from typing import Dict, Any, Optional, Tuple
import numpy as np

from ..core.data_container import MultiDimData


def pca_rotate(data: MultiDimData,
               n_components: Optional[int] = None,
               whiten: bool = False) -> Tuple[MultiDimData, Dict[str, Any]]:
    """
    Rotate (and optionally truncate or whiten) the coordinates of data via PCA.

    The rotation is obtained from the SVD of the centered coordinate matrix,
    Xc = U S Vᵀ, so that the columns of V are the principal directions.

    Parameters
    ----------
    data : MultiDimData
        Input data.
    n_components : int, optional
        Number of principal components to keep (default: keep all).
    whiten : bool, default=False
        If True, scale each principal component by its singular value so that
        the rotated axes have unit variance.

    Returns
    -------
    rotated : MultiDimData
        Rotated data with dimension indices 0..k-1, optionally truncated or
        whitened.
    info : Dict
        Dictionary with "components" (V, the d x k rotation matrix), "mean"
        (centering vector), "singular_values", "explained_variance_ratio",
        and "whiten". Pass it to pca_transform / pca_rotate_back to process
        new data.
    """
    X = data.get_coordinate_matrix()          # (n, d)
    n, d = X.shape
    mean = X.mean(axis=0)
    Xc = X - mean

    # SVD: Xc = U S Vᵀ; the columns of V are the eigenvectors of the
    # covariance matrix (the principal directions)
    _, s, Vt = np.linalg.svd(Xc, full_matrices=False)

    if n_components is None:
        k = d
    else:
        k = int(n_components)
        k = min(max(k, 1), d)

    V = Vt[:k].T                             # (d, k)
    Y = Xc @ V
    if whiten:
        Y = Y / np.maximum(s[:k], 1e-12)

    total_var = float(np.sum(s ** 2))
    ratio = (s ** 2) / total_var if total_var > 0 else np.zeros_like(s)

    rotated = MultiDimData({i: Y[:, i] for i in range(k)})
    rotated.values = data.values             # Target values follow the data points and are unchanged by the coordinate transform

    info = {
        "components": V,
        "mean": mean,
        "singular_values": s,
        "explained_variance_ratio": ratio,
        "whiten": whiten,
    }
    return rotated, info


def pca_transform(data: MultiDimData, info: Dict[str, Any]) -> MultiDimData:
    """
    Transform new data with PCA information fitted by pca_rotate.

    New data (test set or grid) must use the same rotation fitted on the
    training data, otherwise the coordinate systems are inconsistent.
    """
    X = data.get_coordinate_matrix()
    V = info["components"]
    mean = info["mean"]
    k = V.shape[1]

    Y = (X - mean) @ V
    if info.get("whiten"):
        s = info["singular_values"]
        Y = Y / np.maximum(s[:k], 1e-12)

    rotated = MultiDimData({i: Y[:, i] for i in range(k)})
    rotated.values = data.values
    return rotated


def pca_rotate_back(point: Dict[int, float], info: Dict[str, Any]) -> Dict[int, float]:
    """
    Map a single point from the rotated coordinates back to the original
    coordinates (e.g. for interpretation or plotting).

    Parameters
    ----------
    point : Dict[int, float]
        Point in the rotated coordinates (keys 0..k-1).
    info : Dict
        Information returned by pca_rotate.

    Returns
    -------
    Dict[int, float]
        Point in the original coordinates.
    """
    k = len(point)
    y = np.array([point[i] for i in sorted(point.keys())], dtype=float)
    V = info["components"]
    x = y @ V.T + info["mean"]
    return {i: float(xi) for i, xi in enumerate(x)}
