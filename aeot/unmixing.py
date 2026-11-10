"""Helper functions for the AEOT practical on pixel unmixing.

Students do not need to read or modify this file. Use it in a notebook with::

    from aeot.unmixing import *
    img, X, ref = load_data(DATA_DIR, LANDSAT, REFERENCE)
"""
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from matplotlib.patches import Rectangle
from scipy.optimize import nnls

__all__ = [
    "BANDS", "COLORS", "LOG",
    "load_data", "to_img", "composite", "save_tif",
    "unmix", "select", "show_boxes",
    "validate", "log", "show_log",
    "check_todo1", "check_todo2", "check_todo3", "check_todo5", "check_todo6",
]

BANDS = ["B2 Blue", "B3 Green", "B4 Red", "B5 NIR", "B6 SWIR1", "B7 SWIR2"]
COLORS = {"Vegetation": "tab:green", "Soil": "tab:orange", "Impervious": "tab:red", "Water": "tab:blue"}
LOG = []          # experiment log, filled by log()

_state = {}       # image, grid size and raster profile, set by load_data()


# ----------------------------------------------------------------------------- data
def load_data(data_dir, landsat, reference):
    """Read the merged Landsat image (bands B2-B7) and the reference impervious fractions.

    Returns
    -------
    img : array (bands, rows, cols)   the image as a cube
    X   : array (pixels, bands)       one row per pixel, one column per band
    ref : array (pixels,)             reference impervious fraction per pixel (NaN = no data)
    """
    with rasterio.open(os.path.join(data_dir, landsat)) as src:
        img = src.read().astype(float)
        profile = src.profile
    nb, H, W = img.shape
    if nb != len(BANDS):
        raise ValueError(f"Expected {len(BANDS)} bands (B2-B7, in that order) in {landsat}, found {nb}")
    X = img.reshape(nb, -1).T
    with rasterio.open(os.path.join(data_dir, reference)) as src:
        ref = src.read(1).astype(float).ravel()
        if src.nodata is not None:
            ref[ref == src.nodata] = np.nan
    if ref.size != H * W:
        raise ValueError(f"The reference ({ref.size} pixels) does not match the image ({H} x {W} pixels)")
    _state.update(img=img, H=H, W=W, profile=profile)
    print(f"Image: {nb} bands, {H} x {W} = {len(X)} pixels;  reference: {np.isfinite(ref).sum()} valid pixels")
    return img, X, ref


def _need_data():
    if not _state:
        raise RuntimeError("Run load_data(...) first.")


def to_img(v):
    """Pixel vector (pixels,) -> 2D map (rows, cols)."""
    _need_data()
    return np.asarray(v, float).reshape(_state["H"], _state["W"])


def composite(r, g, b, gain=1 / 0.3):
    """RGB composite of bands r, g, b (0 = B2 ... 5 = B7), stretched for display."""
    _need_data()
    img = _state["img"]
    return np.clip(np.dstack([img[r], img[g], img[b]]) * gain, 0, 1)


def save_tif(stack2d, name):
    """Save a list of 2D maps as a multi-band GeoTIFF with the georeference of the Landsat image."""
    _need_data()
    p = _state["profile"].copy()
    p.update(count=len(stack2d), dtype="float32", nodata=-9999)
    with rasterio.open(name, "w", **p) as dst:
        dst.write(np.nan_to_num(np.array(stack2d), nan=-9999).astype("float32"))
    print("saved", name)


# ----------------------------------------------------------------------------- unmixing
def unmix(data, E, mode="fcls", delta=1e3):
    """Linear unmixing of every pixel.

    data : (pixels, bands);  E : (end-members, bands)
    mode : 'unconstrained' | 'sum_to_one' | 'fcls' (sum-to-one + non-negative)
    Returns the fractions (pixels, end-members) and the residual RMSE per pixel.
    """
    if mode == "unconstrained":
        F = np.linalg.lstsq(E.T, data.T, rcond=None)[0].T
    elif mode == "sum_to_one":
        A = np.vstack([E.T, delta * np.ones(len(E))])
        F = np.linalg.lstsq(A, np.vstack([data.T, np.full(len(data), delta)]), rcond=None)[0].T
    elif mode == "fcls":
        A = np.vstack([E.T, delta * np.ones(len(E))])
        F = np.array([nnls(A, np.append(x, delta))[0] for x in data])
    else:
        raise ValueError("mode must be 'unconstrained', 'sum_to_one' or 'fcls'")
    resid = np.sqrt(np.mean((F @ E - data) ** 2, axis=1))
    return F, resid


def select(boxes, scores):
    """Boolean mask per end-member: pixels whose (PC1, PC2) scores fall inside its box."""
    return {k: (scores[:, 0] > a) & (scores[:, 0] < b) & (scores[:, 1] > c) & (scores[:, 1] < d)
            for k, ((a, b), (c, d)) in boxes.items()}


def show_boxes(boxes, sel, E, names, P):
    """Plot the end-member boxes in the PC1-PC2 space, the selected pixels and the end-member spectra."""
    _need_data()
    H, W = _state["H"], _state["W"]
    fig, ax = plt.subplots(1, 3, figsize=(20, 6))
    ax[0].hexbin(P[:, 0], P[:, 1], gridsize=150, bins="log", cmap="Greys")
    lo, hi = P[:, :2].min(0), P[:, :2].max(0)
    for k, ((a, b), (c, d)) in boxes.items():
        a, b, c, d = max(a, lo[0]), min(b, hi[0]), max(c, lo[1]), min(d, hi[1])
        ax[0].add_patch(Rectangle((a, c), b - a, d - c, fill=False, ec=COLORS.get(k, "k"), lw=2, label=k))
    ax[0].set_xlabel("PC1"); ax[0].set_ylabel("PC2"); ax[0].legend(); ax[0].set_title("End-member boxes")
    ax[1].imshow(composite(3, 2, 1) * 0.6)
    for k, m in sel.items():
        r, c = np.unravel_index(np.where(m)[0], (H, W))
        ax[1].scatter(c, r, s=3, color=COLORS.get(k, "k"), label=k)
    ax[1].legend(markerscale=5); ax[1].set_title("Selected pixels")
    for k, e in zip(names, E):
        ax[2].plot(e, color=COLORS.get(k, "k"), lw=2, label=k)
    ax[2].set_xticks(range(len(BANDS))); ax[2].set_xticklabels(BANDS, rotation=30)
    ax[2].legend(); ax[2].set_title("End-member spectra")
    plt.show()


# ----------------------------------------------------------------------------- validation & log
_metrics = {"fn": None}   # the students' metrics function, registered by check_todo5()


def validate(pred, reference, title="", mask=None, plot=True):
    """Error metrics (computed with YOUR metrics function from TODO 5) + density scatterplot."""
    if _metrics["fn"] is None:
        raise RuntimeError("Complete TODO 5 and run check_todo5(metrics) first.")
    ok = np.isfinite(pred) & np.isfinite(reference)
    if mask is not None:
        ok &= mask
    m = {"n": int(ok.sum()), **_metrics["fn"](pred[ok], reference[ok])}
    if plot:
        fig, ax = plt.subplots(figsize=(4.5, 4.5))
        ax.hexbin(reference[ok], pred[ok], gridsize=40, bins="log", cmap="Blues", extent=(0, 1, 0, 1))
        ax.plot([0, 1], [0, 1], "k-", lw=1)
        ax.set_xlabel("Reference fraction"); ax.set_ylabel("Estimated fraction"); ax.set_title(title, fontsize=10)
        ax.text(.03, .97, "\n".join(f"{k} = {v:.3f}" if k != "n" else f"n = {v}" for k, v in m.items()),
                va="top", transform=ax.transAxes, fontsize=9, bbox=dict(fc="w", alpha=.8))
        plt.show()
    return m


def log(experiment, m, note=""):
    """Store a result in the experiment log."""
    LOG.append({"experiment": experiment, **{k: m.get(k, np.nan) for k in ("n", "MAE", "RMSE", "Bias", "r")},
                "note": note})
    print(f"logged: {experiment}  MAE={m['MAE']:.3f}  RMSE={m.get('RMSE', np.nan):.3f}  bias={m.get('Bias', np.nan):+.3f}")


def show_log():
    """Show the experiment log as a table."""
    from IPython.display import display
    display(pd.DataFrame(LOG).style.format({"MAE": "{:.3f}", "RMSE": "{:.3f}", "Bias": "{:+.3f}", "r": "{:.2f}"}))


# ----------------------------------------------------------------------------- checks of the TODOs
def _ok(n):
    print(f"✓ TODO {n} correct")


def check_todo1(Xn, X):
    """Brightness normalisation: every pixel has a mean of 100 and keeps its spectral shape."""
    assert isinstance(Xn, np.ndarray), "TODO 1: Xn should be a NumPy array"
    assert Xn.shape == X.shape, f"TODO 1: Xn should have the same shape as X {X.shape}, not {Xn.shape}"
    assert np.allclose(Xn.mean(axis=1), 100), "TODO 1: not yet correct, every pixel should have a mean of 100"
    assert np.allclose(Xn / Xn.sum(1, keepdims=True), X / X.sum(1, keepdims=True)), \
        "TODO 1: not yet correct, the ratios between the bands of a pixel should not change"
    _ok(1)


def check_todo2(P, Xn):
    """PCA: P holds the scores of the first 3 principal components of Xn."""
    from sklearn.decomposition import PCA
    assert isinstance(P, np.ndarray), "TODO 2: P should be a NumPy array (use .transform)"
    assert P.shape == (len(Xn), 3), f"TODO 2: P should have shape ({len(Xn)}, 3), not {P.shape}"
    expected = PCA(n_components=3).fit(Xn).transform(Xn)
    assert np.allclose(np.abs(P), np.abs(expected), atol=1e-6), \
        "TODO 2: not yet correct, did you fit AND transform the brightness-normalised data Xn?"
    _ok(2)


def check_todo3(boxes):
    """End-member boxes are complete and well formed."""
    assert isinstance(boxes, dict) and len(boxes) >= 3, "TODO 3: define at least 3 end-members in BOXES"
    for k, v in boxes.items():
        flat = sum(v, ())
        assert None not in flat and Ellipsis not in flat, f"TODO 3: fill in the thresholds for {k}"
        (a, b), (c, d) = v
        assert a < b and c < d, f"TODO 3: for {k}, each minimum should be smaller than its maximum"
    assert "Impervious" in boxes, "TODO 3: one end-member must be called 'Impervious'"
    _ok(3)


def check_todo5(metrics):
    """Test the students' metrics function and register it for validate()."""
    t = metrics(np.array([.2, .5, .9]), np.array([.0, .5, 1.]))
    assert isinstance(t, dict) and {"MAE", "RMSE", "Bias", "r"} <= set(t), \
        "TODO 5: metrics should return a dict with MAE, RMSE, Bias and r"
    assert all(isinstance(v, (float, np.floating)) for v in t.values()), "TODO 5: fill in every metric"
    for k, exp in {"MAE": 0.1, "RMSE": 0.1291, "Bias": 0.0333, "r": 0.9966}.items():
        assert abs(t[k] - exp) < 1e-3, f"TODO 5: {k} is not yet correct"
    _metrics["fn"] = metrics
    _ok(5)


def check_todo6(pred_mlr, X):
    """MLR prediction for every pixel."""
    assert isinstance(pred_mlr, np.ndarray) and pred_mlr.shape == (len(X),), \
        f"TODO 6: pred_mlr should have one value per pixel, shape ({len(X)},)"
    _ok(6)
