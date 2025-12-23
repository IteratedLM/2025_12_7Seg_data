#!/usr/bin/env python3
"""
Generate noisy 7-segment glyph images (28x28) and save as either HDF5 (.h5) or Zarr (.zarr),
plus optionally save an example PNG for a quick visual check.

Dependencies:
  pip install numpy scipy h5py zarr imageio

Usage examples:
  python noisy7seg.py --out digits_noisy --format h5   --n_per 100 --mu 0.03 --sigma 4.0 --rho 0.05 --nSub 10 --png
  python noisy7seg.py --out digits_noisy --format zarr --n_per 100 --mu 0.03 --sigma 4.0 --rho 0.05 --nSub 10 --png

Outputs:
  - For HDF5:
      digits_noisy.h5        with datasets: imgs (28,28,N), labels (N,)
      digits_noisy_base.h5   with base glyphs: imgs (28,28,128), labels (128,)
  - For Zarr:
      digits_noisy.zarr/ and digits_noisy_base.zarr/ with arrays: imgs, labels
  - Optional example PNG:
      {out}_example.png (scaled up)
"""

# translated with machine help from the python

from __future__ import annotations

import argparse
import os
import numpy as np
from scipy.ndimage import rotate as nd_rotate
import imageio.v2 as imageio


# -----------------------------------------------------------------------------
# 0) Build a 7-segment “perfect” 28×28 glyph (Julia-compatible: glyph_id in 1..128)
# -----------------------------------------------------------------------------
def make_7seg(
    glyph_id: int,
    *,
    size: int = 28,
    margin: int = 2,
    thickness: int = 4,
    zero_indexed: bool = False,
) -> np.ndarray:
    """
    Return (size,size) float32 image with {0,1} values.
    Julia loop uses glyph_id = 1..128 where mask = glyph_id-1 selects segments.
    """
    if zero_indexed:
        glyph_id += 1
    if not (1 <= glyph_id <= 128):
        raise ValueError("glyph_id must be in 1..128 (or 0..127 with zero_indexed=True).")

    s, m, t = size, margin, thickness
    h = (size - 2 * margin - 3 * thickness) // 2

    # Segment rectangles: [r0:r1), [c0:c1)
    segs = {
        "A": ((m + 0, m + t),                 (m + t,     s - m - t)),
        "F": ((m + t, m + t + h),             (m + 0,     m + t)),
        "B": ((m + t, m + t + h),             (s - m - t, s - m)),
        "G": ((m + t + h, m + 2 * t + h),     (m + t,     s - m - t)),
        "E": ((m + 2 * t + h, m + 2 * t + 2 * h), (m + 0,     m + t)),
        "C": ((m + 2 * t + h, m + 2 * t + 2 * h), (s - m - t, s - m)),
        "D": ((m + 2 * t + 2 * h, m + 3 * t + 2 * h), (m + t, s - m - t)),
    }
    seg_names = ["A", "B", "C", "D", "E", "F", "G"]

    mask = glyph_id - 1
    active = [seg_names[j] for j in range(7) if (mask >> j) & 1]

    img = np.zeros((size, size), dtype=np.float32)
    for seg in active:
        (r0, r1), (c0, c1) = segs[seg]
        img[r0:r1, c0:c1] = 1.0
    return img


# -----------------------------------------------------------------------------
# 1) Noise + warp + downsample
# -----------------------------------------------------------------------------
def transform28(
    img28: np.ndarray,
    n: int,
    *,
    mu: float = 0.05,
    rho: float = 0.05,
    sigma_x: float = 0.0,
    sigma_y: float = 0.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Pipeline:
      1. Upsample to (28n×28n) by pixel-repeat.
      2. Dim all 1.0→max(0,1-ε) with ε∼Exp(scale=mu).
      3. Rotate by θ∼N(0,rho) with bilinear interpolation, keep size.
      4. Translate by integer shifts sampled from N(0,sigma_x/y), rounded to int, zero-fill.
      5. Downsample back to 28×28 by averaging each n×n block.
    """
    if rng is None:
        rng = np.random.default_rng()

    img28 = np.asarray(img28, dtype=np.float32)
    if img28.shape != (28, 28):
        raise ValueError("img28 must have shape (28,28).")

    # (1) upsample by repeat
    big = np.repeat(np.repeat(img28, n, axis=0), n, axis=1).astype(np.float32)
    H, W = big.shape

    # (2) dimming (Exp with mean=mu in Julia => scale=mu here)
    eps = rng.exponential(scale=mu)
    b = max(0.0, 1.0 - float(eps))
    big = np.where(big == 1.0, np.float32(b), np.float32(0.0))

    # (3) rotate about center; scipy rotate uses degrees
    theta = rng.normal(loc=0.0, scale=rho)
    bigr = nd_rotate(
        big,
        angle=np.degrees(theta),
        reshape=False,
        order=1,  # bilinear
        mode="constant",
        cval=0.0,
        prefilter=False,
    ).astype(np.float32)

    # (4) integer translate
    shift_x = int(np.round(rng.normal(0.0, sigma_x)))
    shift_y = int(np.round(rng.normal(0.0, sigma_y)))

    bigt = np.zeros((H, W), dtype=np.float32)

    # destination ranges
    i1s = max(0, 0 + shift_y)
    i1e = min(H, H + shift_y)
    j1s = max(0, 0 + shift_x)
    j1e = min(W, W + shift_x)

    # source ranges
    i0s = max(0, 0 - shift_y)
    i0e = min(H, H - shift_y)
    j0s = max(0, 0 - shift_x)
    j0e = min(W, W - shift_x)

    if i1s < i1e and j1s < j1e:
        bigt[i1s:i1e, j1s:j1e] = bigr[i0s:i0e, j0s:j0e]

    # (5) block-average downsample
    out = bigt.reshape(28, n, 28, n).mean(axis=(1, 3)).astype(np.float32)
    out = np.nan_to_num(out, nan=0.0)
    np.clip(out, 0.0, 1.0, out=out)
    return out


# -----------------------------------------------------------------------------
# 2) PNG helpers
# -----------------------------------------------------------------------------
def _to_uint8(img: np.ndarray) -> np.ndarray:
    img = np.nan_to_num(img, nan=0.0)
    img = np.clip(img, 0.0, 1.0)
    return (img * 255.0 + 0.5).astype(np.uint8)


def save_png(filename: str, img28: np.ndarray, *, scale: int = 10) -> None:
    imgbig = np.repeat(np.repeat(img28, scale, axis=0), scale, axis=1)
    imageio.imwrite(filename, _to_uint8(imgbig))


# -----------------------------------------------------------------------------
# 3) Dataset saving: HDF5 or Zarr
# -----------------------------------------------------------------------------
def _save_h5(path_stem: str, imgs: np.ndarray, labels: np.ndarray) -> str:
    import h5py  # local import so script works even if user only uses zarr

    out_path = path_stem + ".h5"
    with h5py.File(out_path, "w") as f:
        f.create_dataset("imgs", data=imgs, dtype="float32")
        f.create_dataset("labels", data=labels, dtype="int32")
    return out_path


def _save_zarr(path_stem: str, imgs: np.ndarray, labels: np.ndarray, *, chunk_imgs=(28, 28, 1024)) -> str:
    import zarr  # local import

    out_path = path_stem + ".zarr"
    # overwrite existing store
    if os.path.isdir(out_path):
        # zarr doesn't have a universal "rm -r" built-in; do a simple safe removal
        import shutil
        shutil.rmtree(out_path)

    root = zarr.open(out_path, mode="w")
    root.create_dataset("imgs", data=imgs, chunks=chunk_imgs, dtype="float32", overwrite=True)
    root.create_dataset("labels", data=labels, chunks=(min(labels.shape[0], 4096),), dtype="int32", overwrite=True)
    return out_path


def save_dataset(
    out_stem: str,
    *,
    fmt: str = "h5",
    n_per_glyph: int = 500,
    mu: float = 0.025,
    sigma: float = 6.0,
    rho: float = 0.05,
    nSub: int = 10,
    seed: int | None = None,
    save_example_png: bool = False,
    png_scale: int = 10,
) -> None:
    """
    Generates n_per_glyph noisy samples for each glyph_id 1..128.
    Saves both noisy dataset and base glyphs dataset in chosen format.
    Optionally saves an example PNG.
    """
    fmt = fmt.lower().strip()
    if fmt not in {"h5", "zarr"}:
        raise ValueError("fmt must be 'h5' or 'zarr'.")

    rng = np.random.default_rng(seed)

    total_glyphs = 128
    total = total_glyphs * n_per_glyph

    imgs = np.empty((28, 28, total), dtype=np.float32)
    labels = np.empty((total,), dtype=np.int32)

    base_imgs = np.empty((28, 28, total_glyphs), dtype=np.float32)
    base_labels = np.arange(1, total_glyphs + 1, dtype=np.int32)

    idx = 0
    example_img = None

    for glyph_id in range(1, total_glyphs + 1):
        base = make_7seg(glyph_id)
        base_imgs[:, :, glyph_id - 1] = base

        for _ in range(n_per_glyph):
            img = transform28(base, nSub, mu=mu, rho=rho, sigma_x=sigma, sigma_y=sigma, rng=rng)
            imgs[:, :, idx] = img
            labels[idx] = glyph_id

            if example_img is None and glyph_id == 1:
                example_img = img.copy()

            idx += 1

    # Save datasets
    if fmt == "h5":
        noisy_path = _save_h5(out_stem, imgs, labels)
        base_path = _save_h5(out_stem + "_base", base_imgs, base_labels)
    else:
        noisy_path = _save_zarr(out_stem, imgs, labels)
        base_path = _save_zarr(out_stem + "_base", base_imgs, base_labels, chunk_imgs=(28, 28, 128))

    print(f"Saved noisy dataset: {noisy_path}  (N={total})")
    print(f"Saved base dataset:  {base_path}  (N={total_glyphs})")

    # Optional example PNG
    if save_example_png:
        if example_img is None:
            example_img = imgs[:, :, 0]
        png_path = f"{out_stem}_example.png"
        save_png(png_path, example_img, scale=png_scale)
        print(f"Saved example PNG:  {png_path}")


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="digits_noisy", help="Output filename stem (no extension).")
    p.add_argument("--format", default="h5", choices=["h5", "zarr"], help="Save format.")
    p.add_argument("--n_per", type=int, default=100, help="Samples per glyph (128 glyphs total).")
    p.add_argument("--mu", type=float, default=0.03, help="Dimming mean for Exp(scale=mu).")
    p.add_argument("--sigma", type=float, default=4.0, help="Translation std (in upsampled pixels).")
    p.add_argument("--rho", type=float, default=0.05, help="Rotation std (radians).")
    p.add_argument("--nSub", type=int, default=10, help="Upsample factor (big image is 28*nSub).")
    p.add_argument("--seed", type=int, default=None, help="RNG seed (optional).")
    p.add_argument("--png", action="store_true", help="Also save an example PNG.")
    p.add_argument("--png_scale", type=int, default=10, help="Scale factor for example PNG.")
    args = p.parse_args()

    save_dataset(
        args.out,
        fmt=args.format,
        n_per_glyph=args.n_per,
        mu=args.mu,
        sigma=args.sigma,
        rho=args.rho,
        nSub=args.nSub,
        seed=args.seed,
        save_example_png=args.png,
        png_scale=args.png_scale,
    )


if __name__ == "__main__":
    main()
