import jax
import jax.numpy as jnp

import numpy as np
import matplotlib.pyplot as plt
import imageio.v2 as imageio
from pathlib import Path
from matplotlib.colors import Normalize

# ---------------------------
# Bilinear sampler (JAX)
# ---------------------------
def bilinear_sample(img, x, y):
    """
    Args:
        img: (H, W)
        x, y: same shape, in pixel coordinates [0, W-1], [0, H-1]
    Returns: 
        sampled values, same shape
    """
    H, W = img.shape

    # integer corners
    x0 = jnp.floor(x).astype(jnp.int32)
    y0 = jnp.floor(y).astype(jnp.int32)
    x1 = x0 + 1
    y1 = y0 + 1

    # clamp to image bounds
    x0 = jnp.clip(x0, 0, W - 1)
    x1 = jnp.clip(x1, 0, W - 1)
    y0 = jnp.clip(y0, 0, H - 1)
    y1 = jnp.clip(y1, 0, H - 1)

    # gather pixel values
    Ia = img[y0, x0]
    Ib = img[y0, x1]
    Ic = img[y1, x0]
    Id = img[y1, x1]

    # fractional parts
    fx = x - x0.astype(x.dtype)
    fy = y - y0.astype(y.dtype)

    # bilinear weights
    wa = (1.0 - fx) * (1.0 - fy)
    wb = fx * (1.0 - fy)
    wc = (1.0 - fx) * fy
    wd = fx * fy

    return wa * Ia + wb * Ib + wc * Ic + wd * Id

def bilinear_sample_zero(
    img: jnp.ndarray,
    x: jnp.ndarray,
    y: jnp.ndarray,
) -> jnp.ndarray:
    """
    Differentiable bilinear sampler for a 2D image, with **zero padding** outside the image.

    This function samples `img` at (x, y) locations using bilinear interpolation.
    If a sample point lies outside the valid image domain, its value is set to 0.0
    (i.e., "air/background"), which matches typical CT/Radon conventions and avoids
    boundary-clamping artifacts.

    Parameters
    ----------
    img : jnp.ndarray
        2D image array of shape (H, W).
    x : jnp.ndarray
        Sample x-coordinates in **pixel index units**, where valid range is [0, W-1].
        Can be any shape; must match `y` shape (e.g., (n_det, n_samples)).
    y : jnp.ndarray
        Sample y-coordinates in **pixel index units**, where valid range is [0, H-1].
        Same shape as `x`.

    Returns
    -------
    jnp.ndarray
        Interpolated values at (x, y), same shape as `x` and `y`.

    Notes
    -----
    - The function is fully differentiable w.r.t. `img` and (x, y) in JAX.
    - We clip indices only for safe array indexing, but we apply a boolean mask
      so that out-of-bounds samples contribute exactly 0.0 (rather than sampling
      the nearest edge pixel).
    """

    # Image height and width.
    H, W = img.shape

    # Compute the integer pixel corner indices around each (x, y) sample location.
    # (x0, y0) is the "lower-left" corner; (x1, y1) is the "upper-right" corner.
    x0 = jnp.floor(x).astype(jnp.int32)
    y0 = jnp.floor(y).astype(jnp.int32)
    x1 = x0 + 1
    y1 = y0 + 1

    # Identify which sample points lie inside the valid image extent.
    # Points outside will be forced to 0.0 after interpolation.
    in_bounds = (x >= 0.0) & (x <= (W - 1)) & (y >= 0.0) & (y <= (H - 1))

    # Clip indices for safe gather (prevents indexing errors at borders).
    # IMPORTANT: clipping alone would create boundary artifacts (edge replication),
    # so we still apply `in_bounds` to zero-out samples that were actually out-of-range.
    x0c = jnp.clip(x0, 0, W - 1)
    x1c = jnp.clip(x1, 0, W - 1)
    y0c = jnp.clip(y0, 0, H - 1)
    y1c = jnp.clip(y1, 0, H - 1)

    # Gather image values at the four surrounding corners:
    # Ia: (x0, y0), Ib: (x1, y0), Ic: (x0, y1), Id: (x1, y1)
    Ia = img[y0c, x0c]
    Ib = img[y0c, x1c]
    Ic = img[y1c, x0c]
    Id = img[y1c, x1c]

    # Fractional offsets within the cell (in [0, 1] typically).
    # These determine interpolation weights.
    fx = x - x0.astype(x.dtype)
    fy = y - y0.astype(y.dtype)

    # Bilinear interpolation weights for each corner.
    wa = (1.0 - fx) * (1.0 - fy)
    wb = fx * (1.0 - fy)
    wc = (1.0 - fx) * fy
    wd = fx * fy

    # Weighted sum of the four corner values.
    val = wa * Ia + wb * Ib + wc * Ic + wd * Id

    # Enforce zero outside the image domain.
    # This makes rays that miss the object contribute zero attenuation (as in CT).
    return jnp.where(in_bounds, val, 0.0)

# ---------------------------
# Differentiable Radon
# ---------------------------
def radon_jax(image: jnp.ndarray, angles: jnp.ndarray, n_det: int | None = None, n_samples: int | None = None) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Differentiable parallel-beam Radon transform in JAX.
    Args:
        image:  (H, W) array (jnp.ndarray, float32/64)
        angles: (N_theta,) in *degrees*  (e.g. jnp.linspace(0, 180, N_theta, endpoint=False))
        n_det:  number of detector bins (default: max(H, W))
        n_samples: number of sample points along each ray (default: max(H, W))
    
    Returns:
        sinogram: (n_det, N_theta)
        s_vals:   (n_det,) detector coordinates (in pixel units)
    """
    H, W = image.shape
    if n_det is None:
        n_det = max(H, W)
    if n_samples is None:
        n_samples = max(H, W)
    n_samples = n_det
    angles = jnp.deg2rad(angles)

    # coordinate centre (pixel coordinates)
    cx = (W - 1) / 2.0
    cy = (H - 1) / 2.0

    # max radius from centre to corner
    R = jnp.sqrt(cx**2 + cy**2)

    # detector positions (signed distance from centre)
    s_vals = jnp.linspace(-R, R, n_det)

    # samples along each ray (line parameter l)
    l_vals = jnp.linspace(-R, R, n_samples)
    dl = (l_vals[-1] - l_vals[0]) / (n_samples - 1)

    # grids: shape (n_det, n_samples)
    s_grid, l_grid = jnp.meshgrid(s_vals, l_vals, indexing="ij")

    def proj_at_angle(theta: jnp.ndarray) -> jnp.ndarray:
        c = jnp.cos(theta)
        s = jnp.sin(theta)

        # line parameterization in *image-centred* coordinates:
        # x' = s*c - l*s
        # y' = s*s + l*c
        x_prime = s_grid * c - l_grid * s
        y_prime = s_grid * s + l_grid * c

        # convert to pixel indices
        x_pix = x_prime + cx
        y_pix = y_prime + cy
        
        #samples = bilinear_sample(image, x_pix, y_pix)
        samples = bilinear_sample_zero(image, x_pix, y_pix)
        
        # approximate line integral: sum over l * dl
        return jnp.sum(samples, axis=-1) * dl   # (n_det,)

    # vectorize over angles
    sinogram = jax.vmap(proj_at_angle)(angles)   # (N_theta, n_det)
    sinogram = jnp.swapaxes(sinogram, 0, 1)      # (n_det, N_theta)

    return sinogram, s_vals

def save_sinogram_gif(sinograms: np.ndarray, out_path_sino: Path, times: np.ndarray, theta: np.ndarray, fps: int = 5):
    """Saves a GIF animation of sinograms over time.
    Args:
        sinograms: np.ndarray of shape (n_frames, n_s, n_theta)
        out_path_sino: Path to save the GIF
        times: total number of time steps
        theta: angle of the orientaion of the emitted rays. (N_theta,) in *degrees*  (e.g. jnp.linspace(0, 180, N_theta, endpoint=False))
        fps: frames per second
    Returns:
        None
    """
    frames_sino = []

    # Fix color scale over all times to avoid flicker
    vmin = float(sinograms.min())
    vmax = float(sinograms.max())
    norm = Normalize(vmin=vmin, vmax=vmax)

    fig_s, ax_s = plt.subplots(figsize=(4.8, 3.6), dpi=140)
    cb_s = None

    for k, t in enumerate(times):
        S = sinograms[k]  # shape (n_s, n_theta)

        ax_s.clear()
        im_s = ax_s.imshow(
            S,
            origin="lower",
            aspect="auto",
            extent=[theta[0], theta[-1], 0, S.shape[0]],
            cmap="viridis",
            norm=norm,
            interpolation="bilinear",
        )
        ax_s.set_xlabel("theta (degrees)")
        ax_s.set_ylabel("detector coordinate s (index)")
        ax_s.set_title(f"Sinogram at t = {t:0.3f}")

        if cb_s is None:
            cb_s = fig_s.colorbar(im_s, ax=ax_s, label="line integral of c")

        fig_s.tight_layout()

        # capture frame (same pattern as your concentration GIF)
        fig_s.canvas.draw()
        w, h = fig_s.canvas.get_width_height()
        buf = np.frombuffer(fig_s.canvas.buffer_rgba(), dtype=np.uint8)
        frame = buf.reshape(h, w, 4)[..., :3]
        frames_sino.append(frame.copy())

    plt.close(fig_s)

    imageio.mimsave(out_path_sino, frames_sino, fps=fps)
    print(f"Saved sinogram GIF -> {out_path_sino.resolve()}")

def sinogram_from_conc(t_hist_np: jnp.ndarray, ny: int, nx: int, n_theta: int = 180, n_det: int = 200, alpha: float = 1.):
    """
        Computes sinograms from the concentration field at all the time steps.
        Args:
            t_hist_np: Concentration data at each time step. shape (n_frames, s_len, n_theta)
            ny: number of nodes in y direction: 
            nx: number of nodes in x direction
            n_theta: n_theta angles sampled between from 0 and 180 degrees. (the source will be rotated n_theta times and projections are captured.)
            alpha: is the mass attenuation coefficient of iodine, converting concentration into linear X-ray attenuation.
        Returns:
            sinograms: Sinograms at all the time steps. shape (n_frames, s_len, n_theta)
    """
        
    # ----------------------
    # Sinogram parameters
    # ----------------------
    # Parallel-beam angles in [0, 180) degrees
    theta = jnp.linspace(0.0, 180.0, n_theta, endpoint=False)
    sinograms = []

    for k in range(t_hist_np.shape[0]):
        c_num   = t_hist_np[k, :].reshape(ny,nx) # shape (ny, nx)
        
        # Convert to attenuation (iodine-only)
        mu_iodine = alpha * c_num
        # --- Radon transform: p(theta, s, t) ---
        sino_t, s_vals = radon_jax(mu_iodine, angles=theta, n_det=n_det)
        sino_t = sino_t[::-1, ::-1]

        # sino_t: shape (s_len, n_theta)
        sinograms.append(sino_t)

    # Stack over time: 
    sinograms = jnp.stack(sinograms, axis=0)
    return sinograms
    