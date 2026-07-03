#!/usr/bin/env python3
"""
Generate the time-series reconstruction figures for type-A nested
superstatistics.

The script:
1. Simulates a three-level hierarchy:
      u(t)  -> fast Ornstein-Uhlenbeck dynamics
      beta(t) -> intermediate residence process compatible with type A
      n(t), q(t)=1+2/n(t) -> slow residence process compatible with type A
2. Reconstructs beta(t) from the local variance of u(t).
3. Reconstructs the type-A parameter q(t) from local moments of beta(t).
4. Recovers the type-A distribution g(q) by inverse-residence weighting.
5. Produces the four publication figures and a combined two-by-two PDF.
6. Saves all numerical arrays required to reproduce the figures.

NumPy, SciPy, Matplotlib, and Pillow are required.

Example
-------
python generate_nested_timeseries_figures_typeA_refined.py

Optional arguments
------------------
python generate_nested_timeseries_figures_typeA_refined.py \
    --output-dir results \
    --seed 12345 \
    --save-full-timeseries
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageOps
from scipy.signal import lfilter
from scipy.special import digamma, gammaln, polygamma


@dataclass(frozen=True)
class SimulationConfig:
    """Numerical parameters of the hierarchical Langevin model."""

    beta0: float = 1.0
    tau_u: float = 1.0
    tau_beta: float = 50.0
    tau_q: float = 10000.0
    dt: float = 0.2
    n0: float = 20.0
    sigma_eta: float = 0.35
    n_min: float = 5.05
    total_time: float = 1000000.0
    seed: int = 12345

    # Reconstruction parameters
    q_window_blocks: int = 30
    q_window_stride: int = 5

    # Figure ranges
    autocorrelation_max_lag: float = 4000.0
    ratio_x_max: float = 12.0
    ratio_y_min: float = 0.94
    ratio_y_max: float = 1.06


def block_view(values: np.ndarray, block_size: int) -> np.ndarray:
    """Return a two-dimensional nonoverlapping block view."""

    if block_size <= 0:
        raise ValueError("block_size must be positive.")

    n_blocks = values.size // block_size
    if n_blocks == 0:
        raise ValueError("block_size exceeds the length of the input array.")

    trimmed = values[: n_blocks * block_size]
    return trimmed.reshape(n_blocks, block_size)


def local_flatness(signal: np.ndarray, block_size: int) -> float:
    """Average the flatness computed in nonoverlapping windows."""

    blocks = block_view(signal, block_size)
    centered = blocks - blocks.mean(axis=1, keepdims=True)
    second = np.mean(centered**2, axis=1)
    fourth = np.mean(centered**4, axis=1)

    valid = second > 0.0
    if not np.any(valid):
        raise RuntimeError("No window has a nonzero variance.")

    return float(np.mean(fourth[valid] / second[valid] ** 2))


def autocorrelation_fft(signal: np.ndarray, max_lag: int) -> np.ndarray:
    """
    Compute the unbiased normalized autocorrelation using an FFT.

    The result is normalized so that C(0)=1.
    """

    values = np.asarray(signal, dtype=float)
    values = values - values.mean()
    n = values.size

    if max_lag < 0:
        raise ValueError("max_lag must be nonnegative.")

    max_lag = min(max_lag, n - 1)
    fft_size = 1 << (2 * n - 1).bit_length()

    spectrum = np.fft.rfft(values, n=fft_size)
    covariance = np.fft.irfft(spectrum * np.conjugate(spectrum), n=fft_size)
    covariance = covariance[: max_lag + 1]

    normalization = np.arange(n, n - max_lag - 1, -1, dtype=float)
    covariance /= normalization

    if covariance[0] <= 0.0:
        raise RuntimeError("The signal variance is not positive.")

    return covariance / covariance[0]


def log_type_a_normalization(n: np.ndarray | float) -> np.ndarray:
    """Return log(Z_n) for the one-dimensional type-A q-exponential."""

    values = np.asarray(n, dtype=float)
    if np.any(values <= 1.0):
        raise ValueError("The type-A normalization is finite only for n>1.")

    return (
        0.5 * np.log(np.pi * values / 2.0)
        + gammaln((values - 1.0) / 2.0)
        - gammaln(values / 2.0)
    )


def build_n_residence_sampler(
    config: SimulationConfig,
    grid_size: int = 50000,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build an inverse-CDF sampler for rho_n(n) proportional to h(n) Z_n."""

    mu = np.log(config.n0) - 0.5 * config.sigma_eta**2
    y_min = np.log(config.n_min)
    y_max = max(mu + 8.0 * config.sigma_eta, y_min + 8.0 * config.sigma_eta)

    y_grid = np.linspace(y_min, y_max, grid_size)
    n_grid = np.exp(y_grid)

    # The parent log-normal density is Gaussian in y=ln n. The residence
    # density acquires the additional type-A normalization factor Z_n.
    log_density = (
        -0.5 * ((y_grid - mu) / config.sigma_eta) ** 2
        + log_type_a_normalization(n_grid)
    )
    density = np.exp(log_density - np.max(log_density))

    increments = 0.5 * (density[1:] + density[:-1]) * np.diff(y_grid)
    cdf = np.concatenate(([0.0], np.cumsum(increments)))
    cdf /= cdf[-1]

    return y_grid, cdf


def sample_n_residence(
    rng: np.random.Generator,
    size: int,
    y_grid: np.ndarray,
    cdf: np.ndarray,
) -> np.ndarray:
    """Draw n from the residence distribution rho_n(n)."""

    uniforms = rng.random(size=size)
    return np.exp(np.interp(uniforms, cdf, y_grid))


def type_a_parent_quadrature(
    config: SimulationConfig,
    grid_size: int = 20000,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return q nodes and normalized weights for the truncated parent h(n)."""

    mu = np.log(config.n0) - 0.5 * config.sigma_eta**2
    y_min = np.log(config.n_min)
    y_max = max(mu + 8.0 * config.sigma_eta, y_min + 8.0 * config.sigma_eta)

    y_grid = np.linspace(y_min, y_max, grid_size)
    n_grid = np.exp(y_grid)
    density = np.exp(-0.5 * ((y_grid - mu) / config.sigma_eta) ** 2)

    trapezoidal_weights = density.copy()
    trapezoidal_weights[0] *= 0.5
    trapezoidal_weights[-1] *= 0.5
    trapezoidal_weights /= np.sum(trapezoidal_weights)

    q_grid = 1.0 + 2.0 / n_grid
    return q_grid, trapezoidal_weights


def simulate_hierarchical_process(
    config: SimulationConfig,
) -> Dict[str, np.ndarray]:
    """Simulate u(t), beta(t), n(t), and q(t) in the type-A realization."""

    rng = np.random.default_rng(config.seed)

    steps_per_beta = int(round(config.tau_beta / config.dt))
    steps_per_q = int(round(config.tau_q / config.dt))

    if steps_per_beta < 1 or steps_per_q < 1:
        raise ValueError("Time scales must be larger than or equal to dt.")

    if steps_per_q % steps_per_beta != 0:
        raise ValueError(
            "For the piecewise-constant protocol, tau_q/tau_beta must be an integer."
        )

    n_steps_requested = int(config.total_time / config.dt)
    n_beta_blocks = n_steps_requested // steps_per_beta
    n_steps = n_beta_blocks * steps_per_beta
    n_q_blocks = int(np.ceil(n_steps / steps_per_q))

    time = config.dt * np.arange(n_steps)

    # Slow residence process: rho_n(n) proportional to h(n) Z_n.
    residence_y_grid, residence_cdf = build_n_residence_sampler(config)
    n_q_blocks_values = sample_n_residence(
        rng,
        n_q_blocks,
        residence_y_grid,
        residence_cdf,
    )

    beta_blocks_per_q_block = steps_per_q // steps_per_beta
    n_beta_blocks_values = np.repeat(
        n_q_blocks_values,
        beta_blocks_per_q_block,
    )[:n_beta_blocks]

    q_beta_blocks_values = 1.0 + 2.0 / n_beta_blocks_values

    # Intermediate residence process compatible with the type-A weight:
    # Gamma(shape=(n-1)/2, scale=2 beta0/n).
    shape = (n_beta_blocks_values - 1.0) / 2.0
    scale = 2.0 * config.beta0 / n_beta_blocks_values
    beta_beta_blocks_values = rng.gamma(shape=shape, scale=scale)

    beta_time = np.repeat(beta_beta_blocks_values, steps_per_beta)[:n_steps]
    n_time = np.repeat(n_beta_blocks_values, steps_per_beta)[:n_steps]
    q_time = np.repeat(q_beta_blocks_values, steps_per_beta)[:n_steps]

    # Fast process: exact discrete OU update at piecewise-constant beta.
    # scipy.signal.lfilter evaluates the AR(1) recurrence efficiently while
    # preserving the same stochastic update as the explicit loop.
    decay = np.exp(-config.dt / config.tau_u)
    local_variance = config.beta0 / (2.0 * beta_time)

    innovations = np.empty(n_steps, dtype=float)
    innovations[0] = np.sqrt(local_variance[0]) * rng.normal()
    innovations[1:] = (
        np.sqrt((1.0 - decay**2) * local_variance[:-1])
        * rng.normal(size=n_steps - 1)
    )
    u = lfilter([1.0], [1.0, -decay], innovations)

    return {
        "time": time,
        "u": u,
        "beta_time": beta_time,
        "n_time": n_time,
        "q_time": q_time,
        "beta_block_values": beta_beta_blocks_values,
        "n_block_values": n_beta_blocks_values,
        "q_block_values": q_beta_blocks_values,
    }


def infer_beta_scale(
    u: np.ndarray,
    config: SimulationConfig,
) -> Tuple[np.ndarray, np.ndarray, float, int]:
    """
    Determine T_beta from the local-Gaussian flatness criterion.

    A logarithmic grid is used for the displayed curve.  Around the first
    crossing of K_u=3, the search is refined at the resolution of one time
    step.  Because finite records make the flatness estimate noisy, the
    shortest window lying within an absolute tolerance of 0.01 below the
    Gaussian value is selected.  This conservative choice avoids mixing
    neighboring beta-residence intervals.
    """

    candidate_steps = np.unique(
        np.clip(
            np.logspace(np.log10(10), np.log10(2500), 36).astype(int),
            10,
            2500,
        )
    )
    candidate_times = candidate_steps * config.dt
    flatness_values = np.array(
        [local_flatness(u, block_size) for block_size in candidate_steps]
    )

    valid_indices = np.where(candidate_times > 5.0 * config.tau_u)[0]
    if valid_indices.size == 0:
        raise RuntimeError("No admissible candidate window for T_beta.")

    below = valid_indices[flatness_values[valid_indices] <= 3.0]
    above = valid_indices[flatness_values[valid_indices] > 3.0]

    if below.size == 0 or above.size == 0:
        local_index = valid_indices[
            int(np.argmin(np.abs(flatness_values[valid_indices] - 3.0)))
        ]
        return (
            candidate_times,
            flatness_values,
            float(candidate_times[local_index]),
            int(candidate_steps[local_index]),
        )

    lower_index = below[-1]
    upper_candidates = above[above > lower_index]
    if upper_candidates.size == 0:
        upper_index = above[0]
    else:
        upper_index = upper_candidates[0]

    lower_step = int(candidate_steps[min(lower_index, upper_index)])
    upper_step = int(candidate_steps[max(lower_index, upper_index)])
    refined_steps = np.arange(lower_step, upper_step + 1, dtype=int)
    refined_flatness = np.array(
        [local_flatness(u, block_size) for block_size in refined_steps]
    )

    tolerance = 0.01
    admissible = np.where(
        (refined_flatness <= 3.0)
        & (refined_flatness >= 3.0 - tolerance)
    )[0]

    if admissible.size > 0:
        refined_index = int(admissible[0])
    else:
        refined_index = int(np.argmin(np.abs(refined_flatness - 3.0)))

    estimated_steps = int(refined_steps[refined_index])
    estimated_time = float(estimated_steps * config.dt)

    return candidate_times, flatness_values, estimated_time, estimated_steps


def reconstruct_beta(
    u: np.ndarray,
    config: SimulationConfig,
    beta_window_steps: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Reconstruct beta from the local variance of u."""

    blocks = block_view(u, beta_window_steps)
    means = blocks.mean(axis=1, keepdims=True)
    variances = np.mean((blocks - means) ** 2, axis=1)

    if np.any(variances <= 0.0):
        raise RuntimeError("At least one local variance is nonpositive.")

    beta_hat = config.beta0 / (2.0 * variances)
    beta_window_time = beta_window_steps * config.dt
    times = (np.arange(beta_hat.size) + 0.5) * beta_window_time

    return times, beta_hat


def gamma_shape_mle(sample: np.ndarray) -> float:
    """
    Maximum-likelihood estimate of the Gamma shape parameter.

    The scale is treated as unknown.  The estimate solves

        log(k) - psi(k) = log(mean(beta)) - mean(log(beta)).

    A standard analytic approximation is used as the initial value and the
    equation is solved by Newton iterations.
    """

    values = np.asarray(sample, dtype=float)
    if np.any(values <= 0.0):
        raise ValueError("Gamma samples must be strictly positive.")

    statistic = np.log(np.mean(values)) - np.mean(np.log(values))
    if statistic <= 0.0:
        return 1.0e12

    shape = (
        3.0
        - statistic
        + np.sqrt((statistic - 3.0) ** 2 + 24.0 * statistic)
    ) / (12.0 * statistic)

    for _ in range(50):
        function = np.log(shape) - digamma(shape) - statistic
        derivative = 1.0 / shape - polygamma(1, shape)
        updated_shape = shape - function / derivative

        if not np.isfinite(updated_shape) or updated_shape <= 0.0:
            updated_shape = 0.5 * shape

        if abs(updated_shape - shape) <= 1.0e-11 * max(1.0, shape):
            shape = updated_shape
            break

        shape = updated_shape

    return float(shape)


def reconstruct_q(
    beta_hat: np.ndarray,
    beta_hat_times: np.ndarray,
    window: int,
    stride: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Reconstruct the type-A parameter q from the beta residence process.

    The Gamma shape k is estimated by maximum likelihood in each outer
    window, and is converted to the type-A parameter through

        q = 1 + 1/(k + 1/2).

    This is less biased than the raw second-moment estimator for the finite
    windows used in the numerical proof of principle.
    """

    if beta_hat.size < window:
        raise ValueError("The reconstructed beta series is shorter than the q window.")

    q_values = []
    q_times = []

    for start in range(0, beta_hat.size - window + 1, stride):
        stop = start + window
        sample = beta_hat[start:stop]

        shape = gamma_shape_mle(sample)
        q_value = 1.0 + 1.0 / (shape + 0.5)

        q_values.append(q_value)
        q_times.append(np.mean(beta_hat_times[start:stop]))

    return np.asarray(q_times), np.asarray(q_values)


def induced_q_density(
    q: np.ndarray,
    config: SimulationConfig,
) -> np.ndarray:
    """Analytic type-A q density induced by the truncated log-normal h(n)."""

    q = np.asarray(q, dtype=float)
    density = np.zeros_like(q)

    q_max = 1.0 + 2.0 / config.n_min
    valid = (q > 1.0) & (q <= q_max)
    values = q[valid]

    argument = (
        np.log(2.0 / (config.n0 * (values - 1.0)))
        + 0.5 * config.sigma_eta**2
    )

    mu = np.log(config.n0) - 0.5 * config.sigma_eta**2
    z_min = (np.log(config.n_min) - mu) / config.sigma_eta
    truncation_probability = 0.5 * math.erfc(z_min / np.sqrt(2.0))

    density[valid] = (
        np.exp(-argument**2 / (2.0 * config.sigma_eta**2))
        / (
            (values - 1.0)
            * config.sigma_eta
            * np.sqrt(2.0 * np.pi)
            * truncation_probability
        )
    )

    return density


def inverse_residence_weights(q_samples: np.ndarray) -> np.ndarray:
    """Return normalized weights proportional to 1/Z_n for type-A reconstruction."""

    q_samples = np.asarray(q_samples, dtype=float)
    if np.any((q_samples <= 1.0) | (q_samples >= 3.0)):
        raise ValueError("Reconstructed q values must satisfy 1<q<3.")

    n_samples = 2.0 / (q_samples - 1.0)
    log_weights = -log_type_a_normalization(n_samples)
    weights = np.exp(log_weights - np.max(log_weights))
    return weights / np.sum(weights)


def averaged_q_exponential(
    x: np.ndarray,
    q_samples: np.ndarray,
    weights: np.ndarray | None = None,
) -> np.ndarray:
    """Average the type-A q-exponential over q samples."""

    x = np.asarray(x, dtype=float)
    q_samples = np.asarray(q_samples, dtype=float)

    if np.any(q_samples <= 1.0):
        raise ValueError("All q samples must be larger than one.")

    kernel = (
        1.0 + (q_samples[:, None] - 1.0) * x[None, :]
    ) ** (-1.0 / (q_samples[:, None] - 1.0))

    if weights is None:
        return np.mean(kernel, axis=0)

    normalized_weights = np.asarray(weights, dtype=float)
    normalized_weights = normalized_weights / np.sum(normalized_weights)
    return np.sum(normalized_weights[:, None] * kernel, axis=0)


def configure_matplotlib() -> None:
    """Apply the publication style used in the manuscript."""

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
            "mathtext.fontset": "stixsans",
            "font.size": 13,
            "axes.linewidth": 1.2,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def add_panel_label(axis: plt.Axes, label: str) -> None:
    """Place a panel label inside the lower-left corner."""

    axis.text(
        0.03,
        0.07,
        label,
        transform=axis.transAxes,
        fontsize=14,
        va="bottom",
        ha="left",
    )


def save_figure(
    figure: plt.Figure,
    output_stem: Path,
) -> None:
    """Save one figure as vector PDF and 300-dpi PNG."""

    figure.tight_layout(pad=0.55)
    figure.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(output_stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(figure)


def make_figures(
    output_dir: Path,
    config: SimulationConfig,
    figure_data: Dict[str, np.ndarray],
) -> None:
    """Create the four panels and the combined two-by-two figure."""

    configure_matplotlib()

    # Panel (a): local flatness.
    figure, axis = plt.subplots(figsize=(5.3, 4.1))
    axis.plot(
        figure_data["candidate_times"],
        figure_data["flatness_values"],
        marker="o",
        markersize=4,
        linewidth=1.8,
    )
    axis.axhline(3.0, linestyle="--", linewidth=1.4, color="black")
    axis.axvline(
        figure_data["T_beta_est"],
        linestyle=":",
        linewidth=1.4,
        color="black",
    )
    axis.set_xscale("log")
    axis.set_xlabel(r"$\Delta$", fontsize=15)
    axis.set_ylabel(r"$\mathcal{K}_u(\Delta)$", fontsize=15)
    axis.tick_params(which="both", top=True, right=True, labelsize=13)
    add_panel_label(axis, "(a)")
    save_figure(figure, output_dir / "FigTimeSeries_a_flatness")

    # Panel (b): autocorrelation functions.
    figure, axis = plt.subplots(figsize=(5.3, 4.1))
    axis.plot(
        figure_data["lags_u"],
        figure_data["autocorrelation_u"],
        linewidth=1.6,
        label=r"$u$",
    )
    axis.plot(
        figure_data["lags_beta"],
        figure_data["autocorrelation_beta"],
        linewidth=1.8,
        label=r"$\widehat{\beta}$",
    )
    axis.plot(
        figure_data["lags_q"],
        figure_data["autocorrelation_q"],
        linewidth=1.8,
        label=r"$\widehat{q}$",
    )
    axis.set_xlabel("lag", fontsize=15)
    axis.set_ylabel("autocorrelation", fontsize=15)
    axis.set_xlim(0.0, config.autocorrelation_max_lag)
    axis.set_ylim(-0.05, 1.05)
    axis.tick_params(which="both", top=True, right=True, labelsize=13)
    axis.legend(frameon=False, fontsize=11, loc="upper right")
    add_panel_label(axis, "(b)")
    save_figure(figure, output_dir / "FigTimeSeries_b_autocorr")

    # Panel (c): true and reconstructed type-A q distributions.
    figure, axis = plt.subplots(figsize=(5.3, 4.1))
    axis.hist(
        figure_data["q_empirical"],
        bins=20,
        weights=figure_data["q_empirical_weights"],
        density=True,
        alpha=0.5,
        label=r"reconstructed $\widehat{g}(q)$",
    )
    axis.plot(
        figure_data["q_grid"],
        figure_data["g_true"],
        linewidth=1.8,
        label=r"true $g(q)$",
    )
    axis.set_xlabel(r"$q$", fontsize=15)
    axis.set_ylabel("density", fontsize=15)
    axis.tick_params(which="both", top=True, right=True, labelsize=13)
    axis.legend(frameon=False, fontsize=11)
    add_panel_label(axis, "(c)")
    save_figure(figure, output_dir / "FigTimeSeries_c_gq")

    # Panel (d): ratios of reconstructed and fixed-q factors.
    figure, axis = plt.subplots(figsize=(5.3, 4.1))
    axis.plot(
        figure_data["x_grid"],
        figure_data["B_reconstructed"] / figure_data["B_true"],
        linewidth=1.8,
        label=r"$\widehat{\overline{B}}(x)/\overline{B}_{\rm true}(x)$",
    )
    axis.plot(
        figure_data["x_grid"],
        figure_data["B_fixed"] / figure_data["B_true"],
        linewidth=1.8,
        label=r"$B_{\widehat q_0}(x)/\overline{B}_{\rm true}(x)$",
    )
    axis.axhline(1.0, linestyle="--", linewidth=1.4, color="black")
    axis.set_xlabel(r"$x=\beta_0 E$", fontsize=15)
    axis.set_ylabel("ratio", fontsize=15)
    axis.set_xlim(0.0, config.ratio_x_max)
    axis.set_ylim(config.ratio_y_min, config.ratio_y_max)
    axis.tick_params(which="both", top=True, right=True, labelsize=13)
    axis.legend(frameon=False, fontsize=10.5, loc="upper left")
    add_panel_label(axis, "(d)")
    save_figure(figure, output_dir / "FigTimeSeries_d_ratio")

    combine_panels(output_dir)


def combine_panels(output_dir: Path) -> None:
    """Assemble the four PNG panels into one two-by-two PDF and PNG."""

    panel_paths = [
        output_dir / "FigTimeSeries_a_flatness.png",
        output_dir / "FigTimeSeries_b_autocorr.png",
        output_dir / "FigTimeSeries_c_gq.png",
        output_dir / "FigTimeSeries_d_ratio.png",
    ]

    images = [
        ImageOps.expand(Image.open(path).convert("RGB"), border=18, fill="white")
        for path in panel_paths
    ]

    tile_width = max(image.width for image in images)
    tile_height = max(image.height for image in images)

    canvas = Image.new(
        "RGB",
        (2 * tile_width, 2 * tile_height),
        "white",
    )

    positions = [
        (0, 0),
        (tile_width, 0),
        (0, tile_height),
        (tile_width, tile_height),
    ]

    for image, position in zip(images, positions):
        x = position[0] + (tile_width - image.width) // 2
        y = position[1] + (tile_height - image.height) // 2
        canvas.paste(image, (x, y))

    canvas.save(output_dir / "FigTimeSeries_combined.png")
    canvas.save(
        output_dir / "FigTimeSeries_combined.pdf",
        "PDF",
        resolution=300.0,
    )


def prepare_figure_data(
    simulation: Dict[str, np.ndarray],
    config: SimulationConfig,
) -> Tuple[Dict[str, np.ndarray], Dict[str, float]]:
    """Perform the type-A reconstruction and collect all plotted arrays."""

    candidate_times, flatness_values, T_beta_est, beta_window_steps = (
        infer_beta_scale(simulation["u"], config)
    )

    beta_hat_times, beta_hat = reconstruct_beta(
        simulation["u"],
        config,
        beta_window_steps,
    )

    # Sliding windows are retained for the reconstructed q process and its
    # autocorrelation.  The longer outer window is made possible by the
    # strengthened separation tau_beta << tau_q.
    q_hat_times, q_hat = reconstruct_q(
        beta_hat,
        beta_hat_times,
        window=config.q_window_blocks,
        stride=config.q_window_stride,
    )

    # Nonoverlapping windows define the empirical type-A measure g_hat(q).
    q_empirical_times, q_empirical = reconstruct_q(
        beta_hat,
        beta_hat_times,
        window=config.q_window_blocks,
        stride=config.q_window_blocks,
    )
    q_empirical_weights = inverse_residence_weights(q_empirical)

    # Autocorrelation of u is evaluated on a subsampled series with unit time step.
    u_subsample = max(1, int(round(1.0 / config.dt)))
    u_for_correlation = simulation["u"][::u_subsample]
    u_correlation_dt = config.dt * u_subsample
    u_max_lag_steps = int(round(config.autocorrelation_max_lag / u_correlation_dt))

    autocorrelation_u = autocorrelation_fft(
        u_for_correlation,
        u_max_lag_steps,
    )
    lags_u = u_correlation_dt * np.arange(autocorrelation_u.size)

    beta_correlation_dt = T_beta_est
    beta_max_lag_steps = min(
        int(round(config.autocorrelation_max_lag / beta_correlation_dt)),
        beta_hat.size - 1,
    )
    autocorrelation_beta = autocorrelation_fft(
        beta_hat,
        beta_max_lag_steps,
    )
    lags_beta = beta_correlation_dt * np.arange(autocorrelation_beta.size)

    q_correlation_dt = config.q_window_stride * T_beta_est
    q_max_lag_steps = min(
        int(round(config.autocorrelation_max_lag / q_correlation_dt)),
        q_hat.size - 1,
    )
    autocorrelation_q = autocorrelation_fft(
        q_hat,
        q_max_lag_steps,
    )
    lags_q = q_correlation_dt * np.arange(autocorrelation_q.size)

    # Exact type-A reference obtained from the prescribed parent h(n), not
    # from the residence samples used by the Langevin process.
    q_type_a_nodes, q_type_a_weights = type_a_parent_quadrature(config)

    q_min = max(
        1.001,
        min(float(q_empirical.min()), float(q_type_a_nodes.min())) - 0.01,
    )
    q_max = max(float(q_empirical.max()), float(q_type_a_nodes.max())) + 0.01
    q_grid = np.linspace(q_min, q_max, 500)
    g_true = induced_q_density(q_grid, config)

    x_grid = np.linspace(0.0, 35.0, 300)
    B_true = averaged_q_exponential(
        x_grid,
        q_type_a_nodes,
        q_type_a_weights,
    )
    B_reconstructed = averaged_q_exponential(
        x_grid,
        q_empirical,
        q_empirical_weights,
    )

    mean_q_true = float(np.sum(q_type_a_weights * q_type_a_nodes))
    variance_q_true = float(
        np.sum(q_type_a_weights * (q_type_a_nodes - mean_q_true) ** 2)
    )
    mean_q_reconstructed = float(
        np.sum(q_empirical_weights * q_empirical)
    )
    variance_q_reconstructed = float(
        np.sum(
            q_empirical_weights
            * (q_empirical - mean_q_reconstructed) ** 2
        )
    )

    B_fixed = (
        1.0 + (mean_q_reconstructed - 1.0) * x_grid
    ) ** (-1.0 / (mean_q_reconstructed - 1.0))

    figure_data = {
        "candidate_times": candidate_times,
        "flatness_values": flatness_values,
        "T_beta_est": np.asarray(T_beta_est),
        "beta_hat_times": beta_hat_times,
        "beta_hat": beta_hat,
        "q_hat_times": q_hat_times,
        "q_hat": q_hat,
        "q_empirical_times": q_empirical_times,
        "q_empirical": q_empirical,
        "q_empirical_weights": q_empirical_weights,
        "lags_u": lags_u,
        "autocorrelation_u": autocorrelation_u,
        "lags_beta": lags_beta,
        "autocorrelation_beta": autocorrelation_beta,
        "lags_q": lags_q,
        "autocorrelation_q": autocorrelation_q,
        "q_grid": q_grid,
        "g_true": g_true,
        "x_grid": x_grid,
        "B_true": B_true,
        "B_reconstructed": B_reconstructed,
        "B_fixed": B_fixed,
    }

    summary = {
        "T_beta_true": config.tau_beta,
        "T_beta_estimated": T_beta_est,
        "q_window_duration": config.q_window_blocks * T_beta_est,
        "q_window_stride": config.q_window_stride * T_beta_est,
        "mean_q_true": mean_q_true,
        "mean_q_reconstructed": mean_q_reconstructed,
        "standard_deviation_q_true": float(np.sqrt(variance_q_true)),
        "standard_deviation_q_reconstructed": float(
            np.sqrt(variance_q_reconstructed)
        ),
    }

    return figure_data, summary


def save_numerical_outputs(
    output_dir: Path,
    config: SimulationConfig,
    simulation: Dict[str, np.ndarray],
    figure_data: Dict[str, np.ndarray],
    summary: Dict[str, float],
    save_full_timeseries: bool,
) -> None:
    """Save parameters, summary statistics, and all plotted arrays."""

    with (output_dir / "parameters.json").open("w", encoding="utf-8") as handle:
        json.dump(asdict(config), handle, indent=2)

    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    np.savez_compressed(
        output_dir / "figure_data.npz",
        **figure_data,
    )

    if save_full_timeseries:
        np.savez_compressed(
            output_dir / "full_simulated_timeseries.npz",
            time=simulation["time"],
            u=simulation["u"],
            beta=simulation["beta_time"],
            n=simulation["n_time"],
            q=simulation["q_time"],
        )


def parse_arguments() -> argparse.Namespace:
    """Parse command-line options."""

    parser = argparse.ArgumentParser(
        description="Generate the type-A nested-superstatistics time-series figures."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("nested_timeseries_results"),
        help="Directory in which the figures and numerical data are saved.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=12345,
        help="Random seed used for the simulation.",
    )
    parser.add_argument(
        "--save-full-timeseries",
        action="store_true",
        help="Also save the complete simulated u, beta, n, and q time series.",
    )
    return parser.parse_args()


def main() -> None:
    """Run the simulation, reconstruction, and figure generation."""

    arguments = parse_arguments()
    output_dir = arguments.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    config = SimulationConfig(seed=arguments.seed)
    simulation = simulate_hierarchical_process(config)
    figure_data, summary = prepare_figure_data(simulation, config)

    save_numerical_outputs(
        output_dir=output_dir,
        config=config,
        simulation=simulation,
        figure_data=figure_data,
        summary=summary,
        save_full_timeseries=arguments.save_full_timeseries,
    )

    make_figures(
        output_dir=output_dir,
        config=config,
        figure_data=figure_data,
    )

    print(f"Results written to: {output_dir}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
