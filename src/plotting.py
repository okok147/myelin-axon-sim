"""Visualization utilities for disease-modular myelin simulations."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Sequence

import matplotlib.pyplot as plt
import numpy as np

from .axon_builder import SimulationOutput


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def plot_voltage_traces(
    result: SimulationOutput,
    output_path: str | Path,
    title: str,
    n_traces: int = 6,
) -> None:
    output = Path(output_path)
    _ensure_parent(output)

    node_ids = np.linspace(0, result.node_indices.size - 1, min(n_traces, result.node_indices.size), dtype=int)
    comp_ids = [int(result.node_indices[i]) for i in node_ids]

    fig, ax = plt.subplots(figsize=(10.5, 5.3))
    for i, comp in enumerate(comp_ids):
        node_no = int(np.where(result.node_indices == comp)[0][0])
        ax.plot(result.time_ms, result.vm_mV[:, comp], lw=1.6, label=f"Node {node_no}")

    ax.axhline(0.0, ls="--", lw=1.0, color="black", alpha=0.6)
    ax.set_title(title)
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Vm (mV)")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_space_time_heatmap(result: SimulationOutput, output_path: str | Path, title: str) -> None:
    output = Path(output_path)
    _ensure_parent(output)

    fig, ax = plt.subplots(figsize=(11.2, 4.7))
    extent = [result.time_ms[0], result.time_ms[-1], result.x_um[0], result.x_um[-1]]
    im = ax.imshow(
        result.vm_mV.T,
        aspect="auto",
        origin="lower",
        extent=extent,
        cmap="coolwarm",
        vmin=-90,
        vmax=50,
    )

    for pos in result.x_um[result.node_indices]:
        ax.axhline(pos, color="white", lw=0.35, alpha=0.35)

    ax.set_title(title)
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Position (um)")
    cbar = fig.colorbar(im, ax=ax, pad=0.01)
    cbar.set_label("Vm (mV)")
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_block_probability(
    severities: Sequence[float],
    temperatures_C: Sequence[float],
    block_probability: np.ndarray,
    output_path: str | Path,
    title: str,
) -> None:
    output = Path(output_path)
    _ensure_parent(output)

    sev = np.asarray(severities, dtype=float)
    probs = np.asarray(block_probability, dtype=float)

    fig, ax = plt.subplots(figsize=(8.4, 5.2))
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd", "#d62728"]

    for i, temp in enumerate(temperatures_C):
        ax.plot(
            sev,
            probs[i],
            marker="o",
            lw=2.0,
            color=colors[i % len(colors)],
            label=f"{temp:.1f} C",
        )

    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("Demyelination severity")
    ax.set_ylabel("Conduction block probability")
    ax.set_title(title)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_velocity_and_energy(
    severities: Sequence[float],
    velocities: Sequence[float],
    energies: Sequence[float],
    output_path: str | Path,
    title: str,
) -> None:
    output = Path(output_path)
    _ensure_parent(output)

    sev = np.asarray(severities, dtype=float)
    vel = np.asarray(velocities, dtype=float)
    eng = np.asarray(energies, dtype=float)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8.7, 6.8), sharex=True)
    ax1.plot(sev, vel, marker="o", lw=2.0, color="#1f77b4")
    ax1.set_ylabel("Velocity (m/s)")
    ax1.grid(alpha=0.25)

    ax2.plot(sev, eng, marker="s", lw=2.0, color="#d62728")
    ax2.set_xlabel("Demyelination severity")
    ax2.set_ylabel("Na charge proxy (pC)")
    ax2.grid(alpha=0.25)

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_synchrony(
    severities: Sequence[float],
    delta_d_ms: Sequence[float],
    phase_deg_by_freq: Dict[float, Sequence[float]],
    output_path: str | Path,
    title: str,
) -> None:
    output = Path(output_path)
    _ensure_parent(output)

    sev = np.asarray(severities, dtype=float)
    delays = np.asarray(delta_d_ms, dtype=float)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9.0, 7.4), sharex=True)

    ax1.plot(sev, delays, marker="o", lw=2.2, color="#9467bd")
    ax1.axhline(0.0, color="black", lw=1.0, alpha=0.55)
    ax1.set_ylabel("Arrival mismatch Δd (ms)")
    ax1.grid(alpha=0.25)

    palette = ["#2ca02c", "#ff7f0e", "#17becf", "#8c564b", "#d62728"]
    for i, freq in enumerate(sorted(phase_deg_by_freq)):
        phase_deg = np.asarray(phase_deg_by_freq[freq], dtype=float)
        ax2.plot(
            sev,
            phase_deg,
            marker="s",
            lw=2.0,
            color=palette[i % len(palette)],
            label=f"{freq:g} Hz",
        )

    ax2.axhline(0.0, color="black", lw=1.0, alpha=0.55)
    ax2.set_xlabel("Pathway-B demyelination severity")
    ax2.set_ylabel("Phase mismatch Δφ (deg)")
    ax2.grid(alpha=0.25)
    ax2.legend(frameon=False)

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_cap_dispersion(
    time_ms: np.ndarray,
    cap_signal: np.ndarray,
    arrival_times_ms: np.ndarray,
    blocked_mask: np.ndarray,
    output_path: str | Path,
    title: str,
) -> None:
    """Plot bundle-level compound action potential and arrival spread."""

    output = Path(output_path)
    _ensure_parent(output)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9.0, 6.8), sharex=False)

    ax1.plot(time_ms, cap_signal, lw=2.0, color="#1f77b4")
    ax1.set_xlabel("Time (ms)")
    ax1.set_ylabel("Normalized CAP")
    ax1.grid(alpha=0.25)

    finite_arrivals = arrival_times_ms[np.isfinite(arrival_times_ms)]
    if finite_arrivals.size > 0:
        ax2.hist(finite_arrivals, bins=min(12, max(4, finite_arrivals.size // 2)), color="#2ca02c", alpha=0.75)
    ax2.set_xlabel("Fiber arrival time (ms)")
    ax2.set_ylabel("Fiber count")
    ax2.grid(alpha=0.25)

    block_fraction = float(np.mean(blocked_mask.astype(float))) if blocked_mask.size else 0.0
    ax2.set_title(f"Block fraction: {100.0 * block_fraction:.1f}%")

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)
