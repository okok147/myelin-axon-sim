"""Plotting helpers for axon simulations."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import matplotlib.pyplot as plt
import numpy as np

from .axon_model import SimulationResult


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def plot_voltage_traces(
    result: SimulationResult,
    output_path: str | Path,
    title: str,
    max_node_traces: int = 6,
) -> None:
    """Plot Vm(t) for selected nodes plus representative internodes."""

    output = Path(output_path)
    _ensure_parent(output)

    node_indices = result.node_indices
    n_nodes = node_indices.size
    node_sample = np.linspace(0, n_nodes - 1, min(max_node_traces, n_nodes), dtype=int)

    plotted_indices: List[int] = [int(node_indices[i]) for i in node_sample]

    # Add representative internode compartments if available.
    if result.vm_mV.shape[1] > n_nodes:
        internode_candidates = np.setdiff1d(np.arange(result.vm_mV.shape[1]), node_indices)
        if internode_candidates.size > 0:
            mid = internode_candidates[internode_candidates.size // 2]
            near_end = internode_candidates[-1]
            plotted_indices.extend([int(mid), int(near_end)])

    plotted_indices = list(dict.fromkeys(plotted_indices))

    fig, ax = plt.subplots(figsize=(10.5, 5.4))
    for idx in plotted_indices:
        label = f"Node {np.where(node_indices == idx)[0][0]}" if idx in node_indices else f"Internode comp {idx}"
        ax.plot(result.time_ms, result.vm_mV[:, idx], lw=1.5, label=label)

    ax.axhline(0.0, color="black", lw=0.9, ls="--", alpha=0.55, label="0 mV threshold")
    ax.set_title(title)
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Membrane potential (mV)")
    ax.grid(alpha=0.25)
    ax.legend(ncol=2, fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_space_time_heatmap(result: SimulationResult, output_path: str | Path, title: str) -> None:
    """Plot Vm(x,t) heatmap to visualize saltatory propagation."""

    output = Path(output_path)
    _ensure_parent(output)

    fig, ax = plt.subplots(figsize=(11, 4.6))
    extent = [
        result.time_ms[0],
        result.time_ms[-1],
        result.x_um[0],
        result.x_um[-1],
    ]

    im = ax.imshow(
        result.vm_mV.T,
        aspect="auto",
        origin="lower",
        extent=extent,
        cmap="coolwarm",
        vmin=-90,
        vmax=55,
    )

    node_positions = result.x_um[result.node_indices]
    for pos in node_positions:
        ax.axhline(pos, color="white", lw=0.3, alpha=0.35)

    ax.set_title(title)
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Position along axon (um)")
    cbar = fig.colorbar(im, ax=ax, pad=0.01)
    cbar.set_label("Vm (mV)")
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_demyelination_sweep(
    factors: Sequence[float],
    velocities: Sequence[float],
    arrivals_ms: Sequence[float],
    output_path: str | Path,
    title: str,
) -> None:
    """Plot conduction slowing and arrival delay across demyelination levels."""

    output = Path(output_path)
    _ensure_parent(output)

    factors = np.asarray(factors, dtype=float)
    velocities = np.asarray(velocities, dtype=float)
    arrivals_ms = np.asarray(arrivals_ms, dtype=float)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8.8, 7.0), sharex=True)

    ax1.plot(factors, velocities, marker="o", lw=2.0, color="#1f77b4")
    ax1.set_ylabel("Conduction velocity (m/s)")
    ax1.grid(alpha=0.25)

    ax2.plot(factors, arrivals_ms, marker="s", lw=2.0, color="#d62728")
    ax2.set_xlabel("Demyelination factor (Rm/myelin decreases, Cm increases)")
    ax2.set_ylabel("Final-node arrival time (ms)")
    ax2.grid(alpha=0.25)

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_synchrony(
    factors: Sequence[float],
    delays_ms: Sequence[float],
    phase_deg_by_freq: Dict[float, Sequence[float]],
    output_path: str | Path,
    title: str,
) -> None:
    """Plot delay and phase misalignment for converging axonal pathways."""

    output = Path(output_path)
    _ensure_parent(output)

    factors = np.asarray(factors, dtype=float)
    delays_ms = np.asarray(delays_ms, dtype=float)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9.0, 7.2), sharex=True)

    ax1.plot(factors, delays_ms, marker="o", lw=2.1, color="#9467bd")
    ax1.axhline(0.0, color="black", lw=0.9, alpha=0.55)
    ax1.set_ylabel("Arrival-time difference Δd (ms)")
    ax1.grid(alpha=0.25)

    palette = ["#2ca02c", "#ff7f0e", "#17becf", "#8c564b"]
    for i, (freq, phases) in enumerate(sorted(phase_deg_by_freq.items(), key=lambda kv: kv[0])):
        ax2.plot(
            factors,
            np.asarray(phases, dtype=float),
            marker="s",
            lw=1.9,
            color=palette[i % len(palette)],
            label=f"{freq:g} Hz",
        )

    ax2.axhline(0.0, color="black", lw=0.9, alpha=0.55)
    ax2.set_xlabel("Pathway-B demyelination factor")
    ax2.set_ylabel("Phase difference Δφ (deg)")
    ax2.grid(alpha=0.25)
    ax2.legend(frameon=False, ncol=2)

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_energy_comparison(
    labels: Sequence[str],
    sodium_charge_pc: Sequence[float],
    output_path: str | Path,
    title: str,
) -> None:
    """Plot Na-charge energy proxy across conditions."""

    output = Path(output_path)
    _ensure_parent(output)

    labels = list(labels)
    values = np.asarray(sodium_charge_pc, dtype=float)

    fig, ax = plt.subplots(figsize=(6.8, 4.6))
    bars = ax.bar(labels, values, color=["#1f77b4", "#d62728", "#2ca02c"][: len(labels)])
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{val:.1f}", ha="center", va="bottom", fontsize=9)

    ax.set_ylabel("|Na current| integral (pC)")
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)
