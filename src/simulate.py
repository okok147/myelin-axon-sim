"""CLI for modular demyelinating-disease axon simulations.

Educational/research simulation only. Not a diagnostic tool.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

from .axon_builder import (
    AxonGeometry,
    AxonSimulator,
    ChannelParams,
    IntegrityProfile,
    SimulationOutput,
    StimulusParams,
    default_geometry,
    default_membrane_params,
    phase_misalignment,
)
from .disease_models import (
    PRESET_NAMES,
    DiseaseTrajectory,
    generate_disease_trajectory,
    recommended_stage_fraction,
    snapshot_from_trajectory,
)
from .plotting import (
    plot_block_probability,
    plot_cap_dispersion,
    plot_space_time_heatmap,
    plot_synchrony,
    plot_velocity_and_energy,
    plot_voltage_traces,
)


@dataclass
class BundleResult:
    arrival_times_ms: np.ndarray
    blocked_mask: np.ndarray
    cap_time_ms: np.ndarray
    cap_signal: np.ndarray


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Modular myelin/white-matter conduction simulator")

    parser.add_argument("--preset", choices=PRESET_NAMES, default="MS")
    parser.add_argument("--axon_type", choices=["CNS", "PNS"], default="CNS")

    parser.add_argument("--n_nodes", type=int, default=None)
    parser.add_argument("--internode_length_um", type=float, default=None)
    parser.add_argument("--node_length_um", type=float, default=None)
    parser.add_argument("--diameter_um", type=float, default=None)

    parser.add_argument("--temp_C", type=float, default=37.0)
    parser.add_argument("--demyelination_severity", type=float, default=0.7)
    parser.add_argument("--lesion_count", type=int, default=4)
    parser.add_argument("--lesion_len_um", type=float, default=700.0)

    parser.add_argument("--tstop_ms", type=float, default=8.0)
    parser.add_argument("--dt_ms", type=float, default=0.002)
    parser.add_argument("--seed", type=int, default=7)

    parser.add_argument("--sync", action="store_true", help="Run synchrony experiment with two pathways")

    parser.add_argument("--freq_hz", type=float, default=20.0, help="Additional synchrony frequency")
    parser.add_argument("--solver", choices=["solve_ivp", "rk4"], default="solve_ivp")
    parser.add_argument("--output_dir", default="outputs")
    parser.add_argument("--label", default="experiment")

    return parser


def _merge_geometry(args: argparse.Namespace) -> AxonGeometry:
    base = default_geometry(args.axon_type)
    return AxonGeometry(
        n_nodes=base.n_nodes if args.n_nodes is None else args.n_nodes,
        internode_length_um=base.internode_length_um
        if args.internode_length_um is None
        else args.internode_length_um,
        node_length_um=base.node_length_um if args.node_length_um is None else args.node_length_um,
        diameter_um=base.diameter_um if args.diameter_um is None else args.diameter_um,
        internode_segments=base.internode_segments,
    )


def _make_simulator(args: argparse.Namespace) -> AxonSimulator:
    geometry = _merge_geometry(args)
    membrane = default_membrane_params(args.axon_type)
    channels = ChannelParams()
    stimulus = StimulusParams(amplitude_nA=2.0, start_ms=0.2, duration_ms=0.2, target_node=0)
    return AxonSimulator(geometry=geometry, membrane=membrane, channels=channels, stimulus=stimulus)


def _result_summary(name: str, result: SimulationOutput) -> str:
    vel = f"{result.conduction_velocity_m_s:.3f} m/s" if np.isfinite(result.conduction_velocity_m_s) else "block"
    arr = f"{result.arrival_time_ms:.3f} ms" if np.isfinite(result.arrival_time_ms) else "no arrival"
    return (
        f"[{name}] velocity={vel}, arrival={arr}, blocked={result.blocked}, "
        f"NaCharge={result.sodium_charge_pC:.2f} pC"
    )


def run_single_trial(
    simulator: AxonSimulator,
    preset: str,
    axon_type: str,
    severity: float,
    lesion_count: int,
    lesion_len_um: float,
    seed: int,
    temp_C: float,
    tstop_ms: float,
    dt_ms: float,
    solver: str,
) -> Tuple[SimulationOutput, DiseaseTrajectory, IntegrityProfile]:
    trajectory = generate_disease_trajectory(
        preset=preset,
        x_um=simulator.x_positions_m * 1e6,
        is_node=simulator.is_node,
        axon_type=axon_type,
        demyelination_severity=severity,
        lesion_count=lesion_count,
        lesion_len_um=lesion_len_um,
        seed=seed,
    )
    stage = recommended_stage_fraction(preset)
    profile = snapshot_from_trajectory(trajectory, stage)
    result = simulator.run(
        integrity=profile,
        temp_C=temp_C,
        tstop_ms=tstop_ms,
        dt_ms=dt_ms,
        solver=solver,
    )
    return result, trajectory, profile


def estimate_block_probability(
    simulator: AxonSimulator,
    preset: str,
    axon_type: str,
    severities: Sequence[float],
    temperatures_C: Sequence[float],
    lesion_count: int,
    lesion_len_um: float,
    base_seed: int,
    tstop_ms: float,
    dt_ms: float,
    solver: str,
    trials_per_point: int = 5,
) -> np.ndarray:
    block_prob = np.zeros((len(temperatures_C), len(severities)), dtype=float)

    for i, temp_C in enumerate(temperatures_C):
        for j, sev in enumerate(severities):
            blocked = []
            for k in range(trials_per_point):
                seed = base_seed + i * 100 + j * 10 + k
                result, _, _ = run_single_trial(
                    simulator=simulator,
                    preset=preset,
                    axon_type=axon_type,
                    severity=float(sev),
                    lesion_count=lesion_count,
                    lesion_len_um=lesion_len_um,
                    seed=seed,
                    temp_C=float(temp_C),
                    tstop_ms=tstop_ms,
                    dt_ms=dt_ms,
                    solver=solver,
                )
                blocked.append(float(result.blocked))
            block_prob[i, j] = float(np.mean(blocked))
    return block_prob


def _cap_signal_from_arrivals(
    arrivals_ms: np.ndarray,
    blocked_mask: np.ndarray,
    sigma_ms: float,
    tstop_ms: float,
) -> Tuple[np.ndarray, np.ndarray]:
    t = np.linspace(0.0, tstop_ms + 4.0, 2200)
    cap = np.zeros_like(t)
    for arr, blocked in zip(arrivals_ms, blocked_mask):
        if blocked or not np.isfinite(arr):
            continue
        cap += np.exp(-0.5 * ((t - arr) / sigma_ms) ** 2)

    if cap.max() > 0:
        cap = cap / cap.max()
    return t, cap


def simulate_bundle_dispersion(
    simulator: AxonSimulator,
    preset: str,
    axon_type: str,
    severity: float,
    lesion_count: int,
    lesion_len_um: float,
    seed: int,
    temp_C: float,
    tstop_ms: float,
    dt_ms: float,
    solver: str,
    n_fibers: int = 16,
) -> BundleResult:
    rng = np.random.default_rng(seed)
    arrivals = np.full(n_fibers, np.nan, dtype=float)
    blocked = np.zeros(n_fibers, dtype=bool)

    for i in range(n_fibers):
        sev_i = float(np.clip(severity + rng.normal(0.0, 0.08), 0.0, 1.0))
        lesion_i = max(1, int(round(lesion_count + rng.normal(0.0, 1.0))))

        result, _, _ = run_single_trial(
            simulator=simulator,
            preset=preset,
            axon_type=axon_type,
            severity=sev_i,
            lesion_count=lesion_i,
            lesion_len_um=lesion_len_um,
            seed=seed + 1000 + i,
            temp_C=temp_C,
            tstop_ms=tstop_ms,
            dt_ms=dt_ms,
            solver=solver,
        )

        arrivals[i] = result.arrival_time_ms
        blocked[i] = result.blocked

    sigma_map = {"GBS": 0.2, "CIDP": 0.16, "CMT": 0.08}
    sigma = sigma_map.get(preset.upper(), 0.12)
    cap_t, cap_y = _cap_signal_from_arrivals(arrivals_ms=arrivals, blocked_mask=blocked, sigma_ms=sigma, tstop_ms=tstop_ms)
    return BundleResult(arrival_times_ms=arrivals, blocked_mask=blocked, cap_time_ms=cap_t, cap_signal=cap_y)


def run_synchrony_experiment(
    simulator: AxonSimulator,
    preset: str,
    axon_type: str,
    max_severity: float,
    lesion_count: int,
    lesion_len_um: float,
    seed: int,
    temp_C: float,
    tstop_ms: float,
    dt_ms: float,
    solver: str,
    frequencies_hz: Sequence[float],
) -> Tuple[np.ndarray, np.ndarray, Dict[float, np.ndarray]]:
    severities = np.linspace(0.0, max(0.05, max_severity), 7)

    baseline_sev = min(0.22, max_severity * 0.35 + 0.03)
    path_a, _, _ = run_single_trial(
        simulator=simulator,
        preset=preset,
        axon_type=axon_type,
        severity=baseline_sev,
        lesion_count=lesion_count,
        lesion_len_um=lesion_len_um,
        seed=seed + 5000,
        temp_C=temp_C,
        tstop_ms=tstop_ms,
        dt_ms=dt_ms,
        solver=solver,
    )

    if not np.isfinite(path_a.arrival_time_ms):
        raise RuntimeError("Pathway A failed to propagate; cannot compute synchrony")

    delta_ms = np.full(severities.size, np.nan, dtype=float)
    phase_deg: Dict[float, np.ndarray] = {float(f): np.full(severities.size, np.nan, dtype=float) for f in frequencies_hz}

    for i, sev in enumerate(severities):
        path_b, _, _ = run_single_trial(
            simulator=simulator,
            preset=preset,
            axon_type=axon_type,
            severity=float(sev),
            lesion_count=lesion_count,
            lesion_len_um=lesion_len_um,
            seed=seed + 6000 + i,
            temp_C=temp_C,
            tstop_ms=tstop_ms,
            dt_ms=dt_ms,
            solver=solver,
        )

        if np.isfinite(path_b.arrival_time_ms):
            delta = path_a.arrival_time_ms - path_b.arrival_time_ms
        else:
            delta = np.nan

        delta_ms[i] = delta

        for freq in frequencies_hz:
            if np.isfinite(delta):
                _, deg = phase_misalignment(delay_ms=delta, freq_hz=float(freq))
                phase_deg[float(freq)][i] = deg

    return severities, delta_ms, phase_deg


def main() -> None:
    args = _build_parser().parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    simulator = _make_simulator(args)

    healthy = simulator.run(
        integrity=IntegrityProfile.healthy(simulator.n_compartments),
        temp_C=args.temp_C,
        tstop_ms=args.tstop_ms,
        dt_ms=args.dt_ms,
        solver=args.solver,
    )

    disease, trajectory, profile = run_single_trial(
        simulator=simulator,
        preset=args.preset,
        axon_type=args.axon_type,
        severity=float(np.clip(args.demyelination_severity, 0.0, 1.0)),
        lesion_count=args.lesion_count,
        lesion_len_um=args.lesion_len_um,
        seed=args.seed,
        temp_C=args.temp_C,
        tstop_ms=args.tstop_ms,
        dt_ms=args.dt_ms,
        solver=args.solver,
    )

    print(_result_summary("healthy", healthy))
    print(_result_summary(args.preset, disease))

    plot_voltage_traces(
        result=disease,
        output_path=out_dir / f"{args.label}_{args.preset}_traces.png",
        title=f"{args.preset} membrane potentials",
    )
    plot_space_time_heatmap(
        result=disease,
        output_path=out_dir / f"{args.label}_{args.preset}_heatmap.png",
        title=f"{args.preset} Vm(x,t)",
    )

    severity_sweep = np.linspace(0.0, max(0.05, args.demyelination_severity), 5)
    velocities: List[float] = []
    energies: List[float] = []
    for i, sev in enumerate(severity_sweep):
        out, _, _ = run_single_trial(
            simulator=simulator,
            preset=args.preset,
            axon_type=args.axon_type,
            severity=float(sev),
            lesion_count=args.lesion_count,
            lesion_len_um=args.lesion_len_um,
            seed=args.seed + 200 + i,
            temp_C=args.temp_C,
            tstop_ms=args.tstop_ms,
            dt_ms=args.dt_ms,
            solver=args.solver,
        )
        velocities.append(float(out.conduction_velocity_m_s) if np.isfinite(out.conduction_velocity_m_s) else np.nan)
        energies.append(float(out.sodium_charge_pC))

    plot_velocity_and_energy(
        severities=severity_sweep,
        velocities=velocities,
        energies=energies,
        output_path=out_dir / f"{args.label}_{args.preset}_velocity_energy.png",
        title=f"{args.preset}: velocity and Na-current energy proxy",
    )

    if args.preset.upper() == "MS":
        temps = [max(34.0, args.temp_C - 1.5), args.temp_C, args.temp_C + 2.0]
    else:
        temps = [args.temp_C]

    block_prob = estimate_block_probability(
        simulator=simulator,
        preset=args.preset,
        axon_type=args.axon_type,
        severities=severity_sweep,
        temperatures_C=temps,
        lesion_count=args.lesion_count,
        lesion_len_um=args.lesion_len_um,
        base_seed=args.seed + 1000,
        tstop_ms=args.tstop_ms,
        dt_ms=args.dt_ms,
        solver=args.solver,
        trials_per_point=2,
    )

    plot_block_probability(
        severities=severity_sweep,
        temperatures_C=temps,
        block_probability=block_prob,
        output_path=out_dir / f"{args.label}_{args.preset}_block_probability.png",
        title=f"{args.preset}: conduction block probability",
    )

    bundle_stats = None
    if args.preset.upper() in {"GBS", "CIDP", "CMT"}:
        bundle_stats = simulate_bundle_dispersion(
            simulator=simulator,
            preset=args.preset,
            axon_type=args.axon_type,
            severity=float(np.clip(args.demyelination_severity, 0.0, 1.0)),
            lesion_count=args.lesion_count,
            lesion_len_um=args.lesion_len_um,
            seed=args.seed + 3000,
            temp_C=args.temp_C,
            tstop_ms=args.tstop_ms,
            dt_ms=args.dt_ms,
            solver=args.solver,
        )

        plot_cap_dispersion(
            time_ms=bundle_stats.cap_time_ms,
            cap_signal=bundle_stats.cap_signal,
            arrival_times_ms=bundle_stats.arrival_times_ms,
            blocked_mask=bundle_stats.blocked_mask,
            output_path=out_dir / f"{args.label}_{args.preset}_cap_dispersion.png",
            title=f"{args.preset}: compound action potential dispersion",
        )

    sync_payload = None
    if args.sync:
        freq_set = sorted({10.0, 40.0, float(args.freq_hz)})
        sev_sync, d_sync, phase_sync = run_synchrony_experiment(
            simulator=simulator,
            preset=args.preset,
            axon_type=args.axon_type,
            max_severity=float(np.clip(args.demyelination_severity, 0.0, 1.0)),
            lesion_count=args.lesion_count,
            lesion_len_um=args.lesion_len_um,
            seed=args.seed,
            temp_C=args.temp_C,
            tstop_ms=args.tstop_ms,
            dt_ms=args.dt_ms,
            solver=args.solver,
            frequencies_hz=freq_set,
        )

        plot_synchrony(
            severities=sev_sync,
            delta_d_ms=d_sync,
            phase_deg_by_freq={k: v for k, v in phase_sync.items()},
            output_path=out_dir / f"{args.label}_{args.preset}_synchrony.png",
            title=f"{args.preset}: synchrony vs demyelination severity",
        )

        sync_payload = {
            "severities": sev_sync.tolist(),
            "delta_d_ms": [float(v) if np.isfinite(v) else None for v in d_sync],
            "phase_deg": {
                f"{f:g}": [float(v) if np.isfinite(v) else None for v in vals]
                for f, vals in phase_sync.items()
            },
        }

    summary = {
        "preset": args.preset,
        "axon_type": args.axon_type,
        "temperature_C": float(args.temp_C),
        "severity": float(np.clip(args.demyelination_severity, 0.0, 1.0)),
        "healthy": {
            "velocity_m_s": float(healthy.conduction_velocity_m_s) if np.isfinite(healthy.conduction_velocity_m_s) else None,
            "arrival_ms": float(healthy.arrival_time_ms) if np.isfinite(healthy.arrival_time_ms) else None,
            "blocked": bool(healthy.blocked),
            "sodium_charge_pC": float(healthy.sodium_charge_pC),
        },
        "disease": {
            "velocity_m_s": float(disease.conduction_velocity_m_s) if np.isfinite(disease.conduction_velocity_m_s) else None,
            "arrival_ms": float(disease.arrival_time_ms) if np.isfinite(disease.arrival_time_ms) else None,
            "blocked": bool(disease.blocked),
            "sodium_charge_pC": float(disease.sodium_charge_pC),
            "mean_m": float(np.mean(profile.m)),
            "mean_a": float(np.mean(profile.a)),
            "mean_r": float(np.mean(profile.r)),
            "mean_p": float(np.mean(profile.p)),
        },
        "block_probability": {
            "temperatures_C": [float(t) for t in temps],
            "severities": severity_sweep.tolist(),
            "matrix": block_prob.tolist(),
        },
        "synchrony": sync_payload,
        "bundle": None,
    }

    if bundle_stats is not None:
        finite = bundle_stats.arrival_times_ms[np.isfinite(bundle_stats.arrival_times_ms)]
        summary["bundle"] = {
            "fiber_count": int(bundle_stats.arrival_times_ms.size),
            "block_fraction": float(np.mean(bundle_stats.blocked_mask.astype(float))),
            "arrival_mean_ms": float(np.mean(finite)) if finite.size else None,
            "arrival_std_ms": float(np.std(finite)) if finite.size else None,
        }

    with (out_dir / f"{args.label}_{args.preset}_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
