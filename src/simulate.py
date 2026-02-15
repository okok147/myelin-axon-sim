"""CLI entry point for myelinated axon simulations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

from .axon_model import (
    BiophysicalParams,
    GeometryParams,
    MyelinatedAxonModel,
    PathologyParams,
    SimulationResult,
    StimulusParams,
    compare_energy_proxy,
    phase_difference,
)
from .neuron_backend import neuron_is_available, run_neuron_simulation
from .plotting import (
    plot_demyelination_sweep,
    plot_energy_comparison,
    plot_space_time_heatmap,
    plot_synchrony,
    plot_voltage_traces,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Myelinated axon saltatory conduction simulator (pure Python + optional NEURON)."
    )

    parser.add_argument(
        "--mode",
        choices=["single", "sweep", "synchrony"],
        default="single",
        help="single: one condition; sweep: demyelination sweep; synchrony: two-pathway delay alignment",
    )
    parser.add_argument(
        "--condition",
        choices=["myelinated", "demyelinated", "remyelinated"],
        default="myelinated",
        help="Condition used by --mode single.",
    )

    # Requested geometry arguments
    parser.add_argument("--n_nodes", type=int, default=21)
    parser.add_argument("--internode_length_um", type=float, default=500.0)
    parser.add_argument("--node_length_um", type=float, default=1.0)
    parser.add_argument("--diameter_um", type=float, default=10.0)

    # Requested passive electrical arguments
    parser.add_argument("--rm_myelin", type=float, default=30.0, help="Internode specific Rm (ohm*m^2)")
    parser.add_argument("--cm_myelin", type=float, default=2e-4, help="Internode specific Cm (F/m^2)")
    parser.add_argument("--rm_node", type=float, default=0.33, help="Node specific Rm (ohm*m^2)")
    parser.add_argument("--cm_node", type=float, default=1e-2, help="Node specific Cm (F/m^2)")
    parser.add_argument("--ra", type=float, default=0.7, help="Axial resistivity (ohm*m)")

    parser.add_argument(
        "--demyelination_factor",
        type=float,
        default=1.2,
        help="Demyelination severity factor: internode Rm /= factor, Cm *= factor",
    )
    parser.add_argument(
        "--freq_hz_for_phase",
        type=float,
        default=10.0,
        help="Primary oscillation frequency for phase-difference output",
    )
    parser.add_argument("--dt", type=float, default=0.002, help="Time step (ms)")
    parser.add_argument("--tstop_ms", type=float, default=8.0, help="Total simulation duration (ms)")

    parser.add_argument("--internode_segments", type=int, default=6)
    parser.add_argument("--solver", choices=["rk4", "solve_ivp"], default="solve_ivp")
    parser.add_argument("--backend", choices=["auto", "python", "neuron"], default="auto")
    parser.add_argument("--output_dir", default="outputs")
    parser.add_argument("--label", default="experiment")

    parser.add_argument("--stim_amp_nA", type=float, default=2.0)
    parser.add_argument("--stim_start_ms", type=float, default=0.2)
    parser.add_argument("--stim_duration_ms", type=float, default=0.2)

    parser.add_argument("--sweep_points", type=int, default=7)
    parser.add_argument("--remyelination_fraction", type=float, default=0.55)
    parser.add_argument("--remyelination_internode_scale", type=float, default=0.82)
    parser.add_argument("--exposed_nav_fraction", type=float, default=0.0)

    return parser


def _build_geometry(args: argparse.Namespace) -> GeometryParams:
    return GeometryParams(
        n_nodes=args.n_nodes,
        internode_length_um=args.internode_length_um,
        node_length_um=args.node_length_um,
        diameter_um=args.diameter_um,
        internode_segments=args.internode_segments,
    )


def _build_biophys(args: argparse.Namespace) -> BiophysicalParams:
    return BiophysicalParams(
        rm_myelin_ohm_m2=args.rm_myelin,
        cm_myelin_f_m2=args.cm_myelin,
        rm_node_ohm_m2=args.rm_node,
        cm_node_f_m2=args.cm_node,
        ra_ohm_m=args.ra,
    )


def _build_stimulus(args: argparse.Namespace) -> StimulusParams:
    return StimulusParams(
        amplitude_nA=args.stim_amp_nA,
        start_ms=args.stim_start_ms,
        duration_ms=args.stim_duration_ms,
        target_node=0,
    )


def _build_pathology(condition: str, args: argparse.Namespace, demyel_factor: float | None = None) -> PathologyParams:
    factor = args.demyelination_factor if demyel_factor is None else demyel_factor
    if condition == "myelinated":
        return PathologyParams(state="myelinated", demyelination_factor=1.0)
    if condition == "demyelinated":
        return PathologyParams(
            state="demyelinated",
            demyelination_factor=max(1.0, factor),
            exposed_nav_fraction=args.exposed_nav_fraction,
        )
    if condition == "remyelinated":
        return PathologyParams(
            state="remyelinated",
            demyelination_factor=max(1.0, factor),
            remyelination_fraction=args.remyelination_fraction,
            remyelination_internode_scale=args.remyelination_internode_scale,
            exposed_nav_fraction=args.exposed_nav_fraction,
        )
    raise ValueError(f"Unknown condition '{condition}'")


def _run_model(
    geometry: GeometryParams,
    biophys: BiophysicalParams,
    stimulus: StimulusParams,
    pathology: PathologyParams,
    dt_ms: float,
    tstop_ms: float,
    solver: str,
    backend: str,
) -> Tuple[SimulationResult, str]:
    if backend == "python":
        model = MyelinatedAxonModel(geometry=geometry, biophys=biophys, stimulus=stimulus, pathology=pathology)
        return model.run(dt_ms=dt_ms, tstop_ms=tstop_ms, solver=solver), "python"

    if backend == "neuron":
        return run_neuron_simulation(
            geometry=geometry,
            biophys=biophys,
            stimulus=stimulus,
            pathology=pathology,
            dt_ms=dt_ms,
            tstop_ms=tstop_ms,
        ), "neuron"

    # auto mode
    if neuron_is_available():
        result = run_neuron_simulation(
            geometry=geometry,
            biophys=biophys,
            stimulus=stimulus,
            pathology=pathology,
            dt_ms=dt_ms,
            tstop_ms=tstop_ms,
        )
        return result, "neuron"

    model = MyelinatedAxonModel(geometry=geometry, biophys=biophys, stimulus=stimulus, pathology=pathology)
    return model.run(dt_ms=dt_ms, tstop_ms=tstop_ms, solver=solver), "python"


def _print_result_summary(name: str, result: SimulationResult) -> None:
    arrival = result.arrival_time_ms
    arrival_txt = f"{arrival:.3f} ms" if np.isfinite(arrival) else "no spike at final node"
    vel_txt = (
        f"{result.conduction_velocity_m_s:.3f} m/s"
        if np.isfinite(result.conduction_velocity_m_s)
        else "conduction block"
    )
    print(f"[{name}] velocity={vel_txt}, arrival={arrival_txt}, Na-charge={result.sodium_charge_pC:.2f} pC")


def _save_single_plots(out_dir: Path, label: str, condition: str, result: SimulationResult) -> None:
    stem = f"{label}_{condition}"
    plot_voltage_traces(
        result=result,
        output_path=out_dir / f"{stem}_traces.png",
        title=f"{condition.capitalize()} condition: membrane potentials",
    )
    plot_space_time_heatmap(
        result=result,
        output_path=out_dir / f"{stem}_heatmap.png",
        title=f"{condition.capitalize()} condition: Vm(x,t)",
    )


def run_single(args: argparse.Namespace) -> None:
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    geometry = _build_geometry(args)
    biophys = _build_biophys(args)
    stimulus = _build_stimulus(args)

    pathology = _build_pathology(args.condition, args)
    result, backend_used = _run_model(
        geometry=geometry,
        biophys=biophys,
        stimulus=stimulus,
        pathology=pathology,
        dt_ms=args.dt,
        tstop_ms=args.tstop_ms,
        solver=args.solver,
        backend=args.backend,
    )

    print(f"Backend used: {backend_used}")
    _print_result_summary(args.condition, result)
    _save_single_plots(Path(args.output_dir), args.label, args.condition, result)

    # Required energy proxy comparison against demyelination.
    if args.condition == "myelinated":
        demyel = _build_pathology("demyelinated", args)
        demyel_result, _ = _run_model(
            geometry=geometry,
            biophys=biophys,
            stimulus=stimulus,
            pathology=demyel,
            dt_ms=args.dt,
            tstop_ms=args.tstop_ms,
            solver=args.solver,
            backend=args.backend,
        )
        _print_result_summary("demyelinated(reference)", demyel_result)

        energy = compare_energy_proxy(result, demyel_result)
        print(
            "Energy ratio (demyelinated/myelinated) = "
            f"{energy['energy_ratio_demyelinated_over_myelinated']:.3f}"
        )

        plot_energy_comparison(
            labels=["myelinated", "demyelinated"],
            sodium_charge_pc=[result.sodium_charge_pC, demyel_result.sodium_charge_pC],
            output_path=Path(args.output_dir) / f"{args.label}_energy_proxy.png",
            title="Na-current energy proxy comparison",
        )

        with (Path(args.output_dir) / f"{args.label}_energy_proxy.json").open("w", encoding="utf-8") as f:
            json.dump(energy, f, indent=2)


def run_demyelination_sweep(args: argparse.Namespace) -> None:
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    geometry = _build_geometry(args)
    biophys = _build_biophys(args)
    stimulus = _build_stimulus(args)

    factors = np.linspace(1.0, max(1.0, args.demyelination_factor), args.sweep_points)
    velocities: List[float] = []
    arrivals: List[float] = []

    for factor in factors:
        condition = "myelinated" if np.isclose(factor, 1.0) else "demyelinated"
        pathology = _build_pathology(condition, args, demyel_factor=float(factor))
        result, backend_used = _run_model(
            geometry=geometry,
            biophys=biophys,
            stimulus=stimulus,
            pathology=pathology,
            dt_ms=args.dt,
            tstop_ms=args.tstop_ms,
            solver=args.solver,
            backend=args.backend,
        )
        velocities.append(result.conduction_velocity_m_s)
        arrivals.append(result.arrival_time_ms)
        print(f"factor={factor:.2f} ({backend_used}): velocity={result.conduction_velocity_m_s:.3f} m/s")

    plot_demyelination_sweep(
        factors=factors,
        velocities=velocities,
        arrivals_ms=arrivals,
        output_path=out_dir / f"{args.label}_demyelination_sweep.png",
        title="Demyelination sweep: conduction slowing and delay",
    )

    payload = {
        "factors": factors.tolist(),
        "velocities_m_s": [float(v) for v in velocities],
        "arrivals_ms": [float(v) if np.isfinite(v) else None for v in arrivals],
    }
    with (out_dir / f"{args.label}_demyelination_sweep.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def run_synchrony(args: argparse.Namespace) -> None:
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    geometry = _build_geometry(args)
    biophys = _build_biophys(args)
    stimulus = _build_stimulus(args)

    # Pathway A baseline myelinated.
    path_a = _build_pathology("myelinated", args)
    result_a, backend_used = _run_model(
        geometry=geometry,
        biophys=biophys,
        stimulus=stimulus,
        pathology=path_a,
        dt_ms=args.dt,
        tstop_ms=args.tstop_ms,
        solver=args.solver,
        backend=args.backend,
    )
    print(f"Pathway A backend: {backend_used}")
    _print_result_summary("pathway_A", result_a)

    if not np.isfinite(result_a.arrival_time_ms):
        raise RuntimeError("Pathway A did not reach final node threshold; synchrony analysis aborted.")

    factors = np.linspace(1.0, max(1.0, args.demyelination_factor), args.sweep_points)
    delays_ms: List[float] = []

    freq_set = sorted({10.0, 40.0, float(args.freq_hz_for_phase)})
    phase_deg_by_freq: Dict[float, List[float]] = {f: [] for f in freq_set}

    for factor in factors:
        cond = "myelinated" if np.isclose(factor, 1.0) else "demyelinated"
        path_b = _build_pathology(cond, args, demyel_factor=float(factor))

        result_b, _ = _run_model(
            geometry=geometry,
            biophys=biophys,
            stimulus=stimulus,
            pathology=path_b,
            dt_ms=args.dt,
            tstop_ms=args.tstop_ms,
            solver=args.solver,
            backend=args.backend,
        )

        if np.isfinite(result_b.arrival_time_ms):
            delay = result_b.arrival_time_ms - result_a.arrival_time_ms
        else:
            delay = np.nan

        delays_ms.append(delay)

        for freq in freq_set:
            if np.isfinite(delay):
                _, phase_deg = phase_difference(delay_ms=delay, frequency_hz=freq)
            else:
                phase_deg = np.nan
            phase_deg_by_freq[freq].append(phase_deg)

        print(
            f"Pathway-B factor={factor:.2f}: arrival={result_b.arrival_time_ms:.3f} ms, "
            f"Δd={delay:.3f} ms"
        )

    plot_synchrony(
        factors=factors,
        delays_ms=delays_ms,
        phase_deg_by_freq=phase_deg_by_freq,
        output_path=out_dir / f"{args.label}_synchrony.png",
        title="Arrival-time synchrony and phase alignment",
    )

    payload = {
        "factors": factors.tolist(),
        "delay_ms": [float(v) if np.isfinite(v) else None for v in delays_ms],
        "phase_deg": {
            f"{freq:g}": [float(v) if np.isfinite(v) else None for v in phases]
            for freq, phases in phase_deg_by_freq.items()
        },
    }
    with (out_dir / f"{args.label}_synchrony.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    if args.backend == "neuron" and not neuron_is_available():
        raise RuntimeError("Requested --backend neuron but `neuron` package is not installed.")

    if args.backend == "auto" and not neuron_is_available():
        print("NEURON package not detected; falling back to pure-Python backend.")

    if args.mode == "single":
        run_single(args)
    elif args.mode == "sweep":
        run_demyelination_sweep(args)
    elif args.mode == "synchrony":
        run_synchrony(args)
    else:  # pragma: no cover
        raise ValueError(f"Unsupported mode: {args.mode}")


if __name__ == "__main__":
    main()
