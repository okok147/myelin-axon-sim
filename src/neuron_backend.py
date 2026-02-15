"""Optional NEURON backend.

This backend is auto-detected at runtime. If NEURON is not available, callers
should fall back to the pure-Python solver in `axon_model.py`.
"""

from __future__ import annotations

import numpy as np

from .axon_model import (
    BiophysicalParams,
    GeometryParams,
    MyelinatedAxonModel,
    PathologyParams,
    SimulationResult,
    StimulusParams,
)


def neuron_is_available() -> bool:
    try:
        import neuron  # noqa: F401

        return True
    except Exception:
        return False


def run_neuron_simulation(
    geometry: GeometryParams,
    biophys: BiophysicalParams,
    stimulus: StimulusParams,
    pathology: PathologyParams,
    dt_ms: float,
    tstop_ms: float,
) -> SimulationResult:
    """Run an MRG-inspired compartment chain in NEURON.

    Notes
    -----
    This is a lightweight approximation of node/internode organization using
    built-in `hh` at nodes and `pas` at internodes. It does not fully implement
    all details of a double-cable MRG model but preserves key structural
    myelinated mechanisms and enables cross-checking against the pure-Python
    model when `neuron` is installed.
    """

    try:
        from neuron import h
    except Exception as exc:  # pragma: no cover - exercised only when NEURON absent
        raise RuntimeError("NEURON backend requested but package is unavailable.") from exc

    model_ref = MyelinatedAxonModel(
        geometry=geometry,
        biophys=biophys,
        stimulus=stimulus,
        pathology=pathology,
    )

    h.load_file("stdrun.hoc")

    sections = []
    vm_vectors = []
    ina_vectors = []
    ina_areas_cm2 = []

    ra_ohm_cm = biophys.ra_ohm_m * 100.0
    el_mV = biophys.channels.el_mV

    for i in range(model_ref.n_compartments):
        sec = h.Section(name=f"comp_{i}")
        sec.nseg = 1
        sec.L = float(model_ref.lengths_m[i] * 1e6)
        sec.diam = float(geometry.diameter_um)
        sec.Ra = ra_ohm_cm
        sec.cm = float(model_ref.cm_density_f_m2[i] * 100.0)  # F/m^2 -> uF/cm^2

        if model_ref.is_node[i]:
            sec.insert("hh")
            sec.gnabar_hh = float(model_ref.gna_density_s_m2[i] / 1e4)
            sec.gkbar_hh = float(model_ref.gk_density_s_m2[i] / 1e4)
            sec.gl_hh = float(model_ref.gleak_density_s_m2[i] / 1e4)
            sec.el_hh = float(el_mV)
        else:
            sec.insert("pas")
            sec.g_pas = float(model_ref.gleak_density_s_m2[i] / 1e4)
            sec.e_pas = float(el_mV)

        if i > 0:
            sec.connect(sections[i - 1](1.0), 0.0)

        sections.append(sec)

        v_vec = h.Vector().record(sec(0.5)._ref_v)
        vm_vectors.append(v_vec)

        if model_ref.is_node[i]:
            ina_vec = h.Vector().record(sec(0.5)._ref_ina)
            ina_vectors.append(ina_vec)
            area_cm2 = np.pi * (sec.diam * 1e-4) * (sec.L * 1e-4)
            ina_areas_cm2.append(area_cm2)

    stim_idx = model_ref.node_indices[stimulus.target_node]
    iclamp = h.IClamp(sections[int(stim_idx)](0.5))
    iclamp.delay = float(stimulus.start_ms)
    iclamp.dur = float(stimulus.duration_ms)
    iclamp.amp = float(stimulus.amplitude_nA)

    t_vec = h.Vector().record(h._ref_t)

    h.dt = float(dt_ms)
    h.steps_per_ms = 1.0 / float(dt_ms)
    h.tstop = float(tstop_ms)
    h.v_init = float(biophys.resting_potential_mV)
    h.finitialize(h.v_init)
    h.continuerun(h.tstop)

    time_ms = np.asarray(t_vec, dtype=float)
    vm_mV = np.column_stack([np.asarray(vec, dtype=float) for vec in vm_vectors])

    # Optional Na-charge proxy from node sodium current densities.
    time_s = time_ms * 1e-3
    ina_total_a = np.zeros_like(time_s)
    for vec, area_cm2 in zip(ina_vectors, ina_areas_cm2):
        density_ma_cm2 = np.asarray(vec, dtype=float)
        current_a = density_ma_cm2 * 1e-3 * area_cm2
        ina_total_a += np.abs(current_a)
    sodium_charge_pc = float(np.trapezoid(ina_total_a, time_s) * 1e12)

    crossings_ms = MyelinatedAxonModel.threshold_crossing_times_ms(
        time_ms=time_ms,
        vm_mV=vm_mV,
        compartment_indices=model_ref.node_indices,
        threshold_mV=0.0,
    )
    crossings_ms = MyelinatedAxonModel.enforce_contiguous_propagation(crossings_ms)
    velocity = MyelinatedAxonModel.conduction_velocity_m_s(
        node_positions_um=model_ref.x_positions_m[model_ref.node_indices] * 1e6,
        node_crossings_ms=crossings_ms,
    )

    metadata = {
        "solver": "neuron",
        "state": pathology.state,
        "demyelination_factor": float(pathology.demyelination_factor),
    }

    return SimulationResult(
        time_ms=time_ms,
        x_um=model_ref.x_positions_m * 1e6,
        vm_mV=vm_mV,
        node_indices=model_ref.node_indices.copy(),
        node_crossings_ms=crossings_ms,
        conduction_velocity_m_s=velocity,
        sodium_charge_pC=sodium_charge_pc,
        metadata=metadata,
    )
