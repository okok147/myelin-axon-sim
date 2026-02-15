"""Pure-Python myelinated axon compartment model.

This module implements a runnable cable model with active nodes of Ranvier and
passive myelinated internodes. It is designed to remain lightweight while still
capturing core biophysical mechanisms relevant to saltatory conduction:

- Node-to-node active regeneration via HH-style Na/K conductances.
- Passive internodal electrotonic spread with myelin-modified Rm/Cm.
- Demyelination/remyelination perturbations for conduction speed and reliability.

All internal calculations use SI units. User-facing helpers/CLI convert from
common neuroscience units (um, ms, mV).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
from scipy.integrate import solve_ivp

UM_TO_M = 1e-6
MS_TO_S = 1e-3
MV_TO_V = 1e-3
S_TO_MS = 1e3
M_TO_UM = 1e6
V_TO_MV = 1e3


@dataclass
class GeometryParams:
    """Geometric axon properties.

    Attributes
    ----------
    n_nodes:
        Number of nodes of Ranvier.
    internode_length_um:
        Length between neighboring nodes.
    node_length_um:
        Node of Ranvier length.
    diameter_um:
        Axon outer diameter (single diameter approximation).
    internode_segments:
        Number of passive compartments per internode.
    """

    n_nodes: int = 21
    internode_length_um: float = 500.0
    node_length_um: float = 1.0
    diameter_um: float = 10.0
    internode_segments: int = 6

    def __post_init__(self) -> None:
        if self.n_nodes < 3:
            raise ValueError("n_nodes must be >= 3 for propagation metrics.")
        if self.internode_segments < 1:
            raise ValueError("internode_segments must be >= 1.")
        if self.internode_length_um <= 0 or self.node_length_um <= 0 or self.diameter_um <= 0:
            raise ValueError("All geometric lengths/diameter must be > 0.")


@dataclass
class ChannelParams:
    """Hodgkin-Huxley channel parameters for active compartments.

    Conductances follow the classical HH convention in mS/cm^2 and are
    converted to SI (S/m^2) internally.
    """

    gna_bar_mS_cm2: float = 3000.0
    gk_bar_mS_cm2: float = 500.0
    gl_node_mS_cm2: float = 1.0
    ena_mV: float = 50.0
    ek_mV: float = -77.0
    el_mV: float = -54.4
    q10: float = 3.0
    reference_temp_c: float = 6.3
    simulation_temp_c: float = 37.0

    @property
    def gna_bar_s_m2(self) -> float:
        return self.gna_bar_mS_cm2 * 10.0

    @property
    def gk_bar_s_m2(self) -> float:
        return self.gk_bar_mS_cm2 * 10.0

    @property
    def gl_node_s_m2(self) -> float:
        return self.gl_node_mS_cm2 * 10.0


@dataclass
class BiophysicalParams:
    """Passive membrane and axial electrical properties in SI units."""

    rm_myelin_ohm_m2: float = 30.0
    cm_myelin_f_m2: float = 2e-4
    rm_node_ohm_m2: float = 0.33
    cm_node_f_m2: float = 1e-2
    ra_ohm_m: float = 0.7
    resting_potential_mV: float = -65.0
    channels: ChannelParams = field(default_factory=ChannelParams)

    def __post_init__(self) -> None:
        if (
            self.rm_myelin_ohm_m2 <= 0
            or self.cm_myelin_f_m2 <= 0
            or self.rm_node_ohm_m2 <= 0
            or self.cm_node_f_m2 <= 0
            or self.ra_ohm_m <= 0
        ):
            raise ValueError("Biophysical Rm/Cm/Ra values must be > 0.")


@dataclass
class StimulusParams:
    """Current-clamp stimulation protocol."""

    amplitude_nA: float = 2.0
    start_ms: float = 0.2
    duration_ms: float = 0.2
    target_node: int = 0

    def __post_init__(self) -> None:
        if self.duration_ms <= 0:
            raise ValueError("Stimulus duration must be > 0 ms.")


@dataclass
class PathologyParams:
    """Myelin pathology state.

    state:
        - "myelinated": baseline internodes
        - "demyelinated": Rm decreases and Cm increases by demyelination_factor
        - "remyelinated": partial restoration from demyelinated toward baseline
    """

    state: str = "myelinated"
    demyelination_factor: float = 1.0
    remyelination_fraction: float = 0.5
    remyelination_internode_scale: float = 0.8
    exposed_nav_fraction: float = 0.0

    def __post_init__(self) -> None:
        valid = {"myelinated", "demyelinated", "remyelinated"}
        if self.state not in valid:
            raise ValueError(f"state must be one of {sorted(valid)}")
        if self.demyelination_factor < 1.0:
            raise ValueError("demyelination_factor must be >= 1.")
        if not (0.0 <= self.remyelination_fraction <= 1.0):
            raise ValueError("remyelination_fraction must be in [0, 1].")
        if not (0.0 < self.remyelination_internode_scale <= 1.0):
            raise ValueError("remyelination_internode_scale must be in (0, 1].")
        if not (0.0 <= self.exposed_nav_fraction <= 1.0):
            raise ValueError("exposed_nav_fraction must be in [0, 1].")


@dataclass
class SimulationResult:
    """Container for simulation outputs and derived metrics."""

    time_ms: np.ndarray
    x_um: np.ndarray
    vm_mV: np.ndarray
    node_indices: np.ndarray
    node_crossings_ms: np.ndarray
    conduction_velocity_m_s: float
    sodium_charge_pC: float
    metadata: Dict[str, float | str]

    @property
    def node_positions_um(self) -> np.ndarray:
        return self.x_um[self.node_indices]

    @property
    def arrival_time_ms(self) -> float:
        return float(self.node_crossings_ms[-1]) if np.isfinite(self.node_crossings_ms[-1]) else np.nan


class MyelinatedAxonModel:
    """Multi-compartment myelinated axon model (pure Python)."""

    def __init__(
        self,
        geometry: Optional[GeometryParams] = None,
        biophys: Optional[BiophysicalParams] = None,
        stimulus: Optional[StimulusParams] = None,
        pathology: Optional[PathologyParams] = None,
    ) -> None:
        self.geometry = geometry or GeometryParams()
        self.biophys = biophys or BiophysicalParams()
        self.stimulus = stimulus or StimulusParams()
        self.pathology = pathology or PathologyParams()

        if self.stimulus.target_node < 0 or self.stimulus.target_node >= self.geometry.n_nodes:
            raise ValueError("Stimulus target_node is outside node range.")

        self._build_compartments()

    def _resolve_myelin_state(self) -> Tuple[float, float, float, float]:
        """Resolve internode properties under current pathology.

        Returns
        -------
        rm_internode, cm_internode, internode_length_scale, exposed_channel_fraction
        """

        rm_base = self.biophys.rm_myelin_ohm_m2
        cm_base = self.biophys.cm_myelin_f_m2

        if self.pathology.state == "myelinated":
            return rm_base, cm_base, 1.0, 0.0

        factor = self.pathology.demyelination_factor
        rm_demyel = rm_base / factor
        cm_demyel = cm_base * factor

        if self.pathology.state == "demyelinated":
            return rm_demyel, cm_demyel, 1.0, self.pathology.exposed_nav_fraction

        restore = self.pathology.remyelination_fraction
        rm_remyel = rm_demyel + restore * (rm_base - rm_demyel)
        cm_remyel = cm_demyel + restore * (cm_base - cm_demyel)
        length_scale = self.pathology.remyelination_internode_scale
        exposed = self.pathology.exposed_nav_fraction * (1.0 - restore)
        return rm_remyel, cm_remyel, length_scale, exposed

    def _build_compartments(self) -> None:
        geom = self.geometry
        bio = self.biophys
        ch = bio.channels

        rm_internode, cm_internode, internode_scale, exposed_fraction = self._resolve_myelin_state()

        node_len_m = geom.node_length_um * UM_TO_M
        internode_len_m = geom.internode_length_um * internode_scale * UM_TO_M
        internode_seg_len_m = internode_len_m / geom.internode_segments
        diameter_m = geom.diameter_um * UM_TO_M

        lengths: List[float] = []
        is_node: List[bool] = []
        node_indices: List[int] = []

        for node_id in range(geom.n_nodes):
            node_indices.append(len(lengths))
            lengths.append(node_len_m)
            is_node.append(True)
            if node_id < geom.n_nodes - 1:
                for _ in range(geom.internode_segments):
                    lengths.append(internode_seg_len_m)
                    is_node.append(False)

        self.n_compartments = len(lengths)
        self.lengths_m = np.asarray(lengths, dtype=float)
        self.is_node = np.asarray(is_node, dtype=bool)
        self.node_indices = np.asarray(node_indices, dtype=int)

        x_positions = np.cumsum(self.lengths_m) - 0.5 * self.lengths_m
        self.x_positions_m = x_positions

        area_m2 = np.pi * diameter_m * self.lengths_m
        self.area_m2 = area_m2
        self.cross_section_m2 = np.pi * (diameter_m * 0.5) ** 2

        self.cm_density_f_m2 = np.where(self.is_node, bio.cm_node_f_m2, cm_internode)
        self.rm_density_ohm_m2 = np.where(self.is_node, bio.rm_node_ohm_m2, rm_internode)
        self.capacitance_f = self.cm_density_f_m2 * self.area_m2

        # Node leak uses the larger of explicit HH leak and specific Rm-derived leak.
        node_leak = max(1.0 / bio.rm_node_ohm_m2, ch.gl_node_s_m2)
        self.gleak_density_s_m2 = np.where(self.is_node, node_leak, 1.0 / rm_internode)

        self.gna_density_s_m2 = np.where(
            self.is_node,
            ch.gna_bar_s_m2,
            ch.gna_bar_s_m2 * exposed_fraction,
        )
        self.gk_density_s_m2 = np.where(
            self.is_node,
            ch.gk_bar_s_m2,
            ch.gk_bar_s_m2 * 0.25 * exposed_fraction,
        )

        half_r = self.biophys.ra_ohm_m * (0.5 * self.lengths_m) / self.cross_section_m2
        self.axial_conductance_s = 1.0 / (half_r[:-1] + half_r[1:])

        self.ena_v = ch.ena_mV * MV_TO_V
        self.ek_v = ch.ek_mV * MV_TO_V
        self.el_v = ch.el_mV * MV_TO_V
        self.v_rest_v = self.biophys.resting_potential_mV * MV_TO_V

        self.q10_scale = ch.q10 ** ((ch.simulation_temp_c - ch.reference_temp_c) / 10.0)

    @staticmethod
    def _vtrap(x: np.ndarray, y: float) -> np.ndarray:
        out = np.empty_like(x)
        ratio = np.clip(x / y, -60.0, 60.0)
        small = np.abs(ratio) < 1e-6
        out[~small] = x[~small] / np.expm1(ratio[~small])
        out[small] = y * (1.0 - 0.5 * ratio[small])
        return out

    def _hh_rates_per_s(self, vm_mV: np.ndarray) -> Tuple[np.ndarray, ...]:
        """Return HH alpha/beta rates in 1/s for m, h, n gates."""

        v = np.clip(vm_mV, -120.0, 120.0) + 65.0
        alpha_m = 0.1 * self._vtrap(25.0 - v, 10.0)
        beta_m = 4.0 * np.exp(np.clip(-v / 18.0, -60.0, 60.0))

        alpha_h = 0.07 * np.exp(np.clip(-v / 20.0, -60.0, 60.0))
        beta_h = 1.0 / (np.exp(np.clip((30.0 - v) / 10.0, -60.0, 60.0)) + 1.0)

        alpha_n = 0.01 * self._vtrap(10.0 - v, 10.0)
        beta_n = 0.125 * np.exp(np.clip(-v / 80.0, -60.0, 60.0))

        scale = self.q10_scale * 1000.0  # ms^-1 to s^-1
        return (
            alpha_m * scale,
            beta_m * scale,
            alpha_h * scale,
            beta_h * scale,
            alpha_n * scale,
            beta_n * scale,
        )

    def initial_state(self) -> np.ndarray:
        vm0_mV = np.full(self.n_compartments, self.biophys.resting_potential_mV, dtype=float)
        am, bm, ah, bh, an, bn = self._hh_rates_per_s(vm0_mV)
        m0 = am / (am + bm)
        h0 = ah / (ah + bh)
        n0 = an / (an + bn)

        v0 = vm0_mV * MV_TO_V
        return np.concatenate((v0, m0, h0, n0))

    def _stimulus_current_a(self, t_s: float) -> np.ndarray:
        stim = np.zeros(self.n_compartments, dtype=float)
        start = self.stimulus.start_ms * MS_TO_S
        stop = (self.stimulus.start_ms + self.stimulus.duration_ms) * MS_TO_S
        if start <= t_s <= stop:
            stim_index = self.node_indices[self.stimulus.target_node]
            stim[stim_index] = self.stimulus.amplitude_nA * 1e-9
        return stim

    def _rhs(self, t_s: float, y: np.ndarray) -> np.ndarray:
        n = self.n_compartments
        vm = np.clip(y[:n], -0.12, 0.08)
        m = np.clip(y[n : 2 * n], 0.0, 1.0)
        h = np.clip(y[2 * n : 3 * n], 0.0, 1.0)
        ng = np.clip(y[3 * n :], 0.0, 1.0)

        vm_mV = vm * V_TO_MV
        am, bm, ah, bh, an, bn = self._hh_rates_per_s(vm_mV)

        dm = am * (1.0 - m) - bm * m
        dh = ah * (1.0 - h) - bh * h
        dn = an * (1.0 - ng) - bn * ng

        gna = self.gna_density_s_m2 * (m**3) * h
        gk = self.gk_density_s_m2 * (ng**4)
        gleak = self.gleak_density_s_m2

        i_ion_density = gna * (vm - self.ena_v) + gk * (vm - self.ek_v) + gleak * (vm - self.el_v)
        i_ion = i_ion_density * self.area_m2

        i_axial = np.zeros(n, dtype=float)
        dv = vm[1:] - vm[:-1]
        edge_flux = self.axial_conductance_s * dv
        i_axial[:-1] += edge_flux
        i_axial[1:] -= edge_flux

        i_stim = self._stimulus_current_a(t_s)
        dvm = (i_axial - i_ion + i_stim) / self.capacitance_f

        return np.concatenate((dvm, dm, dh, dn))

    def _stability_sanity(self, dt_ms: float) -> None:
        """Basic dt sanity check against shortest passive membrane time constant."""

        tau_passive_s = np.min(self.rm_density_ohm_m2 * self.cm_density_f_m2)
        max_dt_ms = tau_passive_s * S_TO_MS * 0.35
        if dt_ms > max_dt_ms:
            raise ValueError(
                f"dt={dt_ms:.4f} ms is likely unstable for this parameter set; "
                f"use dt <= {max_dt_ms:.4f} ms."
            )

    def _rk4_integrate(self, dt_s: float, tstop_s: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        times = np.arange(0.0, tstop_s + 0.5 * dt_s, dt_s)
        n_steps = times.size
        n = self.n_compartments

        max_internal_dt_s = 1e-7  # 0.0001 ms for stiff nodal dynamics.
        n_substeps = max(1, int(np.ceil(dt_s / max_internal_dt_s)))
        dt_internal_s = dt_s / n_substeps

        y = self.initial_state()
        vm = np.zeros((n_steps, n), dtype=float)
        ina_abs_total = np.zeros(n_steps, dtype=float)

        current_t = 0.0
        for i in range(n_steps):
            v = y[:n]
            m = y[n : 2 * n]
            h = y[2 * n : 3 * n]
            ina_density = self.gna_density_s_m2 * (m**3) * h * (v - self.ena_v)

            vm[i] = v * V_TO_MV
            ina_abs_total[i] = np.sum(np.abs(ina_density * self.area_m2))

            if i == n_steps - 1:
                continue

            for _ in range(n_substeps):
                k1 = self._rhs(current_t, y)
                k2 = self._rhs(current_t + 0.5 * dt_internal_s, y + 0.5 * dt_internal_s * k1)
                k3 = self._rhs(current_t + 0.5 * dt_internal_s, y + 0.5 * dt_internal_s * k2)
                k4 = self._rhs(current_t + dt_internal_s, y + dt_internal_s * k3)

                y = y + (dt_internal_s / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
                y[:n] = np.clip(y[:n], -0.12, 0.08)
                y[n:] = np.clip(y[n:], 0.0, 1.0)
                current_t += dt_internal_s

        return times, vm, ina_abs_total

    def _solve_ivp_integrate(self, dt_s: float, tstop_s: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        times = np.arange(0.0, tstop_s + 0.5 * dt_s, dt_s)
        n = self.n_compartments

        sol = solve_ivp(
            self._rhs,
            t_span=(0.0, tstop_s),
            y0=self.initial_state(),
            method="BDF",
            t_eval=times,
            rtol=1e-5,
            atol=1e-8,
        )
        if not sol.success:
            raise RuntimeError(f"solve_ivp failed: {sol.message}")

        y = sol.y
        vm = y[:n].T * V_TO_MV
        m = y[n : 2 * n].T
        h = y[2 * n : 3 * n].T
        v = y[:n].T

        ina_density = self.gna_density_s_m2[None, :] * (m**3) * h * (v - self.ena_v)
        ina_abs_total = np.sum(np.abs(ina_density * self.area_m2[None, :]), axis=1)
        return times, vm, ina_abs_total

    @staticmethod
    def threshold_crossing_times_ms(
        time_ms: np.ndarray,
        vm_mV: np.ndarray,
        compartment_indices: Iterable[int],
        threshold_mV: float = 0.0,
    ) -> np.ndarray:
        """First upward threshold crossing times using linear interpolation."""

        out: List[float] = []
        for idx in compartment_indices:
            trace = vm_mV[:, int(idx)]
            above = trace >= threshold_mV
            crossings = np.where((~above[:-1]) & above[1:])[0]
            if crossings.size == 0:
                out.append(np.nan)
                continue

            i = int(crossings[0])
            t0, t1 = time_ms[i], time_ms[i + 1]
            v0, v1 = trace[i], trace[i + 1]
            if np.isclose(v1, v0):
                out.append(float(t1))
            else:
                frac = (threshold_mV - v0) / (v1 - v0)
                out.append(float(t0 + frac * (t1 - t0)))
        return np.asarray(out, dtype=float)

    @staticmethod
    def enforce_contiguous_propagation(node_crossings_ms: np.ndarray) -> np.ndarray:
        """Keep only strictly increasing contiguous node crossings.

        If propagation fails at node i, all downstream nodes are marked NaN. This
        removes non-physiologic late crossings from reflected/noisy activity.
        """

        cleaned = np.full_like(node_crossings_ms, np.nan, dtype=float)
        prev = -np.inf
        for i, value in enumerate(node_crossings_ms):
            if not np.isfinite(value):
                break
            if value <= prev:
                break
            cleaned[i] = value
            prev = value
        return cleaned

    @staticmethod
    def conduction_velocity_m_s(
        node_positions_um: np.ndarray,
        node_crossings_ms: np.ndarray,
        min_node_index: int = 1,
    ) -> float:
        """Estimate velocity from first/last valid threshold crossings."""

        valid = np.where(np.isfinite(node_crossings_ms))[0]
        valid = valid[valid >= min_node_index]
        if valid.size < 2:
            return np.nan

        i0 = int(valid[0])
        i1 = int(valid[-1])
        dt_s = (node_crossings_ms[i1] - node_crossings_ms[i0]) * MS_TO_S
        if dt_s <= 0:
            return np.nan

        dx_m = (node_positions_um[i1] - node_positions_um[i0]) * UM_TO_M
        return float(dx_m / dt_s)

    def run(self, dt_ms: float = 0.002, tstop_ms: float = 8.0, solver: str = "rk4") -> SimulationResult:
        """Run the compartment simulation and return traces + derived metrics."""

        self._stability_sanity(dt_ms)
        dt_s = dt_ms * MS_TO_S
        tstop_s = tstop_ms * MS_TO_S

        if solver == "rk4":
            time_s, vm_mV, ina_abs = self._rk4_integrate(dt_s, tstop_s)
        elif solver == "solve_ivp":
            time_s, vm_mV, ina_abs = self._solve_ivp_integrate(dt_s, tstop_s)
        else:
            raise ValueError("solver must be 'rk4' or 'solve_ivp'.")

        time_ms = time_s * S_TO_MS
        crossings_ms_raw = self.threshold_crossing_times_ms(time_ms, vm_mV, self.node_indices)
        crossings_ms = self.enforce_contiguous_propagation(crossings_ms_raw)
        velocity = self.conduction_velocity_m_s(self.x_positions_m[self.node_indices] * M_TO_UM, crossings_ms)

        sodium_charge_c = np.trapezoid(ina_abs, time_s)
        sodium_charge_pc = float(sodium_charge_c * 1e12)

        metadata = {
            "solver": solver,
            "state": self.pathology.state,
            "demyelination_factor": float(self.pathology.demyelination_factor),
            "internode_scale": float(self._resolve_myelin_state()[2]),
        }

        return SimulationResult(
            time_ms=time_ms,
            x_um=self.x_positions_m * M_TO_UM,
            vm_mV=vm_mV,
            node_indices=self.node_indices.copy(),
            node_crossings_ms=crossings_ms,
            conduction_velocity_m_s=velocity,
            sodium_charge_pC=sodium_charge_pc,
            metadata=metadata,
        )


def phase_difference(delay_ms: float, frequency_hz: float) -> Tuple[float, float]:
    """Return phase difference in radians and degrees for delay at given frequency."""

    omega = 2.0 * np.pi * frequency_hz
    phi_rad = omega * (delay_ms * MS_TO_S)
    phi_deg = np.degrees(phi_rad)
    return float(phi_rad), float(phi_deg)


def compare_energy_proxy(myelinated: SimulationResult, demyelinated: SimulationResult) -> Dict[str, float]:
    """Return energy proxy comparison dictionary."""

    ratio = np.nan
    if myelinated.sodium_charge_pC > 0:
        ratio = demyelinated.sodium_charge_pC / myelinated.sodium_charge_pC

    return {
        "myelinated_sodium_charge_pC": float(myelinated.sodium_charge_pC),
        "demyelinated_sodium_charge_pC": float(demyelinated.sodium_charge_pC),
        "energy_ratio_demyelinated_over_myelinated": float(ratio),
    }
