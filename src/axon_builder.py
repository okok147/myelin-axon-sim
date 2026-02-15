"""Axon compartment builder and simulator.

Educational/research simulator for saltatory conduction in myelinated fibers.
This is not a clinical/diagnostic model.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Tuple

import numpy as np
from scipy.integrate import solve_ivp

UM_TO_M = 1e-6
MS_TO_S = 1e-3
MV_TO_V = 1e-3
M_TO_UM = 1e6
S_TO_MS = 1e3
V_TO_MV = 1e3


@dataclass
class AxonGeometry:
    """Axon geometry in user-friendly units."""

    n_nodes: int = 21
    internode_length_um: float = 900.0
    node_length_um: float = 1.0
    diameter_um: float = 8.0
    internode_segments: int = 6

    def __post_init__(self) -> None:
        if self.n_nodes < 3:
            raise ValueError("n_nodes must be >= 3")
        if self.internode_segments < 1:
            raise ValueError("internode_segments must be >= 1")
        if (
            self.internode_length_um <= 0
            or self.node_length_um <= 0
            or self.diameter_um <= 0
        ):
            raise ValueError("Geometry values must be > 0")


@dataclass
class ChannelParams:
    """HH-like channel parameters in nodal membrane."""

    gna_bar_mS_cm2: float = 3000.0
    gk_bar_mS_cm2: float = 500.0
    gl_node_mS_cm2: float = 1.2
    ena_mV: float = 55.0
    ek_mV: float = -90.0
    el_mV: float = -70.0
    q10: float = 3.0
    reference_temp_c: float = 6.3

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
class MembraneParams:
    """Passive membrane and axial properties in SI units."""

    rm_myelin_ohm_m2: float
    cm_myelin_f_m2: float
    rm_node_ohm_m2: float
    cm_node_f_m2: float
    ra_ohm_m: float
    resting_potential_mV: float = -70.0


@dataclass
class StimulusParams:
    """Current clamp stimulation."""

    amplitude_nA: float = 2.0
    start_ms: float = 0.2
    duration_ms: float = 0.2
    target_node: int = 0


@dataclass
class IntegrityProfile:
    """Spatial integrity fields sampled at one disease stage.

    m: myelin integrity in [0,1]
    a: axonal integrity in [0,1]
    r: node remodeling in [0,1]
    p: paranodal/periaxonal seal integrity in [0,1]
    s: support/homeostasis field in [0,1]
    """

    m: np.ndarray
    a: np.ndarray
    r: np.ndarray
    p: np.ndarray
    s: np.ndarray

    @classmethod
    def healthy(cls, n_compartments: int) -> "IntegrityProfile":
        ones = np.ones(n_compartments, dtype=float)
        zeros = np.zeros(n_compartments, dtype=float)
        return cls(m=ones.copy(), a=ones.copy(), r=zeros, p=ones.copy(), s=ones.copy())


@dataclass
class SimulationOutput:
    """Simulation traces and derived metrics."""

    time_ms: np.ndarray
    x_um: np.ndarray
    vm_mV: np.ndarray
    node_indices: np.ndarray
    node_crossings_ms: np.ndarray
    conduction_velocity_m_s: float
    sodium_charge_pC: float
    blocked: bool
    metadata: Dict[str, float | str]

    @property
    def arrival_time_ms(self) -> float:
        return float(self.node_crossings_ms[-1]) if np.isfinite(self.node_crossings_ms[-1]) else np.nan


def default_membrane_params(axon_type: str) -> MembraneParams:
    """CNS/PNS passive parameter presets."""

    axon_type_u = axon_type.upper()
    if axon_type_u == "CNS":
        return MembraneParams(
            rm_myelin_ohm_m2=30.0,
            cm_myelin_f_m2=2e-4,
            rm_node_ohm_m2=0.33,
            cm_node_f_m2=1e-2,
            ra_ohm_m=0.7,
            resting_potential_mV=-70.0,
        )
    if axon_type_u == "PNS":
        return MembraneParams(
            rm_myelin_ohm_m2=45.0,
            cm_myelin_f_m2=1.5e-4,
            rm_node_ohm_m2=0.33,
            cm_node_f_m2=1e-2,
            ra_ohm_m=0.6,
            resting_potential_mV=-70.0,
        )
    raise ValueError("axon_type must be CNS or PNS")


def default_geometry(axon_type: str) -> AxonGeometry:
    axon_type_u = axon_type.upper()
    if axon_type_u == "CNS":
        return AxonGeometry(
            n_nodes=21,
            internode_length_um=900.0,
            node_length_um=1.0,
            diameter_um=8.0,
            internode_segments=6,
        )
    if axon_type_u == "PNS":
        return AxonGeometry(
            n_nodes=21,
            internode_length_um=1200.0,
            node_length_um=1.2,
            diameter_um=10.0,
            internode_segments=6,
        )
    raise ValueError("axon_type must be CNS or PNS")


class AxonSimulator:
    """1D multi-compartment myelinated axon simulator."""

    def __init__(
        self,
        geometry: AxonGeometry,
        membrane: MembraneParams,
        channels: ChannelParams,
        stimulus: StimulusParams,
    ) -> None:
        self.geometry = geometry
        self.membrane = membrane
        self.channels = channels
        self.stimulus = stimulus

        if self.stimulus.target_node < 0 or self.stimulus.target_node >= self.geometry.n_nodes:
            raise ValueError("Stimulus target node is out of range.")

        self._build_mesh()

    def _build_mesh(self) -> None:
        g = self.geometry

        node_len_m = g.node_length_um * UM_TO_M
        internode_seg_len_m = (g.internode_length_um / g.internode_segments) * UM_TO_M
        diameter_m = g.diameter_um * UM_TO_M

        lengths: List[float] = []
        is_node: List[bool] = []
        node_indices: List[int] = []

        for i in range(g.n_nodes):
            node_indices.append(len(lengths))
            lengths.append(node_len_m)
            is_node.append(True)
            if i < g.n_nodes - 1:
                for _ in range(g.internode_segments):
                    lengths.append(internode_seg_len_m)
                    is_node.append(False)

        self.lengths_m = np.asarray(lengths, dtype=float)
        self.is_node = np.asarray(is_node, dtype=bool)
        self.node_indices = np.asarray(node_indices, dtype=int)
        self.n_compartments = int(self.lengths_m.size)

        self.x_positions_m = np.cumsum(self.lengths_m) - 0.5 * self.lengths_m

        diameter_m_arr = np.full(self.n_compartments, diameter_m)
        self.area_m2 = np.pi * diameter_m_arr * self.lengths_m
        self.cross_area_m2 = np.pi * (diameter_m_arr * 0.5) ** 2

        half_r = self.membrane.ra_ohm_m * (0.5 * self.lengths_m) / self.cross_area_m2
        self.base_axial_conductance_s = 1.0 / (half_r[:-1] + half_r[1:])

        self.ena_v = self.channels.ena_mV * MV_TO_V
        self.ek_v = self.channels.ek_mV * MV_TO_V
        self.el_v = self.channels.el_mV * MV_TO_V

    @staticmethod
    def _vtrap(x: np.ndarray, y: float) -> np.ndarray:
        ratio = np.clip(x / y, -60.0, 60.0)
        out = np.empty_like(ratio)
        small = np.abs(ratio) < 1e-6
        out[~small] = x[~small] / np.expm1(ratio[~small])
        out[small] = y * (1.0 - 0.5 * ratio[small])
        return out

    def _hh_rates_per_s(self, vm_mV: np.ndarray, temp_C: float) -> Tuple[np.ndarray, ...]:
        v = np.clip(vm_mV, -120.0, 120.0) + 65.0

        alpha_m = 0.1 * self._vtrap(25.0 - v, 10.0)
        beta_m = 4.0 * np.exp(np.clip(-v / 18.0, -60.0, 60.0))

        alpha_h = 0.07 * np.exp(np.clip(-v / 20.0, -60.0, 60.0))
        beta_h = 1.0 / (np.exp(np.clip((30.0 - v) / 10.0, -60.0, 60.0)) + 1.0)

        alpha_n = 0.01 * self._vtrap(10.0 - v, 10.0)
        beta_n = 0.125 * np.exp(np.clip(-v / 80.0, -60.0, 60.0))

        q10_scale = self.channels.q10 ** ((temp_C - self.channels.reference_temp_c) / 10.0)
        scale = 1000.0 * q10_scale
        return (
            alpha_m * scale,
            beta_m * scale,
            alpha_h * scale,
            beta_h * scale,
            alpha_n * scale,
            beta_n * scale,
        )

    def initial_state(self) -> np.ndarray:
        vm0_mV = np.full(self.n_compartments, self.membrane.resting_potential_mV, dtype=float)
        am, bm, ah, bh, an, bn = self._hh_rates_per_s(vm0_mV, temp_C=37.0)
        m0 = am / (am + bm)
        h0 = ah / (ah + bh)
        n0 = an / (an + bn)
        return np.concatenate((vm0_mV * MV_TO_V, m0, h0, n0))

    def _stimulus_a(self, t_s: float) -> np.ndarray:
        out = np.zeros(self.n_compartments, dtype=float)
        t0 = self.stimulus.start_ms * MS_TO_S
        t1 = (self.stimulus.start_ms + self.stimulus.duration_ms) * MS_TO_S
        if t0 <= t_s <= t1:
            out[self.node_indices[self.stimulus.target_node]] = self.stimulus.amplitude_nA * 1e-9
        return out

    def _effective_fields(self, integrity: IntegrityProfile) -> IntegrityProfile:
        def clip01(arr: np.ndarray) -> np.ndarray:
            return np.clip(np.asarray(arr, dtype=float), 0.0, 1.0)

        if (
            integrity.m.shape[0] != self.n_compartments
            or integrity.a.shape[0] != self.n_compartments
            or integrity.r.shape[0] != self.n_compartments
            or integrity.p.shape[0] != self.n_compartments
            or integrity.s.shape[0] != self.n_compartments
        ):
            raise ValueError("Integrity field arrays must match n_compartments")

        m = clip01(integrity.m)
        a = clip01(integrity.a)
        r = clip01(integrity.r)
        p = clip01(integrity.p)
        s = clip01(integrity.s)

        # Nodes sense adjacent myelin/support states.
        for idx in self.node_indices:
            left = max(0, idx - 1)
            right = min(self.n_compartments - 1, idx + 1)
            m[idx] = 0.5 * (m[left] + m[right])
            p[idx] = 0.5 * (p[left] + p[right])
            s[idx] = 0.5 * (s[left] + s[right])

        return IntegrityProfile(m=m, a=a, r=r, p=p, s=s)

    def _build_run_parameters(
        self,
        integrity: IntegrityProfile,
        temp_C: float,
    ) -> Dict[str, np.ndarray | float]:
        fields = self._effective_fields(integrity)
        m, a, r, p, s = fields.m, fields.a, fields.r, fields.p, fields.s

        # Internodal membrane transforms from myelin integrity.
        rm_internode = self.membrane.rm_myelin_ohm_m2 * (0.06 + 0.94 * m**2)
        cm_internode = self.membrane.cm_myelin_f_m2 * (1.0 + 8.0 * (1.0 - m))

        cm_density = np.where(self.is_node, self.membrane.cm_node_f_m2, cm_internode)
        cap_f = cm_density * self.area_m2

        node_leak_density = max(1.0 / self.membrane.rm_node_ohm_m2, self.channels.gl_node_s_m2)
        leak_density = np.where(self.is_node, node_leak_density, 1.0 / rm_internode)

        paranode_leak_multiplier = 1.0 + 5.0 * (1.0 - p)
        support_leak_multiplier = 1.0 + 2.0 * (1.0 - s)
        leak_density *= paranode_leak_multiplier * support_leak_multiplier

        # Node channel remodeling + Uhthoff-like temperature stress in low-myelin zones.
        local_stress = (1.0 - m) * (1.0 + 0.6 * (1.0 - s))
        uhthoff_penalty = np.exp(-0.035 * max(temp_C - 37.0, 0.0) * local_stress)

        gna_density = np.zeros(self.n_compartments, dtype=float)
        gk_density = np.zeros(self.n_compartments, dtype=float)

        node_scale = (0.65 + 0.35 * a) * (1.0 + 0.45 * r) * uhthoff_penalty
        gna_density[self.is_node] = self.channels.gna_bar_s_m2 * node_scale[self.is_node]
        gk_density[self.is_node] = self.channels.gk_bar_s_m2 * (0.7 + 0.3 * a[self.is_node]) * (1.0 + 0.2 * r[self.is_node])

        # Optional internodal channel redistribution in severe demyelination.
        internodal_exposure = np.clip((1.0 - m) * (0.15 + 0.85 * r), 0.0, 1.0)
        gna_density[~self.is_node] = self.channels.gna_bar_s_m2 * 0.04 * internodal_exposure[~self.is_node]
        gk_density[~self.is_node] = self.channels.gk_bar_s_m2 * 0.02 * internodal_exposure[~self.is_node]

        # Axonal integrity and paranodes reduce effective axial coupling.
        edge_a = 0.5 * (a[:-1] + a[1:])
        edge_p = 0.5 * (p[:-1] + p[1:])
        edge_scale = (0.25 + 0.75 * edge_a**2) * (0.45 + 0.55 * edge_p)
        axial_conductance = self.base_axial_conductance_s * edge_scale

        return {
            "m": m,
            "a": a,
            "r": r,
            "p": p,
            "s": s,
            "cap_f": cap_f,
            "leak_density": leak_density,
            "gna_density": gna_density,
            "gk_density": gk_density,
            "axial_conductance": axial_conductance,
            "temp_C": float(temp_C),
        }

    def _rhs(
        self,
        t_s: float,
        y: np.ndarray,
        run_params: Dict[str, np.ndarray | float],
    ) -> np.ndarray:
        n = self.n_compartments
        vm = np.clip(y[:n], -0.12, 0.08)
        m = np.clip(y[n : 2 * n], 0.0, 1.0)
        h = np.clip(y[2 * n : 3 * n], 0.0, 1.0)
        ng = np.clip(y[3 * n :], 0.0, 1.0)

        vm_mV = vm * V_TO_MV
        am, bm, ah, bh, an, bn = self._hh_rates_per_s(vm_mV, temp_C=float(run_params["temp_C"]))

        dm = am * (1.0 - m) - bm * m
        dh = ah * (1.0 - h) - bh * h
        dn = an * (1.0 - ng) - bn * ng

        gna_density = np.asarray(run_params["gna_density"])
        gk_density = np.asarray(run_params["gk_density"])
        leak_density = np.asarray(run_params["leak_density"])
        cap_f = np.asarray(run_params["cap_f"])

        gna = gna_density * (m**3) * h
        gk = gk_density * (ng**4)

        i_ion_density = gna * (vm - self.ena_v) + gk * (vm - self.ek_v) + leak_density * (vm - self.el_v)
        i_ion = i_ion_density * self.area_m2

        i_ax = np.zeros(n, dtype=float)
        dv = vm[1:] - vm[:-1]
        edge_flux = np.asarray(run_params["axial_conductance"]) * dv
        i_ax[:-1] += edge_flux
        i_ax[1:] -= edge_flux

        i_stim = self._stimulus_a(t_s)
        dvm = (i_ax - i_ion + i_stim) / cap_f

        return np.concatenate((dvm, dm, dh, dn))

    def run(
        self,
        integrity: IntegrityProfile,
        temp_C: float = 37.0,
        tstop_ms: float = 8.0,
        dt_ms: float = 0.002,
        solver: str = "solve_ivp",
        threshold_mV: float = 0.0,
    ) -> SimulationOutput:
        run_params = self._build_run_parameters(integrity=integrity, temp_C=temp_C)
        tstop_s = tstop_ms * MS_TO_S
        dt_s = dt_ms * MS_TO_S
        t_eval = np.arange(0.0, tstop_s + 0.5 * dt_s, dt_s)

        y0 = self.initial_state()

        if solver == "solve_ivp":
            sol = solve_ivp(
                fun=lambda t, y: self._rhs(t, y, run_params),
                t_span=(0.0, tstop_s),
                y0=y0,
                method="BDF",
                t_eval=t_eval,
                rtol=1e-5,
                atol=1e-8,
            )
            if not sol.success:
                raise RuntimeError(f"solve_ivp failed: {sol.message}")
            y_mat = sol.y.T
            time_s = sol.t
        elif solver == "rk4":
            y = y0.copy()
            y_rows = [y.copy()]
            time_s = [0.0]
            t_curr = 0.0
            while t_curr < tstop_s - 1e-15:
                h = min(dt_s, tstop_s - t_curr)
                k1 = self._rhs(t_curr, y, run_params)
                k2 = self._rhs(t_curr + 0.5 * h, y + 0.5 * h * k1, run_params)
                k3 = self._rhs(t_curr + 0.5 * h, y + 0.5 * h * k2, run_params)
                k4 = self._rhs(t_curr + h, y + h * k3, run_params)
                y = y + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
                y[: self.n_compartments] = np.clip(y[: self.n_compartments], -0.12, 0.08)
                y[self.n_compartments :] = np.clip(y[self.n_compartments :], 0.0, 1.0)
                t_curr += h
                y_rows.append(y.copy())
                time_s.append(t_curr)

            y_mat = np.asarray(y_rows)
            time_s = np.asarray(time_s)
        else:
            raise ValueError("solver must be solve_ivp or rk4")

        n = self.n_compartments
        vm_v = y_mat[:, :n]
        m = y_mat[:, n : 2 * n]
        h = y_mat[:, 2 * n : 3 * n]

        vm_mV = vm_v * V_TO_MV
        time_ms = time_s * S_TO_MS

        gna_density = np.asarray(run_params["gna_density"])
        ina_density = gna_density[None, :] * (m**3) * h * (vm_v - self.ena_v)
        ina_abs_total = np.sum(np.abs(ina_density * self.area_m2[None, :]), axis=1)
        sodium_charge_pc = float(np.trapezoid(ina_abs_total, time_s) * 1e12)

        crossings = threshold_crossings_ms(
            time_ms=time_ms,
            vm_mV=vm_mV,
            indices=self.node_indices,
            threshold_mV=threshold_mV,
        )
        crossings = enforce_contiguous_node_crossings(crossings)

        velocity = conduction_velocity_m_s(
            node_positions_um=self.x_positions_m[self.node_indices] * M_TO_UM,
            node_crossings_ms=crossings,
        )

        blocked = not np.isfinite(crossings[-1])

        metadata = {
            "solver": solver,
            "temp_C": float(temp_C),
            "mean_myelin": float(np.mean(integrity.m)),
            "mean_axon_integrity": float(np.mean(integrity.a)),
        }

        return SimulationOutput(
            time_ms=time_ms,
            x_um=self.x_positions_m * M_TO_UM,
            vm_mV=vm_mV,
            node_indices=self.node_indices.copy(),
            node_crossings_ms=crossings,
            conduction_velocity_m_s=velocity,
            sodium_charge_pC=sodium_charge_pc,
            blocked=blocked,
            metadata=metadata,
        )


def threshold_crossings_ms(
    time_ms: np.ndarray,
    vm_mV: np.ndarray,
    indices: Iterable[int],
    threshold_mV: float,
) -> np.ndarray:
    out: List[float] = []
    for idx in indices:
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


def enforce_contiguous_node_crossings(node_crossings_ms: np.ndarray) -> np.ndarray:
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


def conduction_velocity_m_s(node_positions_um: np.ndarray, node_crossings_ms: np.ndarray) -> float:
    valid = np.where(np.isfinite(node_crossings_ms))[0]
    if valid.size < 2:
        return np.nan
    i0 = int(valid[0])
    i1 = int(valid[-1])
    dt_s = (node_crossings_ms[i1] - node_crossings_ms[i0]) * MS_TO_S
    if dt_s <= 0:
        return np.nan
    dx_m = (node_positions_um[i1] - node_positions_um[i0]) * UM_TO_M
    return float(dx_m / dt_s)


def phase_misalignment(delay_ms: float, freq_hz: float) -> Tuple[float, float]:
    """Return phase offset (rad, deg) for delay at frequency."""

    phi_rad = 2.0 * np.pi * freq_hz * delay_ms * MS_TO_S
    return float(phi_rad), float(np.degrees(phi_rad))
