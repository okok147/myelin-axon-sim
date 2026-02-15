"""Behavioral tests for modular demyelinating-disease simulator."""

from __future__ import annotations

import unittest

import numpy as np

from src.axon_builder import (
    AxonGeometry,
    AxonSimulator,
    ChannelParams,
    IntegrityProfile,
    StimulusParams,
    default_membrane_params,
)
from src.simulate import (
    estimate_block_probability,
    run_single_trial,
    run_synchrony_experiment,
    simulate_bundle_dispersion,
)


class DiseaseSimulatorTests(unittest.TestCase):
    def setUp(self) -> None:
        geometry = AxonGeometry(
            n_nodes=8,
            internode_length_um=500.0,
            node_length_um=1.0,
            diameter_um=8.0,
            internode_segments=5,
        )
        self.sim = AxonSimulator(
            geometry=geometry,
            membrane=default_membrane_params("CNS"),
            channels=ChannelParams(),
            stimulus=StimulusParams(amplitude_nA=2.2, start_ms=0.2, duration_ms=0.2, target_node=0),
        )
        self.dt_ms = 0.002
        self.tstop_ms = 6.0

    def test_baseline_myelinated_propagation(self) -> None:
        out = self.sim.run(
            integrity=IntegrityProfile.healthy(self.sim.n_compartments),
            temp_C=37.0,
            tstop_ms=self.tstop_ms,
            dt_ms=self.dt_ms,
            solver="solve_ivp",
        )
        propagated = int(np.sum(np.isfinite(out.node_crossings_ms)))
        self.assertGreaterEqual(propagated, 6)

    def test_demyelination_slows_and_increases_block_probability(self) -> None:
        healthy, _, _ = run_single_trial(
            simulator=self.sim,
            preset="MS",
            axon_type="CNS",
            severity=0.0,
            lesion_count=4,
            lesion_len_um=650.0,
            seed=21,
            temp_C=37.0,
            tstop_ms=self.tstop_ms,
            dt_ms=self.dt_ms,
            solver="solve_ivp",
        )
        diseased, _, _ = run_single_trial(
            simulator=self.sim,
            preset="MS",
            axon_type="CNS",
            severity=0.8,
            lesion_count=4,
            lesion_len_um=650.0,
            seed=21,
            temp_C=37.0,
            tstop_ms=self.tstop_ms,
            dt_ms=self.dt_ms,
            solver="solve_ivp",
        )

        self.assertTrue(np.isfinite(healthy.conduction_velocity_m_s))
        if np.isfinite(diseased.conduction_velocity_m_s):
            self.assertLess(diseased.conduction_velocity_m_s, healthy.conduction_velocity_m_s)
        else:
            self.assertTrue(diseased.blocked)

        probs = estimate_block_probability(
            simulator=self.sim,
            preset="MS",
            axon_type="CNS",
            severities=[0.0, 0.5, 0.9],
            temperatures_C=[37.0],
            lesion_count=4,
            lesion_len_um=650.0,
            base_seed=111,
            tstop_ms=self.tstop_ms,
            dt_ms=self.dt_ms,
            solver="solve_ivp",
            trials_per_point=4,
        )
        self.assertGreaterEqual(probs[0, -1], probs[0, 0])

    def test_ms_temperature_increases_block_probability(self) -> None:
        probs = estimate_block_probability(
            simulator=self.sim,
            preset="MS",
            axon_type="CNS",
            severities=[0.85],
            temperatures_C=[36.0, 39.0],
            lesion_count=4,
            lesion_len_um=650.0,
            base_seed=222,
            tstop_ms=self.tstop_ms,
            dt_ms=self.dt_ms,
            solver="solve_ivp",
            trials_per_point=5,
        )
        self.assertGreaterEqual(probs[1, 0], probs[0, 0])

    def test_gbs_cidp_vs_cmt_patterns(self) -> None:
        gbs = simulate_bundle_dispersion(
            simulator=self.sim,
            preset="GBS",
            axon_type="PNS",
            severity=0.8,
            lesion_count=4,
            lesion_len_um=700.0,
            seed=333,
            temp_C=37.0,
            tstop_ms=self.tstop_ms,
            dt_ms=self.dt_ms,
            solver="solve_ivp",
            n_fibers=10,
        )
        cidp = simulate_bundle_dispersion(
            simulator=self.sim,
            preset="CIDP",
            axon_type="PNS",
            severity=0.8,
            lesion_count=4,
            lesion_len_um=700.0,
            seed=444,
            temp_C=37.0,
            tstop_ms=self.tstop_ms,
            dt_ms=self.dt_ms,
            solver="solve_ivp",
            n_fibers=10,
        )
        cmt = simulate_bundle_dispersion(
            simulator=self.sim,
            preset="CMT",
            axon_type="PNS",
            severity=0.8,
            lesion_count=4,
            lesion_len_um=700.0,
            seed=555,
            temp_C=37.0,
            tstop_ms=self.tstop_ms,
            dt_ms=self.dt_ms,
            solver="solve_ivp",
            n_fibers=10,
        )

        gbs_block = float(np.mean(gbs.blocked_mask.astype(float)))
        cidp_block = float(np.mean(cidp.blocked_mask.astype(float)))
        cmt_block = float(np.mean(cmt.blocked_mask.astype(float)))

        gbs_arr = gbs.arrival_times_ms[np.isfinite(gbs.arrival_times_ms)]
        cidp_arr = cidp.arrival_times_ms[np.isfinite(cidp.arrival_times_ms)]
        cmt_arr = cmt.arrival_times_ms[np.isfinite(cmt.arrival_times_ms)]

        gbs_std = float(np.std(gbs_arr)) if gbs_arr.size else np.inf
        cidp_std = float(np.std(cidp_arr)) if cidp_arr.size else np.inf
        cmt_std = float(np.std(cmt_arr)) if cmt_arr.size else 0.0

        self.assertGreaterEqual(gbs_block, cmt_block)
        self.assertGreaterEqual(cidp_block, cmt_block)
        self.assertTrue((gbs_std >= cmt_std) or (cidp_std >= cmt_std))

    def test_synchrony_monotonic_with_severity(self) -> None:
        severities, delays_ms, phase = run_synchrony_experiment(
            simulator=self.sim,
            preset="MS",
            axon_type="CNS",
            max_severity=0.45,
            lesion_count=4,
            lesion_len_um=650.0,
            seed=777,
            temp_C=37.0,
            tstop_ms=self.tstop_ms,
            dt_ms=self.dt_ms,
            solver="solve_ivp",
            frequencies_hz=[10.0, 40.0],
        )

        finite = np.isfinite(delays_ms)
        abs_delay = np.abs(delays_ms[finite])
        self.assertGreaterEqual(abs_delay.size, 4)

        delay_diff = np.diff(abs_delay)
        self.assertTrue(np.all(delay_diff >= -0.05), msg=f"abs delay not monotonic: {abs_delay}")

        abs_phase40 = np.abs(phase[40.0][finite])
        phase_diff = np.diff(abs_phase40)
        self.assertTrue(np.all(phase_diff >= -2.5), msg=f"phase not monotonic: {abs_phase40}")


if __name__ == "__main__":
    unittest.main()
