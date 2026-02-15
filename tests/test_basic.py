"""Basic regression tests for the myelinated axon simulator."""

from __future__ import annotations

import unittest

import numpy as np

from src.axon_model import (
    BiophysicalParams,
    GeometryParams,
    MyelinatedAxonModel,
    PathologyParams,
    StimulusParams,
)


class AxonSimulationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.geometry = GeometryParams(
            n_nodes=8,
            internode_length_um=500.0,
            node_length_um=1.0,
            diameter_um=8.0,
            internode_segments=5,
        )
        self.biophys = BiophysicalParams(
            rm_myelin_ohm_m2=30.0,
            cm_myelin_f_m2=2e-4,
            rm_node_ohm_m2=0.33,
            cm_node_f_m2=1e-2,
            ra_ohm_m=0.7,
        )
        self.stimulus = StimulusParams(amplitude_nA=2.2, start_ms=0.2, duration_ms=0.2, target_node=0)
        self.dt_ms = 0.002
        self.tstop_ms = 6.0

    def _run(self, pathology: PathologyParams):
        model = MyelinatedAxonModel(
            geometry=self.geometry,
            biophys=self.biophys,
            stimulus=self.stimulus,
            pathology=pathology,
        )
        return model.run(dt_ms=self.dt_ms, tstop_ms=self.tstop_ms, solver="solve_ivp")

    def test_action_potential_propagates_across_nodes(self) -> None:
        result = self._run(PathologyParams(state="myelinated"))
        valid = np.isfinite(result.node_crossings_ms)
        propagated_nodes = int(np.sum(valid))
        self.assertGreaterEqual(
            propagated_nodes,
            6,
            msg=f"Expected AP to propagate across at least 6 nodes, got {propagated_nodes}",
        )

    def test_demyelination_reduces_conduction_velocity(self) -> None:
        baseline = self._run(PathologyParams(state="myelinated"))
        demyel = self._run(
            PathologyParams(
                state="demyelinated",
                demyelination_factor=4.0,
                exposed_nav_fraction=0.08,
            )
        )

        self.assertTrue(np.isfinite(baseline.conduction_velocity_m_s))
        if np.isfinite(demyel.conduction_velocity_m_s):
            self.assertLess(
                demyel.conduction_velocity_m_s,
                baseline.conduction_velocity_m_s,
                msg="Demyelination should reduce conduction velocity.",
            )
        else:
            # Conduction block is also a valid slowing phenotype.
            self.assertTrue(np.isnan(demyel.conduction_velocity_m_s))

    def test_arrival_delay_increases_with_demyelination(self) -> None:
        baseline = self._run(PathologyParams(state="myelinated"))
        self.assertTrue(np.isfinite(baseline.arrival_time_ms))

        factors = [1.0, 1.05, 1.1, 1.2]
        delays = []

        for factor in factors:
            if np.isclose(factor, 1.0):
                result = self._run(PathologyParams(state="myelinated"))
            else:
                result = self._run(
                    PathologyParams(
                        state="demyelinated",
                        demyelination_factor=float(factor),
                        exposed_nav_fraction=0.0,
                    )
                )
            self.assertTrue(np.isfinite(result.arrival_time_ms), msg=f"No arrival for factor={factor}")
            delays.append(result.arrival_time_ms - baseline.arrival_time_ms)

        diffs = np.diff(np.asarray(delays))
        self.assertTrue(np.all(diffs >= -0.03), msg=f"Expected near-monotonic delay growth, got delays={delays}")


if __name__ == "__main__":
    unittest.main()
