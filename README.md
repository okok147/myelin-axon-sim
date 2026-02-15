# Myelin Axon Simulation

Research-grade, runnable simulation of saltatory conduction in myelinated axons.

## What this repo models

- Multi-compartment axon with repeating units:
  - Nodes of Ranvier (active HH-style Na/K dynamics)
  - Passive internodes with myelin-modified membrane properties (`Rm↑`, `Cm↓`)
- Saltatory conduction with node-to-node regeneration.
- Demyelination/remyelination perturbations and their effect on:
  - conduction velocity
  - reliability/conduction block
  - Na-current energy proxy
- Synchrony/alignment experiment for two converging pathways:
  - arrival-time mismatch `Δd`
  - phase mismatch `Δφ = ω·Δd`

## Installation

```bash
pip install -r requirements.txt
```

Optional high-fidelity backend:

- `neuron` package (auto-detected by CLI)

## Project layout

- `src/axon_model.py`: pure-Python compartment model + metrics
- `src/simulate.py`: CLI runner (single, sweep, synchrony)
- `src/plotting.py`: publication-style figure outputs
- `src/neuron_backend.py`: optional NEURON backend fallback
- `tests/test_basic.py`: propagation/slowing/synchrony tests

## Run examples

### 1) Baseline myelinated conduction

```bash
python -m src.simulate --mode single --condition myelinated --label baseline --output_dir outputs
```

### 2) Demyelination sweep (slowing + possible block)

```bash
python -m src.simulate --mode sweep --demyelination_factor 3 --label demyel_sweep --output_dir outputs
```

### 3) Synchrony experiment (Δd, Δφ at 10 Hz/40 Hz and custom frequency)

```bash
python -m src.simulate --mode synchrony --demyelination_factor 1.2 --freq_hz_for_phase 10 --label synchrony --output_dir outputs
```

### Optional remyelination example

```bash
python -m src.simulate --mode single --condition remyelinated --demyelination_factor 2 --remyelination_fraction 0.6 --label remyelinated --output_dir outputs
```

## CLI parameters (core)

- `--n_nodes`, `--internode_length_um`, `--node_length_um`, `--diameter_um`
- `--rm_myelin`, `--cm_myelin`, `--rm_node`, `--cm_node`, `--ra`
- `--demyelination_factor`, `--freq_hz_for_phase`, `--dt`, `--tstop_ms`

Useful extras: `--mode`, `--condition`, `--backend`, `--internode_segments`, `--solver`.

Notes:

- Internal units are SI; CLI accepts common neuroscience units (`um`, `ms`, `mV`).
- Default solver is `solve_ivp` (`BDF`) for robustness.
- `--solver rk4` is available; use sufficiently small `--dt`.

## NEURON backend behavior

- `--backend auto` (default): use NEURON if installed, otherwise pure Python.
- `--backend python`: force pure-Python model.
- `--backend neuron`: force NEURON (errors if not installed).

## Tests

```bash
python -m unittest tests/test_basic.py
```

Covers:

- AP propagation across multiple nodes
- velocity reduction under demyelination
- monotonic increase in delay mismatch with worsening myelin

## Mechanistic provenance (paper-backed)

- Hodgkin & Huxley 1952 (active spike mechanism)
  - DOI: `10.1113/jphysiol.1952.sp004764`
- McIntyre, Richardson, Grill 2002; Richardson et al. 2000 (myelinated axon modeling/MRG context)
  - DOI: `10.1152/jn.00353.2001`
  - DOI: `10.1007/BF02345014`
- Waxman & Brill 1978 (demyelination and conduction slowing/block)
  - DOI: `10.1136/jnnp.41.5.408`
- Hartline & Colman 2007 (myelin and conduction speed)
  - DOI: `10.1016/j.cub.2006.11.042`
- Salami et al. 2003; Seidl 2014; Pajevic et al. 2014 (delay alignment/synchrony role of myelin)
  - DOI: `10.1073/pnas.0937380100`
  - DOI: `10.1016/j.neuroscience.2013.06.047`
  - DOI: `10.3389/fncel.2014.00155`
- Alle et al. 2009; Harris & Attwell 2012 (Na-entry/energetic interpretation)
  - DOI: `10.1126/science.1174331`
  - DOI: `10.1523/JNEUROSCI.3430-11.2012`

## Limitations

- Not a full molecularly detailed nodal/paranodal model.
- Geometry is a regular linear chain (no branching, no extracellular field model).
- NEURON path is MRG-inspired and intended for cross-checking, not a complete
  finite-impedance double-cable reconstruction of all published variants.
