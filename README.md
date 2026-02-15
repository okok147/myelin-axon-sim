# Myelin / White-Matter Conduction Simulator

Educational and research-oriented simulator for saltatory conduction with
modular demyelinating/white-matter disease signatures.

## Important scope

This project is **not** a medical diagnostic tool. Disease presets are
phenomenological abstractions for mechanistic exploration and education.

## Features

- 1D multi-compartment cable model with repeating units:
  - active nodes of Ranvier (HH-like `INa`, `IK`, `IL`)
  - passive internodes with myelin-dependent `Rm` and `Cm`
  - paranodal/periaxonal seal field for leak/safety-factor effects
- Integrity fields over space/time:
  - `m(x,t)` myelin integrity
  - `a(x,t)` axonal integrity
  - `r(x,t)` node remodeling
  - `p(x,t)` paranodal seal integrity
  - `s(x,t)` support/homeostasis field
- CNS and PNS presets
- Disease modules:
  - `MS`, `NMOSD`, `MOGAD`, `ADEM`, `PML`
  - `ALD`, `KRABBE`, `MLD`
  - `GBS`, `CIDP`, `CMT`
- Metrics:
  - conduction velocity
  - conduction block probability
  - Na-current energy proxy (`∫|INa| dt`)
  - synchrony (`Δd`, `Δφ`)
- Bundle simulation for CAP temporal dispersion (GBS/CIDP/CMT)

## Install

```bash
pip install -r requirements.txt
```

Dependencies are intentionally minimal:

- `numpy`
- `scipy`
- `matplotlib`

## File layout

- `src/axon_builder.py`:
  - compartment mesh, coupling, HH-like dynamics, conduction metrics
- `src/disease_models.py`:
  - disease presets as space-time field generators
- `src/simulate.py`:
  - CLI entry point and experiments
- `src/plotting.py`:
  - traces, heatmap, block probability, synchrony, CAP plots
- `tests/test_basic.py`:
  - propagation, demyelination, temperature, pattern-separation, synchrony tests

## CLI

```bash
python -m src.simulate \
  --preset {MS,NMOSD,MOGAD,ADEM,PML,ALD,KRABBE,MLD,GBS,CIDP,CMT} \
  --axon_type {CNS,PNS} \
  --n_nodes 21 --internode_length_um 900 --node_length_um 1 --diameter_um 8 \
  --temp_C 37 --demyelination_severity 0.7 --lesion_count 4 --lesion_len_um 700 \
  --tstop_ms 8 --dt_ms 0.002 --seed 7 --sync
```

Notes:

- Use `--sync` to run two-pathway arrival/phase alignment analysis.
- Outputs (plots + summary JSON) are written to `--output_dir`.

## Example commands

### MS baseline + block-probability + heatmaps/traces

```bash
python -m src.simulate --preset MS --axon_type CNS --demyelination_severity 0.7 --temp_C 37 --label ms_demo --output_dir outputs
```

### MS synchrony (10 Hz + 40 Hz + user frequency)

```bash
python -m src.simulate --preset MS --axon_type CNS --demyelination_severity 0.7 --sync --freq_hz 20 --label ms_sync --output_dir outputs
```

### GBS CAP dispersion bundle

```bash
python -m src.simulate --preset GBS --axon_type PNS --demyelination_severity 0.8 --label gbs_bundle --output_dir outputs
```

## Tests

```bash
python -m unittest tests/test_basic.py
```

The suite checks:

- baseline propagation across nodes
- demyelination-driven slowing and increased block probability
- Uhthoff-like MS temperature sensitivity
- CIDP/GBS segmental block/dispersion distinction vs uniform CMT slowing
- monotonic synchrony changes (`Δd`, `Δφ`) with severity

## Assumptions and limitations

- SI units are used internally; CLI uses neuroscience-friendly units.
- Disease evolution is represented through parameterized fields, not patient-specific pathology.
- Channel remodeling and lesion dynamics are simplified abstractions.
- No branching, extracellular fields, ephaptic coupling, or full glial network model.

## Mechanistic provenance (paper-backed)

- Hodgkin & Huxley 1952 (action potential dynamics)
  - DOI: `10.1113/jphysiol.1952.sp004764`
- McIntyre, Richardson, Grill 2002 (myelinated axon modeling context)
  - DOI: `10.1152/jn.00353.2001`
- Waxman & Brill 1978 (demyelination, slowing/block)
  - DOI: `10.1136/jnnp.41.5.408`
- Hartline & Colman 2007 (myelin and conduction speed)
  - DOI: `10.1016/j.cub.2006.11.042`
- Salami et al. 2003; Seidl 2014; Pajevic et al. 2014 (delay tuning/synchrony)
  - DOI: `10.1073/pnas.0937380100`
  - DOI: `10.1016/j.neuroscience.2013.06.047`
  - DOI: `10.3389/fncel.2014.00155`
