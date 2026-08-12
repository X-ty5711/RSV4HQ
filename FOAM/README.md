# OpenFOAM_case

**CFD-trained deep-learning framework for bathymetry-free river discharge estimation — OpenFOAM demo case**

[![OpenFOAM](https://img.shields.io/badge/OpenFOAM-8-blue)](https://openfoam.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

This repository provides the **OpenFOAM demo case** that accompanies the paper

> **Discharge estimation from surface velocity without bathymetry: a CFD-trained deep learning approach**
>
> Tianyi Xu, Yuncheng Xu\*, Dongyang He, Xuan Yu
> (*Journal of Hydrology*, in review)

## Abstract

River discharge is fundamental for water-resources management and flood-risk
assessment, yet conventional methods rely on extensive bathymetric surveys and
site-specific calibration. This study proposes a bathymetry-free CFD-driven
deep-learning framework that inverts river discharge using **only
surface-velocity profiles and channel width**. A multi-scale synthetic dataset
was generated via **OpenFOAM** simulations of fully developed subcritical
channel flows, and gradient-based feature enhancement, half-profile adaptation,
cross-section screening, and noise-augmented training were adopted to improve
accuracy and transferability. Validation on the Swindale Beck natural river
reach yielded R² of 0.842 (depth) and 0.790 (mean velocity).

## Repository contents

```
JoH_CFD-trained_case/
├── README.md           # this file
├── LICENSE             # MIT license
├── CITATION.cff        # citation metadata (GitHub "Cite this repository")
└── OpenFOAM_case/      # the OpenFOAM demo case
    └── README.md       # full case documentation (run instructions, parameters, outputs)
```

### OpenFOAM_case/

A single, ready-to-run OpenFOAM 8 case of a fully developed rectangular channel
flow (uniform `0.05 m` mesh, cyclic inlet–outlet, `k-ω SST` turbulence,
`simpleFoam`/SIMPLE). It reproduces one representative CFD simulation of the
paper's database and outputs the **transverse surface-velocity profile** that is
the input to the deep-learning model.

See [`OpenFOAM_case/README.md`](OpenFOAM_case/README.md) for requirements,
quick-start instructions, parameter tables, and how to generate other
configurations.

## Quick start

```bash
source /opt/openfoam8/etc/bashrc
cd OpenFOAM_case
./Allmesh    # reset case and build the mesh
./Allrun     # decompose, run simpleFoam, reconstruct, sample profiles
```

## Citation

```bibtex
@article{xu2026discharge,
  title   = {Discharge estimation from surface velocity without bathymetry:
             a CFD-trained deep learning approach},
  author  = {Xu, Tianyi and Xu, Yuncheng and He, Dongyang and Yu, Xuan},
  journal = {Journal of Hydrology},
  year    = {2026},
  note    = {in review}
}
```

## License

Distributed under the [MIT License](LICENSE).

## Contact

Yuncheng Xu — ycxu@cau.edu.cn

State Key Laboratory of Efficient Utilization of Agricultural Water Resources &
College of Water Resources and Intelligence Engineering, China Agricultural
University, Beijing, China.

Funded by the National Natural Science Foundation of China (52209103) and the
National Key Research and Development Program of China (2023YFE0208200,
2023YFD1900701-01).
