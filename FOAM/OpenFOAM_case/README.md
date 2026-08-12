# OpenFOAM case: fully developed rectangular channel flow (demo)

This is the **OpenFOAM demo case** accompanying the paper

> *Discharge estimation from surface velocity without bathymetry: a CFD-trained
> deep learning approach* (Journal of Hydrology, in review).

The case reproduces one representative CFD simulation used to build the
training database of the paper. It simulates a fully developed,
subcritical open-channel flow in a smooth rectangular channel and outputs the
**transverse surface-velocity profile** — the input of the deep-learning model
that inverts river discharge from surface velocity and channel width alone
(no bathymetry required).

This single case demonstrates the numerical setup; the full database was built
by sweeping the channel width `B`, water depth `H` and cross-sectional mean
velocity `V` (see *Reproducing other cases* below).

---

## 1. Requirements

- **OpenFOAM 8** (openfoam.org release, or a compatible version).
- A Linux environment (native or WSL). The scripts use the standard OpenFOAM
  run functions (`RunFunctions` / `CleanFunctions`).
- `mpiexec`/`mpirun` only if you run in parallel (the default `Allrun` does use
  parallel execution with `decomposePar`).

## 2. Quick start

```bash
# 1. Source the OpenFOAM environment (example for OpenFOAM 8)
source /opt/openfoam8/etc/bashrc

# 2. Build the mesh (also resets the case from 0.orig/)
./Allmesh

# 3. Run the steady simulation and sample the profiles
./Allrun
```

- `./Allclean` removes the mesh, results, post-processing and processor
  directories, returning the case to a pristine state.
- `./rerun` re-runs the parallel solve without rebuilding the mesh (useful
  after changing `constant/Ubar.H`).
- `./postProcess` re-extracts the sampled profiles from the latest time.

> `Allrun` runs in parallel with the number of subdomains set in
> `system/decomposeParDict` (default 8). Adjust it to the number of your cores.

## 3. Outputs

After `Allrun`, the sampled profiles are written under
`postProcessing/sampleLines/<time>/`:

| File  | Content | Purpose |
|-------|---------|---------|
| `line1_*.xy` | Transverse **surface-velocity** profile (spanwise `y`, at `z ≈ H − un/2`) | **Input to the deep-learning model** (paper Secs. 2.1–2.2) |
| `line2_*.xy` | Vertical velocity profile (`z = 0 … H`) | Physical-consistency check against the entropy velocity law (paper Sec. 2.1) |

You can also open the case in ParaView with the bundled marker file:

```bash
paraview test.foam
```

## 4. Case setup (mapping to the paper, Sec. 2.1)

| Setting | Value | Location |
|---------|-------|----------|
| Solver | `simpleFoam` (steady RANS, SIMPLE) | `system/controlDict`, `system/fvSolution` |
| Turbulence model | `kOmegaSST` | `constant/momentumTransport` |
| Cell size | uniform `un = 0.05 m` | `system/blockMeshDict` |
| Streamwise length | single grid layer (`x2 = un`) | `system/blockMeshDict` |
| Channel width `B` | `y2 − y1` (demo: 2 m) | `system/blockMeshDict` |
| Water depth `H` | demo: 2 m | `system/H.H` |
| Mean velocity `V` (`Ubar`) | demo: 2.5 m/s | `constant/Ubar.H` |
| Inlet / outlet | `cyclic` (fully developed flow) | `0/U`, `0/p`, … |
| Bed & side walls | `noSlip` + Spalding wall function (`nutUSpaldingWallFunction`) | `0/U`, `0/nut` |
| Free surface | `symmetryPlane` (free-slip rigid lid) | `0/U` |
| Driving force | `meanVelocityForce` maintains `Ubar` | `constant/fvOptions` |
| Kinematic viscosity | `nu = 1e-6 m²/s` (water) | `constant/transportProperties` |

This setup matches the paper's description: uniform `0.05 m` mesh, cyclic
inlet–outlet enforcing fully developed flow, no-slip smooth walls with the
Spalding wall function, a free-slip rigid lid, and a momentum source that keeps
the target cross-sectional mean velocity.

## 5. Reproducing other cases (dataset parameters)

The paper's database sweeps the following ranges (Table 2):

| Parameter | Symbol | Paper range | Demo value |
|-----------|--------|-------------|------------|
| Cross-sectional mean velocity | `V` | 0.1 – 3.0 m/s | 2.5 m/s |
| Water depth | `H` | 0.3 – 10 m | 2 m |
| Channel width | `B` | 1 – 10 m | 2 m |

To run a different configuration:

1. **Depth `H`** — edit `system/H.H`.
2. **Mean velocity `V`** — edit `constant/Ubar.H`.
3. **Width `B`** — edit `system/Y.H`.
4. **Cell size `un`** (optional) — edit `system/blockMeshDict`.

> **Important:** `system/sampleLines` samples the surface profile at
> `z = H − un/2` (top cell layer). If you change `H` (or `un`), update `z` in
> the `line1` block of `system/sampleLines` accordingly.

The demo value `H = 2 m`, `B = 2 m`, `V = 2.5 m/s` is a representative
single point of the database. Batch generation over the full parameter space
was performed externally; the scripts in this folder run one case at a time.

## 6. File structure

```
OpenFOAM_case/
├── Allrun            # build mesh -> decompose -> run -> reconstruct -> sample
├── Allclean          # remove mesh, results, post-processing
├── Allmesh           # reset case (from 0.orig/) and build blockMesh grid
├── rerun             # re-run the parallel solve without remeshing
├── postProcess       # re-extract sampled profiles
├── test.foam         # ParaView marker file
├── 0/                # initial fields (restored from 0.orig/ by ./Allmesh)
├── 0.orig/           # pristine initial fields
├── constant/
│   ├── transportProperties
│   ├── momentumTransport        # k-omega SST
│   ├── momentumTransport.kOmegaSST  # backup restored by ./Allmesh
│   ├── fvOptions                # meanVelocityForce (maintains Ubar)
│   └── Ubar.H                   # target mean velocity V
├── system/
│   ├── blockMeshDict            # mesh + width B, cell size un
│   ├── H.H                      # water depth H
│   ├── controlDict              # simpleFoam, run controls, functions
│   ├── fvSchemes / fvSolution   # discretisation and SIMPLE solver settings
│   ├── sampleLines              # profile sampling (model inputs)
│   ├── decomposeParDict         # parallel decomposition
│   └── forces                   # wall-force monitoring (informational)
└── _archive/        # unused/legacy files from earlier experiments
```

## 7. Notes and limitations

- The case assumes **hydraulically smooth** walls. Roughness was deliberately
  excluded during model development and is discussed as a limitation in the
  paper (Sec. 2.1, Sec. 4.4).
- The rectangular geometry is a local approximation of natural river reaches
  (wide rectangular channel assumption).
- Files in `_archive/` are not used by the case; they are kept for reference
  only.

## 8. License and citation

Distributed under the MIT License — see the repository root `LICENSE`.

If you use this case or the associated method, please cite:

> Xu, T., Xu, Y., He, D., & Yu, X. (in review). Discharge estimation from
> surface velocity without bathymetry: a CFD-trained deep learning approach.
> *Journal of Hydrology*.
