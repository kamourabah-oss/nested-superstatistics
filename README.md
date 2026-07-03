# Nested superstatistics: numerical reproducibility files

This repository contains Python scripts and numerical output files associated with the manuscript

**Nested superstatistics: When the fluctuation parameter fluctuates**.

The files reproduce the numerical results for:

- **Fig. 1**: hierarchical Langevin simulations and survival-probability comparisons between the adiabatic reference, slow Langevin dynamics, and fast Langevin dynamics;
- **Fig. 5**: reconstruction of the distribution `g(q)` from a simulated time series of the fast variable.

The observational data used for the solar-wind electron distribution and the AMS-02 cosmic-ray electron spectrum are not redistributed here. They were taken from the public sources cited in the manuscript.

## Repository structure

```text
nested_superstatistics_public_deposit/
├── README.md
├── requirements.txt
├── scripts/
│   ├── make_fig1_langevin_typeA.py
│   └── make_fig5_timeseries_reconstruction.py
└── results/
    ├── fig1_langevin/
    └── fig5_reconstruction/
```

## Requirements

The scripts require Python 3 and the packages listed in `requirements.txt`.

```bash
pip install -r requirements.txt
```

## Reproducing Fig. 1

Run

```bash
python scripts/make_fig1_langevin_typeA.py --output-dir results/fig1_langevin
```

The script performs a type-A-consistent hierarchical Langevin simulation and saves:

```text
results/fig1_langevin/Fig-Langevin.pdf
results/fig1_langevin/Fig-Langevin.png
results/fig1_langevin/Fig-Langevin-data.npz
results/fig1_langevin/Fig-Langevin-parameters.json
```

The `.npz` file contains the plotted survival probabilities and ratios. The `.json` file records the numerical parameters used in the simulation.

## Reproducing Fig. 5

Run

```bash
python scripts/make_fig5_timeseries_reconstruction.py --output-dir results/fig5_reconstruction
```

The script simulates the three-level type-A hierarchy, reconstructs the intermediate inverse-temperature process, infers local values of the superstatistical parameter, constructs the empirical distribution of `q`, and generates the four panels of the time-series reconstruction figure. It saves:

```text
results/fig5_reconstruction/FigTimeSeries_combined.pdf
results/fig5_reconstruction/FigTimeSeries_combined.png
results/fig5_reconstruction/FigTimeSeries_a_flatness.pdf
results/fig5_reconstruction/FigTimeSeries_a_flatness.png
results/fig5_reconstruction/FigTimeSeries_b_autocorr.pdf
results/fig5_reconstruction/FigTimeSeries_b_autocorr.png
results/fig5_reconstruction/FigTimeSeries_c_gq.pdf
results/fig5_reconstruction/FigTimeSeries_c_gq.png
results/fig5_reconstruction/FigTimeSeries_d_ratio.pdf
results/fig5_reconstruction/FigTimeSeries_d_ratio.png
results/fig5_reconstruction/figure_data.npz
results/fig5_reconstruction/parameters.json
results/fig5_reconstruction/summary.json
```

The `.npz` file contains the plotted arrays. The `.json` files summarize the numerical parameters and reconstruction statistics.
