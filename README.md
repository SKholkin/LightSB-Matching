# Light and Optimal Schrdinger Bridge Matching

PyTorch implementation.

![image](alae_transfer.png)

## Installation

Install Entropic Optimal Transport Benchmark from [link](https://github.com/ngushchin/EntropicOTBenchmark/) (see their instructions)

Install project requirements

```pip install -r requirements.txt```

For ALAE experiments install ALAE requirements

```pip install -r ALAE/requirements.txt```

## Repository structure:

```ALAE``` - Code for the ALAE model.

```src``` - LightSBM implementation with discrete optimal transport.

```notebooks``` - Jupyter notebooks with experiments for LightSBM.

### LightSBM

```notebooks/LightSBM_EOT.ipynb``` - code for EOT Benchmark problems.

```notebooks/LightSBM_MSCI.ipynb``` - code for single cell data analysis problems.

```notebooks/LightSBM_swiss_roll.ipynb``` - code for Swiss Roll experiments.

```notebooks/LightSBM_ALAE.ipynb``` - Code for image experiments with ALAE.

