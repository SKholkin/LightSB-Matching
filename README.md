# Light and Optimal Schrödinger Bridge Matching

This is the official PyTorch implementation of the pre-print [Light and Optimal Schrödinger Bridge Matching](https://arxiv.org/abs/2402.03207) by Nikita Gushchin, Sergei Kholkin, Evgeny Burnaev, Alexander Korotin. 

![image](alae_transfer.png)

## Installation

python=3.10

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

