# SimCLR-GPNP-Habitat-Suitability

Code repository for habitat suitability modeling in Giant Panda National Park (GPNP) using self-supervised contrastive learning (SimCLR) and ecologically constrained background sampling.

## Overview

This repository contains the code associated with the study:

**A self-supervised contrastive learning framework for giant panda habitat suitability modeling with an ecologically constrained background sampling strategy**

The workflow includes:

* Background point generation using random and MCMC-based sampling strategies
* Self-supervised contrastive learning (SimCLR) for environmental representation learning
* Habitat suitability modeling (HSM)
* Model evaluation and comparison
* Habitat suitability mapping

## Repository Structure

```text
simclr-gpnp-habitat-suitability/
│
├── code/
│   ├── sampling/
│   │   ├── MCMC_points_generate.py
│   │   ├── Random_points_generate.py
│   │   ├── README_MCMC.md
│   │   └── README_Random.md
│   │
│   └── hsm/
│       ├── Simclr.py
│       ├── Maxent.py
│       └── README.md
│
├── data/
│   ├── background/
│   │   ├── MCMC_samples/
│   │   └── random_samples/
│   │
│   ├── environmental/
│   └── occurrence/
│
└── results/
    ├── Maxent_evaluation_mapping/
    └── Simclr_evaluation_mapping/
```

## Data Availability

Environmental predictor datasets are archived on Zenodo:

https://zenodo.org/records/19308437

The repository also includes the environmental table used for model development and evaluation.

Original giant panda occurrence coordinates are not publicly distributed because they contain sensitive location information for a threatened species.

## Citation

If you use this repository, please cite the associated publication and the Zenodo dataset.

## License

MIT License
