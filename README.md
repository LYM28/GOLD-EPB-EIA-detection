# GOLD-EPB-EIA-detection

Deep learning-based detection and segmentation of equatorial plasma bubbles (EPBs) and equatorial ionization anomalies (EIAs) from GOLD airglow observations.

## Overview

This repository provides the data processing, deep learning model training, prediction, and evaluation codes used for the automatic detection of equatorial ionospheric irregularities from observations of the Global-scale Observations of the Limb and Disk (GOLD) mission.

The study focuses on the detection and segmentation of equatorial plasma bubbles (EPBs) and equatorial ionization anomalies (EIAs) in GOLD airglow images centered on the O I 135.6 nm emission.

The repository includes implementations based on U-Net, DeepLabV3+, and Feature Pyramid Network (FPN) for automatic identification of ionospheric irregular structures.

## Models

The following deep learning models are included:

- U-Net
- DeepLabV3+
- Feature Pyramid Network (FPN)

These models are used for semantic segmentation of ionospheric irregular structures in GOLD airglow images.

## Dataset

The annotated GOLD airglow image dataset used in this study is publicly available on Zenodo:

https://doi.org/10.5281/zenodo.22705373

The dataset contains GOLD airglow images and corresponding annotation masks for EPB and EIA detection.

The dataset is organized into training/validation and independent testing subsets:

- 2021–2023: 917 image-mask pairs for model training and validation
- 2024: 595 image-mask pairs for independent testing

The dataset contains four directories:

```text
GOLD-EPB-EIA-Dataset/
├── 21-23image/
├── 21-23mask/
├── 24image/
└── 24mask/
