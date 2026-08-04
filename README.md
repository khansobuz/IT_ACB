# IT-ACB: Information-Theoretic Adversarial Cancelable Biometrics

## Overview

This repository provides the official implementation of **IT-ACB**, an information-theoretic adversarial framework for secure and privacy-preserving cancelable biometric template protection.

Cancelable biometric template protection aims to protect biometric identities by generating secure, revocable, and unlinkable representations while maintaining recognition performance. The proposed IT-ACB framework is evaluated on both palmprint and face biometric recognition tasks.

---


## Framework Overview

<img width="2319" height="1014" alt="main" src="https://github.com/user-attachments/assets/10945a4a-1758-4440-980b-35d282e77919" />

**Overview of the proposed IT-ACB framework for secure cancelable biometric template protection.**


## Datasets

The experiments are conducted on multiple benchmark biometric datasets, including:

- **TJU Palmprint Database**
- **PolyU Palmprint Database**
- **LFW (Labeled Faces in the Wild)**
- **CASIA-WebFace**

The extracted features and processed data are organized in the corresponding folders.

---

## Repository Structure

<pre>
IT_ACB/
│
├── Feature_extraction/
│   └── Feature extraction codes
│
├── LFW127/
│   └── LFW face dataset features
│
├── PolyU_data/
│   └── PolyU palmprint dataset features
│
├── TJU_data/
│   └── TJU palmprint dataset features
│
├── WebFace/
│   └── CASIA-WebFace dataset features
│
└── Proposed/
    └── Implementation of the proposed IT-ACB framework
</pre>
## Requirements

- Python 3.x
- PyTorch
- CUDA (recommended)

Install the required packages:

```bash
pip install -r requirements.txt
Usage
1. Feature Extraction

The feature extraction codes are provided in:

Feature_extraction/

Extracted biometric features should be placed into the corresponding dataset folders.

2. Run IT-ACB

The proposed framework is available in:

Proposed/

Follow the provided scripts to train and evaluate the model.
