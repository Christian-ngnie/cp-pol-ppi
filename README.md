# CP-POL + PPI: Conformal Guarantees in Partially-Observed Label Space

This repository contains the code to reproduce all experiments from the paper:

> **CP-POL + PPI: Conformal Guarantees in Partially-Observed Label Space**  
> Christian NGNIE  
> *Transactions on Machine Learning Research (TMLR), 2025*

## Abstract

We study Conformal Prediction (CP) in the practical regime where labeled training/calibration data cover only a subset of the label space. We introduce CP-POL, a pipeline combining split CP with a calibrated novelty test and Prediction‑Powered Inference (PPI). This repository reproduces all synthetic and real‑world experiments (CIFAR‑100) from the paper.

## Requirements

- Python 3.9+
- PyTorch 2.0+
- torchvision
- numpy, scipy, pandas, matplotlib, seaborn, tqdm, scikit‑learn
- (Optional) CUDA for GPU acceleration

Install with:

```bash
pip install -r requirements.txt
