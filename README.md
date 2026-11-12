# Towards a Foundation Model for Type IIB Flux Vacua

Code for the paper **[Towards a foundation model for Type IIB flux vacua](https://arxiv.org/abs/XXXX.XXXXX)**
by S. Krippendorf, M. Larfors, Z. Liu, A. Schachner, and M. Walden.

## Overview

We investigate generative models as a framework for sampling Type IIB flux vacua across multiple Calabi–Yau geometries and for directing searches towards specified physical properties. We compare **transformers** and **discrete masked diffusion models** trained to generate quantised fluxes together with approximate values of the complex structure moduli and axio-dilaton. Training proceeds in stages: pre-training on a multi-geometry dataset, followed by post-training with Group Relative Policy Optimisation (GRPO), and optionally fine-tuning on new geometries. Both models support conditional generation targeting prescribed values of $N_\text{flux}$, $|W_0|$, and $g_s$.

## Dataset

All models are trained on a dataset of flux vacua across **20 Calabi–Yau geometries** with $h^{1,2} \in \{2,3,4,5\}$, containing $10^5$ distinct vacua per geometry from [Ebelt et al. (2023)](https://arxiv.org/abs/2307.15749).

| $h^{1,2}$ | $(h^{1,2},\, h^{1,1})$ pairs | Flux dimension |
|:---:|:---|:---:|
| 2 | (2, 120), (2, 128), (2, 132), (2, 144), (2, 272) | 6 |
| 3 | (3, 99), (3, 107), (3, 115), (3, 119), (3, 243) | 8 |
| 4 | (4, 98), (4, 118), (4, 122), (4, 130), (4, 142) | 10 |
| 5 | (5, 89), (5, 149), (5, 153), (5, 165), (5, 213) | 12 |

Pre-training uses all 20 geometries ($h^{1,2} \in \{2,3,4,5\}$). Transfer-learning experiments pre-train on $h^{1,2} \in \{2,3,4\}$ (15 geometries) and fine-tune on $h^{1,2}=5$ (5 geometries, 1,000 vacua each).

## Models

### Transformer

An encoder–decoder transformer that takes the Calabi–Yau geometry (and optionally physical targets) as input and autoregressively generates a sequence of tokens representing the flux vector $\mathbf{F} = (\mathbf{f}, \mathbf{h})$ and initial guesses $(\mathbf{z}_0, \tau_0)$ for the complex-structure moduli and axio-dilaton.

| Model | Training | Description |
|:---|:---:|:---|
| **Pre-trained** | all 20 geometries | Baseline multi-geometry generation |
| **Pre-trained + GRPO** | all 20 geometries | Post-training with Group Relative Policy Optimisation; rewards valid, novel solutions |
| **Benchmark for fine-tuning** | $h^{1,2}=5$ only | Trained from scratch on 1,000 vacua per $h^{1,2}=5$ geometry |
| **Pre-trained → Fine-tuned** | $h^{1,2}\in\{2,3,4\}$ then $h^{1,2}=5$ | Transfer learning: pre-trained model fine-tuned on 1,000 vacua per $h^{1,2}=5$ geometry |

### Diffusion Model

An absorbing-state discrete masked diffusion model using a bidirectional transformer to reconstruct masked flux coordinates from partially unmasked configurations. A separate initial-guess generator predicts the continuous moduli $(\mathbf{z}_0, \tau_0)$ from the generated flux vector.

| Model | Training | Description |
|:---|:---:|:---|
| **Pre-trained** | all 20 geometries | Baseline multi-geometry generation |
| **Pre-trained + GRPO** | all 20 geometries | Post-training with Group Diffusion Policy Optimisation |
| **Benchmark for fine-tuning** | $h^{1,2}=5$ only | Trained from scratch on 1,000 vacua per $h^{1,2}=5$ geometry |
| **Pre-trained → Fine-tuned** | $h^{1,2}\in\{2,3,4\}$ then $h^{1,2}=5$ | Transfer learning: pre-trained model fine-tuned on 1,000 vacua per $h^{1,2}=5$ geometry |

### Conditional Generation

Both model families support conditioning on physical observables to direct the search towards vacua with prescribed properties:

| Observable | Symbol | Verified |
|:---|:---:|:---|
| Flux-induced D3-brane charge | $N_\text{flux}$ | Before numerical solving (exact) |
| Vacuum superpotential magnitude | $\|W_0\|$ | After numerical solving |
| Inverse string coupling | $1/g_s$ | After numerical solving |

---

## Notebooks

### Transformer

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/StringJAX/towards_type_IIB_foundation_model/blob/main/transformer/sample.ipynb) **`transformer/sample.ipynb`** *(GPU)*

Samples flux vacua from all transformer variants studied in the paper:
- Pre-trained transformer (all 20 geometries)
- Pre-trained + GRPO (post-trained with Group Relative Policy Optimisation)
- Fine-tuned from scratch on $h^{1,2}=5$ geometries (1,000 vacua each)
- Pre-trained on $h^{1,2}\in\{2,3,4\}$ → fine-tuned on $h^{1,2}=5$ geometries
- Conditional generation targeting prescribed values of $N_\text{flux}$, $|W_0|$, and $g_s$

---

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/StringJAX/towards_type_IIB_foundation_model/blob/main/transformer/evaluate.ipynb) **`transformer/evaluate.ipynb`** *(CPU)*

Evaluates all transformer variants using the six-step validation routine from the paper (tadpole, $F$-terms, fundamental domain, LCS control, uniqueness, novelty). Reports validity rates per geometry and per model, and conditional accuracy for $N_\text{flux}$, $|W_0|$, and $g_s$ targets.

---

### Diffusion Model

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/StringJAX/towards_type_IIB_foundation_model/blob/main/diffusion/sample.ipynb) **`diffusion/sample.ipynb`** *(GPU)*

Samples flux vacua from all diffusion model variants studied in the paper:
- Pre-trained diffusion model (all 20 geometries)
- Pre-trained + GRPO (post-trained with Group Diffusion Policy Optimisation)
- Fine-tuned from scratch on $h^{1,2}=5$ geometries (1,000 vacua each)
- Pre-trained on $h^{1,2}\in\{2,3,4\}$ → fine-tuned on $h^{1,2}=5$ geometries
- Conditional generation targeting prescribed values of $N_\text{flux}$, $|W_0|$, and $g_s$

---

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/StringJAX/towards_type_IIB_foundation_model/blob/main/diffusion/evaluate.ipynb) **`diffusion/evaluate.ipynb`** *(CPU)*

Evaluates all diffusion model variants using the same six-step validation routine. Reports validity rates per geometry and per model, and conditional accuracy for $N_\text{flux}$, $|W_0|$, and $g_s$ targets.

---

## Citation

If you use this code, please cite:

```bibtex
@article{Krippendorf:2025xxx,
    author = {Krippendorf, Sven and Larfors, Magdalena and Liu, Zhimei and Schachner, Andreas and Walden, Moritz},
    title = {{Towards a foundation model for Type IIB flux vacua}},
    year = {2025},
    eprint = {XXXX.XXXXX},
    archivePrefix = {arXiv},
}
```
