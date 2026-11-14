# Towards a Foundation Model for Type IIB Flux Vacua

Code for the paper **Towards a foundation model for Type IIB flux vacua** (arXiv number to follow) by S. Krippendorf,
M. Larfors, Z. Liu, A. Schachner, and M. Walden.

<!-- docs:overview-start -->
## Overview

We investigate generative models as a framework for sampling Type IIB flux vacua across multiple Calabi–Yau geometries and for directing searches towards specified physical properties. We compare **transformers** and **discrete masked diffusion models** trained to generate quantised fluxes together with approximate values of the complex structure moduli and axio-dilaton. Training proceeds in stages: pre-training on a multi-geometry dataset, followed by post-training with Group Relative Policy Optimisation (GRPO), and optionally fine-tuning on new geometries. Both models support conditional generation targeting prescribed values of $N_\text{flux}$, $|W_0|$, and $g_s$.

<!-- docs:overview-end -->

<!-- docs:dataset-start -->
## Dataset

All models are trained on a dataset of flux vacua across **20 Calabi–Yau geometries** with $h^{1,2} \in \{2,3,4,5\}$, containing $10^5$ distinct vacua per geometry from [Ebelt et al. (2023)](https://arxiv.org/abs/2307.15749).

| $h^{1,2}$ | $(h^{1,2}, h^{1,1})$ pairs |
|:---:|:---|
| 2 | (2, 120), (2, 128), (2, 132), (2, 144), (2, 272) |
| 3 | (3, 99), (3, 107), (3, 115), (3, 119), (3, 243) |
| 4 | (4, 98), (4, 118), (4, 122), (4, 130), (4, 142) |
| 5 | (5, 89), (5, 149), (5, 153), (5, 165), (5, 213) |

Pre-training uses all 20 geometries ($h^{1,2} \in \{2,3,4,5\}$). Transfer-learning experiments pre-train on $h^{1,2} \in \{2,3,4\}$ (15 geometries) and fine-tune on $h^{1,2}=5$ (5 geometries, 1,000 vacua each).

<!-- docs:dataset-end -->

<!-- docs:models-start -->
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
| **Pre-trained + GRPO** | all 20 geometries | Post-training with GRPO, adapted to masked diffusion (see the paper's appendix) |
| **Benchmark for fine-tuning** | $h^{1,2}=5$ only | Trained from scratch on 1,000 vacua per $h^{1,2}=5$ geometry |
| **Pre-trained → Fine-tuned** | $h^{1,2}\in\{2,3,4\}$ then $h^{1,2}=5$ | Transfer learning: pre-trained model fine-tuned on 1,000 vacua per $h^{1,2}=5$ geometry |

### Conditional Generation

Both model families support conditioning on physical observables to direct the search towards vacua with prescribed properties:

| Observable | Symbol |
|:---|:---:|
| Flux-induced D3-brane charge | $N_\text{flux}$ |
| Vacuum superpotential magnitude | $\|W_0\|$ |
| Inverse string coupling | $1/g_s$ |
<!-- docs:models-end -->

---

<!-- docs:installation-start -->
## Installation

Python ≥ 3.12. Some data files are stored with [Git LFS](https://git-lfs.com), so install it before cloning:

```bash
git lfs install
git clone https://github.com/StringJAX/towards_type_IIB_foundation_model.git
cd towards_type_IIB_foundation_model
pip install -r requirements-cpu.txt   # or requirements-gpu.txt
```

| File | Use |
|:---|:---|
| `requirements-cpu.txt` | CPU on Linux and macOS: evaluation, and sampling without a GPU |
| `requirements-gpu.txt` | Sampling on a Linux GPU with CUDA 13, as in the Colab sample notebooks |
| `requirements.txt` | The same as `requirements-gpu.txt` |

> **JAX on CPU.** jax/jaxlib 0.8.0–0.12.0.1 can miscompile some JAXVacua functions on CPU without raising an error
> ([jax-ml/jax#42589](https://github.com/jax-ml/jax/issues/42589)). In JAXVacua's tests, inverse Kähler metrics,
> scalar potentials, Hessians and mass matrices were affected, while the F-terms and their Jacobian were not. The
> evaluation here (F-terms, their Jacobian, the superpotential, the map to the fundamental domain) has not been checked
> separately. To avoid the problem, set
> `XLA_FLAGS=--xla_backend_optimization_level=2` before JAX is imported.
<!-- docs:installation-end -->

---

<!-- docs:notebooks-start -->
## Notebooks

### Transformer

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/StringJAX/towards_type_IIB_foundation_model/blob/main/transformer/sample.ipynb) **`transformer/sample.ipynb`** *(GPU)*

Samples flux vacua from all transformer variants studied in the paper:
- Pre-trained transformer (all 20 geometries)
- Pre-trained + GRPO (post-trained with Group Relative Policy Optimisation)
- Benchmark: trained from scratch on $h^{1,2}=5$ geometries (1,000 vacua each)
- Pre-trained on $h^{1,2}\in\{2,3,4\}$ → fine-tuned on $h^{1,2}=5$ geometries
- Conditional generation targeting prescribed values of $N_\text{flux}$, $|W_0|$, and $g_s$

---

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/StringJAX/towards_type_IIB_foundation_model/blob/main/transformer/evaluate.ipynb) **`transformer/evaluate.ipynb`** *(CPU)*

Evaluates all transformer variants using the validation routine from the paper (tadpole, $F$-terms, fundamental domain, LCS control, uniqueness, novelty). Reports validity rates per geometry and per model, and conditional accuracy for $N_\text{flux}$, $|W_0|$, and $g_s$ targets.

---

### Diffusion Model

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/StringJAX/towards_type_IIB_foundation_model/blob/main/diffusion/sample.ipynb) **`diffusion/sample.ipynb`** *(GPU)*

Samples flux vacua from all diffusion model variants studied in the paper:
- Pre-trained diffusion model (all 20 geometries)
- Pre-trained + GRPO (post-trained with GRPO, adapted to masked diffusion)
- Benchmark: trained from scratch on $h^{1,2}=5$ geometries (1,000 vacua each)
- Pre-trained on $h^{1,2}\in\{2,3,4\}$ → fine-tuned on $h^{1,2}=5$ geometries
- Conditional generation targeting prescribed values of $N_\text{flux}$, $|W_0|$, and $g_s$

---

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/StringJAX/towards_type_IIB_foundation_model/blob/main/diffusion/evaluate.ipynb) **`diffusion/evaluate.ipynb`** *(CPU)*

Evaluates all diffusion model variants using the same validation routine. Reports validity rates per geometry and per model, and conditional accuracy for $N_\text{flux}$, $|W_0|$, and $g_s$ targets.
<!-- docs:notebooks-end -->

---

<!-- docs:reproduce-start -->
## Reproducing the paper

The checkpoints behind each result. Transformer checkpoints are in `transformer/checkpoints/run_<N>/`, given as
pre-trained (or fine-tuned) and then GRPO-refined; diffusion checkpoints are in `diffusion/checkpoints/`.

| Result in the paper | Transformer | Diffusion model |
|:---|:---|:---|
| Pre-training and GRPO, all 20 geometries | `run_37`: `checkpoint_2000`, `GRPO_2/checkpoint_200` | `uncon_pretrained.pt`, `uncon_grpo.pt` (see the note below) |
| Transfer to $h^{1,2}=5$: benchmark trained from scratch | `run_44`: `checkpoint_1000`, `GRPO_2/checkpoint_200` | `h12_5_scratch_pretrain.pt`, `h12_5_scratch_grpo.pt` |
| Transfer to $h^{1,2}=5$: pre-trained on $h^{1,2}\le 4$, then fine-tuned | `run_45`: `checkpoint_1000`, `GRPO_2/checkpoint_200` | `h12_5_finetuned_pretrain.pt`, `h12_5_finetuned_grpo.pt` (see the note below) |
| Conditioning on $N_\text{flux}$ | `run_38`: `checkpoint_2000`, `GRPO_8/checkpoint_400` | `nflux_pretrained.pt`, `nflux_grpo.pt` |
| Conditioning on $\|W_0\|$ | `run_39`: `checkpoint_2000`, `GRPO_2/checkpoint_400` | `w0_pretrained.pt`, `w0_grpo.pt` |
| Conditioning on $1/g_s$ | `run_46`: `checkpoint_2000`, `GRPO_5/checkpoint_400` | `gs_pretrained.pt`, `gs_grpo.pt` |

The transformer notebooks load these runs (`RUN_ID`, `load_ckpt`). The scaling study and the timing comparison are not
part of this repository.

**Note on the diffusion checkpoints.** `uncon_grpo.pt` and `h12_5_finetuned_grpo.pt` are currently identical files,
and `diffusion/src/pipeline.py` still loads the GRPO checkpoints under their earlier names (`*_grpo_best.pt`). Both
will be corrected; until then the diffusion sample notebook cannot load its default checkpoints.
<!-- docs:reproduce-end -->

---

<!-- docs:citation-start -->
## Citation

If you use this code, please cite the paper (the arXiv number and the citation key will follow):

```bibtex
@article{Krippendorf:2026xxx,
    author = "Krippendorf, Sven and Larfors, Magdalena and Liu, Zhimei and Schachner, Andreas and Walden, Moritz",
    title = "{Towards a foundation model for Type IIB flux vacua}",
    eprint = "",
    archivePrefix = "arXiv",
    month = "11",
    year = "2026"
}
```
<!-- docs:citation-end -->
