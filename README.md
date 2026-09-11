# Generative Diffusion for Amorphous Coal Networks

An end-to-end implementation adapting atomistic score-based generative diffusion to model disordered amorphous carbon/coal networks within consumer hardware constraints (Google Colab T4 GPU).

This project ports the core paradigm of the **DM2 (Diffusion Models for Disordered Materials)** framework from inorganic silica glass ($a\text{-}\mathrm{SiO}_2$) to complex organic amorphous coal, replacing heavy tensor-product architectures with an efficient, polarizable message-passing pipeline.

---

## Research Question
Can score-based generative diffusion models accurately capture the short- and medium-range order of amorphous coal networks while running entirely within a 15 GB VRAM budget?

---

## Key Adaptations from DM2

| Feature | Original DM2 Framework | This Implementation |
| :--- | :--- | :--- |
| **Target Material** | Amorphous Silica ($a\text{-}\mathrm{SiO}_2$), Metallic Glasses | Amorphous Coal / Disordered Carbon |
| **Compute Budget** | Multi-GPU Enterprise Clusters (RTX A6000 / A100) | Single Consumer GPU (Google Colab T4, 15 GB) |
| **Backbone Architecture** | Heavy `e3nn` (Spherical Harmonics Tensor Products) | Polarizable Atom Interaction Neural Network (**PaiNN**) |
| **Boundary Conditions** | Fractional cell coordinate convolutions | Hard PBC with **Minimum Image Convention (MIC)** |
| **Diffusion Setup** | Continuous SDE / DDPM hybrid | Discrete DDPM with coordinate space normalization |

---

## Repository Contents

This repository is structured as a self-contained, single-notebook submission:

* **`amorphous_coal_diffusion.ipynb`**: Complete executable pipeline containing:
  * Dataset fetching & spatial cropping (Caro/Deringer amorphous carbon trajectory).
  * Minimum Image Convention (`pbc_radius_graph`) construction.
  * 6-layer, 256-hidden-dim Heavyweight PaiNN denoiser.
  * DDPM forward diffusion and reverse sampling loops.
  * Radial Distribution Function (RDF) structural validation.

---

## Quickstart

1. Open `amorphous_coal_diffusion.ipynb` in [Google Colab](https://colab.research.google.com/).
2. Navigate to **Runtime > Change runtime type** and select **T4 GPU**.
3. Run all cells sequentially. The notebook will automatically download the required atomic trajectory, crop the training domain, initialize the PaiNN model, and generate the 3D atomic network (`generated_coal_pbc.xyz`).

---

## References & Attribution

* **Original DM2 Repository:** [digital-synthesis-lab/DM2](https://github.com/digital-synthesis-lab/DM2)
* **Primary Reference:** Yang, K., & Schwalbe-Koda, D. *"A Generative Diffusion Model for Amorphous Materials"*, npj Computational Materials (2025) / [arXiv:2507.05024](https://arxiv.org/abs/2507.05024).
* **Dataset Reference:** Caro, M. A., et al. Amorphous carbon structures generated via Gaussian Approximation Potentials (GAP).
