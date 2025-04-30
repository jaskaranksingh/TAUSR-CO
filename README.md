# TAUSR-CO: Texture-Attentive Ultrasound Super-Resolution with Contrast Optimization

This repository contains a modular implementation of the TAUSR-CO architecture described in our MICCAI 2025 submission.<br>
TAUSR-CO is a hierarchical, frequency-aware super-resolution model designed specifically for ultrasound imaging.<br>
TAUSR-CO focuses on restoring fine-grained textures, enhancing contrast, and refining depth—all while handling the unique challenges of ultrasound data like speckle noise and low signal fidelity.


##  Highlights
- Shallow → Structural → Contextual multi-layer processing
- Frequency & spatial domain fusion using FFT
- Contrast enhancement, speckle suppression, and depth-aware refinement
- Modularized for easy expansion and experimentation

> 📌 **Note**: Full training code and pretrained models will be released upon paper acceptance.

### Architecture Diagram
![TAUSR-CO Architecture](assets/tausr_architecture_diagram.png)

---

## Visual Results

### Comparison with SOTA Models
![PSNR/SSIM Comparison](assets/ultrasound_metrics_comparison.png)

### Ultrasound Texture + Contrast Restoration
![Color-Mapped Ultrasound SR](assets/ultrasound_comparison_grid.png)

---

## 📁 Repository Structure

- `models/` – All architectural blocks (FSFE, TEM, SNR, ECM, EPM, etc.)
- `core/` – Training and inference pipelines
- `utils/` – Utility functions for FFT, metrics, plotting
- `configs/` – Configurations for training and testing
- `assets/` – Figures, diagrams, and illustrations
- `data/` – Ultrasound Dataset

---

## Sample Run

```bash
python main.py --config configs/config.yaml
```

## Requirements

See `requirements.txt` for dependencies.

## 📝 Citation
BibTeX will be added post acceptance at MICCAI.
