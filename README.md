# Fake-Mamba: Real-Time Speech Deepfake Detection Using Bidirectional Mamba as Self-Attention’s Alternative  
(Accepted at [ASRU 2025](https://2025.ieeeasru.org/) 🇺🇸 🌴)

[![arXiv](https://img.shields.io/badge/arXiv-2508.09294v1-b31b1b.svg)](https://arxiv.org/abs/2508.09294v1)

### **Xi Xuan**, **Zimo Zhu**, **Wenxin Zhang**, **Yi-Cheng Lin**, **Tomi Kinnunen**

> 🔊 *Detecting synthetic speech in real time — without self-attention.*


## 📘 Abstract

Advances in speech synthesis intensify security threats, motivating real-time deepfake detection research. In this work, we investigate whether **bidirectional Mamba** can serve as a competitive alternative to Self-Attention in detecting synthetic speech.

We propose **Fake-Mamba**, a novel framework that combines the pretrained XLSR front-end with bidirectional Mamba blocks to capture both local and global artifacts. Our core innovation introduces three efficient encoders: TransBiMamba, ConBiMamba, and PN-BiMamba. Leveraging XLSR's rich linguistic representations, **PN-BiMamba** can effectively capture the subtle cues of synthetic speech.

Evaluated on benchmark datasets, Fake-Mamba sets new state-of-the-art results:
- 📉 **0.97% EER** on ASVspoof2021 LA
- 📉 **1.74% EER** on ASVspoof2021 DF
- 📉 **5.85% EER** on In-The-Wild (ITW)

These results represent significant improvements over prior SOTA models such as XLSR-Conformer and XLSR-Mamba, while maintaining **real-time inference** across variable-length utterances. Fake-Mamba demonstrates strong generalization and practical deployment potential.

---

## 💻 Code & Reproducibility

This repository contains the official implementation of **Fake-Mamba**, accepted at **ASRU 2025**.

🔧 **Framework**:  
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?logo=PyTorch&logoColor=white)](https://pytorch.org)
[![Fairseq](https://img.shields.io/badge/Fairseq-%23007FFF.svg?logo=Facebook&logoColor=white)](https://github.com/facebookresearch/fairseq)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-%23FFD100.svg?logo=Hugging%20Face&logoColor=black)](https://huggingface.co)
[![Mamba-SSM](https://img.shields.io/badge/Mamba--SSM-%23000000.svg?logo=github&logoColor=white)](https://github.com/state-spaces/mamba)

📦 Includes: Training scripts, inference pipeline, pretrained models, and evaluation tools.

👉 [Get Started: `README.md`](./README.md) | 📂 [Model Checkpoints](./checkpoints/) | 📈 [Training](./logs/)


## 📚 Citation
@article{xuan2025fake,
  title={Fake-Mamba: Real-Time Speech Deepfake Detection Using Bidirectional Mamba as Self-Attention's Alternative},
  author={Xuan, Xi and Zhu, Zimo and Zhang, Wenxin and Lin, Yi-Cheng and Kinnunen, Tomi},
  journal={arXiv preprint arXiv:2508.09294},
  year={2025}
}

If you find this work useful, please cite our paper:

```bibtex
