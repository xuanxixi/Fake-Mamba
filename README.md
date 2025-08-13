# Fake-Mamba: Real-Time Speech Deepfake Detection Using Bidirectional Mamba as Self-Attention’s Alternative
(Accepeted at ASRU 2025 🇺🇸 🌴)
### Xi Xuan, Zimo Zhu, Wenxin Zhang, Yi-Cheng Lin, Tomi Kinnunen
In our paper, we proposed Fake-Mamba.

**Abstract:** 
Advances in speech synthesis intensify security threats, motivating real-time deepfake detection research. We investigate whether bidirectional Mamba can serve as a competitive alternative to Self-Attention in detecting synthetic speech. Our solution, Fake-Mamba, integrates an XLSR front-end with bidirectional Mamba to capture both local and global artifacts. Our core innovation introduces three efficient encoders: TransBiMamba, ConBiMamba, and PN-BiMamba. Leveraging XLSR's rich linguistic representations, PN-BiMamba can effectively capture the subtle cues of synthetic speech. Evaluated on ASVspoof 21 LA, 21 DF, and In-The-Wild benchmarks, Fake-Mamba achieves 0.97%, 1.74%, and 5.85% EER, respectively, representing substantial relative gains over SOTA models XLSR-Conformer and XLSR-Mamba. The framework maintains real-time inference across utterance lengths, demonstrating strong generalization and practical viability.

This source code is for the Fake-Mamba accepted by ASRU 2025.
