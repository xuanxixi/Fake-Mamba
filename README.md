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

## 💻 Getting Started

This repository contains the official implementation of **Fake-Mamba**, accepted at **ASRU 2025**.

🔧 **Framework**:  
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?logo=PyTorch&logoColor=white)](https://pytorch.org)
[![Fairseq](https://img.shields.io/badge/Fairseq-%23007FFF.svg?logo=Facebook&logoColor=white)](https://github.com/facebookresearch/fairseq)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-%23FFD100.svg?logo=Hugging%20Face&logoColor=black)](https://huggingface.co)
[![Mamba-SSM](https://img.shields.io/badge/Mamba--SSM-%23000000.svg?logo=github&logoColor=white)](https://github.com/state-spaces/mamba)

---

## Installation
First, clone the repository locally, create and activate a conda environment, and install the requirements :
```
$ git clone https://github.com/xxuan-acoustics/Fake-Mamba.git
$ cd Fake-Mamba
$ unzip fairseq-a54021305d6b3c4c5959ac9395135f63202db8f1.zip
$ conda create -n Fake-Mamba python=3.7
$ conda activate Fake-Mamba
$ pip install torch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1
$ cd fairseq-a54021305d6b3c4c5959ac9395135f63202db8f1
$ pip install --editable ./
$ cd ..
$ pip install -r requirements.txt
```

## Experiments

### Dataset
Our experiments are performed on the public dataset logical access (LA) and deepfake (DF) partition of the ASVspoof 2021 dataset and In-the-Wild dataset(train on 2019 LA training and evaluate on 2021 LA and DF, In-the-Wild evaluation database).

The ASVspoof 2019 dataset, which can can be downloaded from [here](https://datashare.is.ed.ac.uk/handle/10283/3336).

The ASVspoof 2021 database is released on the zenodo site.

LA [here](https://zenodo.org/record/4837263#.YnDIinYzZhE)

DF [here](https://zenodo.org/record/4835108#.YnDIb3YzZhE)

The In-the-Wild dataset can be downloaded from [here](https://deepfake-total.com/in_the_wild)

For ASVspoof 2021 dataset keys (labels) and metadata are available [here](https://www.asvspoof.org/index2021.html)

## Pre-trained wav2vec 2.0 XLS-R (300M)
Download the XLS-R models from [here](https://github.com/pytorch/fairseq/tree/main/examples/wav2vec/xlsr)

## Training model
To train the model run:
```
CUDA_VISIBLE_DEVICES=0 python main.py --track=DF --lr=0.000001 --batch_size=20 --loss=WCE  --num_epochs=50
```
## Testing

To evaluate your own model on the DF, LA, and In-the-Wild evaluation datasets: The code below will generate three 'score.txt' files, one for each evaluation dataset, and these files will be used to compute the EER(%).
```
CUDA_VISIBLE_DEVICES=0 python main.py   --track=DF --is_eval --eval 
                                        --model_path=/path/to/your/best_model.pth
                                        --protocols_path=database/ASVspoof_DF_cm_protocols/ASVspoof2021.DF.cm.eval.trl.txt 
                                        --database_path=/path/to/your/ASVspoof2021_DF_eval/ 
                                        --eval_output=/path/to/your/scores_DF.txt

CUDA_VISIBLE_DEVICES=0 python main.py   --track=LA --is_eval --eval 
                                        --model_path=/path/to/your/best_model.pth
                                        --protocols_path=database/ASVspoof_DF_cm_protocols/ASVspoof2021.LA.cm.eval.trl.txt 
                                        --database_path=/path/to/your/ASVspoof2021_LA_eval/ 
                                        --eval_output=/path/to/your/scores_LA.txt

```


Compute the EER(%) use three 'scores.txt' file
```
python evaluate_2021_DF.py scores/scores_DF.txt ./keys eval

python evaluate_2021_LA.py scores/scores_LA.txt ./keys eval

python SITW-DF_test.py
``` 

## Results using pre-trained model:
| Dataset                     | EER (%) |
|-----------------------------|---------|
| ASVspoof 2021 DF            | 1.74    |
| ASVspoof 2021 LA            | 0.97    |
| In-the-Wild                 | 5.85    |




## 📚 Citation

If you find this work useful, please cite our paper:

```
@inproceedings{xuan2025fakemamba,
  title        = {Fake-Mamba: Real-Time Speech Deepfake Detection Using Bidirectional Mamba as Self-Attention's Alternative},
  author       = {Xuan, Xi and Zhu, Zimo and Zhang, Wenxin and Lin, Yi-Cheng and Kinnunen, Tomi},
  booktitle    = {Proceedings of the IEEE ASRU},
  year         = {2025}
}
```
