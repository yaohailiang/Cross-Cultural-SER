# SERBench: A Comprehensive Multilingual Benchmark for Cross-Cultural Speech Emotion Recognition

SERBench is the first comprehensive benchmark specifically designed to evaluate the cross-cultural speech emotion recognition (SER) performance of Audio Large Language Models (ALLMs).

--- 

## 📂 Supplementary Material

For detailed dataset construction, Emotion Wheel definitions, and additional ablation studies, please refer to the Appendix:

* **[Appendix.pdf](./Appendix.pdf)** (Uploaded to the repository `Cross-Cultural-SER/Appendix.pdf`).

--- 

## 📊 SERD-CC Dataset

We construct the Cross-Cultural Speech Emotion Recognition Dataset (SERD-CC), comprising **10 public corpora** across **8 languages**.

### Original Corpus Access Links

| Dataset | Language | Source Link |
| :--- | :--- | :--- |
| **ShEMO** | Persian | [Github](https://github.com/mansourehk/ShEMO) |
| **EmoDB** | German | [Link](http://emodb.bilderbar.info/docu/#emodb) |
| **CaFE** | French | [Zenodo](https://doi.org/10.5281/zenodo.1219621) |
| **SAVEE** | English | [Link](http://kahlan.eps.surrey.ac.uk/savee/) |
| **URDU** | Urdu | [Github](https://github.com/siddiquelatif/URDU-Dataset) |
| **SUBESCO** | Bengali | [Zenodo](https://doi.org/10.5281/zenodo.4526477) |
| **RAVDESS** | English | [Zenodo](https://doi.org/10.5281/zenodo.1188976) |
| **AESDD** | Greek | [Link](https://m3c.web.auth.gr/?s=AESDD) |
| **MER2023** | Chinese | [HuggingFace](https://huggingface.co/datasets/MERChallenge/MER2023) |
| **MER2024** | Chinese | [Website](https://zeroqiaoba.github.io/MER2024-website/) |

### Data Statistics

* **Total samples**: 15,975 speech utterances.
* **Number of speakers**: Over 1,600.
* **Total duration**: Approximately 17.4 hours.

--- 

## ⚙️ Core Method: LPF Framework

We propose the **Ling-Para Fusion (LPF) framework**, which enhances the emotional perception capability of ALLMs through a dual-stream architecture:
* **Linguistic semantics**: Obtains textual contextual information from subtitles.
* **Paralinguistic features**: Extracts pitch, energy, and speaking rate using Librosa, and converts them into textual descriptions.

--- 

## 📏 Evaluation Metric: Hitrate

To address the open-ended output of generative models, we introduce a novel evaluation metric called **Hitrate (HR)**:
* **Emotion wheel mapping**: Maps predictions into a hierarchical emotion wheel space for evaluation.
* **Semantic robustness**: Overcomes the limitations of traditional classification metrics in multi-label and zero-shot scenarios.
