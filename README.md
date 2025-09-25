# Inferring Mental Strategy (Holistic vs. Piecemeal) using Eye-Tracking Data

### IITB EdTech Internship 2025 | T1_G34_Stratix

This repository contains the complete project for **Problem Statement 06** as part of the IITB EdTech Internship 2025, in association with DYPCET. The project focuses on using unsupervised machine learning to analyze eye-tracking data and infer the cognitive strategies of participants engaged in mental rotation tasks.

---

## 🎯 Problem Statement

**Problem ID - 06: Inferring Mental Strategy (Holistic vs. Piecemeal)**

* **Objective**: To cluster participants based on their eye-tracking saccades and fixations to infer whether they employ a "Holistic" (global, gist-based) or "Piecemeal" (local, detail-oriented) problem-solving strategy.
* **ML Task**: This is an unsupervised learning problem that leverages a combination of feature-based **Clustering** and **Hidden Markov Models (HMMs)** for sequence analysis. An optional deep learning approach using LSTMs is also explored.

---

## 📊 Dataset

The analysis is performed on the publicly available dataset:

* **Title**: "A multisensor dataset of south asian post-graduate students working on mental rotation tasks"
* **Source**: [SpringerNature Figshare](https://springernature.figshare.com/articles/dataset/A_multisensor_dataset_of_south_asian_post_graduate_students_working_on_mental_rotation_tasks/28120670?file=51439640)

For this problem, we primarily use the `_IVT.csv` files (which contain processed fixation and saccade data) and the `_PSY.csv` files (for trial start/end timings) for all 38 participants.

---

## ⚙️ Methodology & Pipeline

Our approach is a multi-stage pipeline implemented across a series of Jupyter notebooks. The core idea is to perform a robust, per-trial analysis, extract a rich set of features, and then use both traditional and advanced machine learning models to identify behavioral patterns.

1.  **Preprocessing (`01_preprocessing_eye.ipynb`)**:
    * Downloads and loads data for all 38 participants.
    * Integrates `PSY.csv` timings to slice the continuous `IVT.csv` data into discrete trials.
    * Cleans and saves the processed per-trial data.

2.  **Feature Engineering (`02_feature_engineering_scanpaths.ipynb`)**:
    * Defines data-driven Areas of Interest (AOIs) using KMeans.
    * Extracts a rich set of features for each trial, including fixation stats, saccade metrics, and sequence patterns (e.g., scanpath length, entropy).
    * Aggregates trial-level features to the participant level for clustering.

3.  **Clustering Baselines (`03_clustering_baselines.ipynb`)**:
    * Applies KMeans, GMM, and HDBSCAN to the aggregated participant features.
    * Evaluates cluster quality using the Silhouette Score and interprets clusters based on feature means.
    * Assigns the final "Holistic" or "Piecemeal" label to each participant.

4.  **Sequence Modeling (`04_sequence_models_hmm_hsmm.ipynb`)**:
    * Trains separate Hidden Markov Models (HMMs) for each strategy group on their AOI sequences.
    * This step validates the clusters by revealing the different underlying dynamic patterns of eye movements.

5.  **Deep Learning (Optional) (`05_deep_sequence_embeddings.ipynb`)**:
    * Implements an advanced LSTM Autoencoder in PyTorch to learn "embeddings" (fingerprints) of each trial's scanpath.
    * Clusters these learned embeddings as an alternative, powerful method for identifying strategies.

6.  **Visualization & Validation (`06_validation_visualization.ipynb`)**:
    * Generates all key visualizations: PCA plots, feature comparison boxplots, prototype heatmaps/scanpaths, and HMM transition matrices.
    * Presents the final report and conclusions.

---

## 📂 File & Code Organization

The repository is structured as follows:
```
project/
├── data/                  # Stores intermediate data files (created by notebooks)
├── models/                # Stores saved, trained models (created by notebooks)
├── notebooks/             # Contains the sequential Jupyter notebooks for the pipeline
│   ├── 01_preprocessing_eye.ipynb
│   ├── 02_feature_engineering_scanpaths.ipynb
│   ├── 03_clustering_baselines.ipynb
│   ├── 04_sequence_models_hmm_hsmm.ipynb
│   ├── 05_deep_sequence_embeddings.ipynb
│   └── 06_validation_visualization.ipynb
└── README.md              # This file
```
---

## ✨ Key Findings

The analysis successfully identified two distinct behavioral clusters that align with the theoretical definitions of Piecemeal and Holistic strategies.

* **Piecemeal Strategy**: Characterized by a significantly higher number of fixations, longer total scanpath length, and lower scanpath entropy. Visualizations show dense, localized fixation patterns.
* **Holistic Strategy**: Characterized by fewer fixations, larger average saccade amplitudes, and higher scanpath entropy. Visualizations show sparse, globally distributed fixation patterns.

The HMM analysis further validated these findings, revealing that the Piecemeal group exhibited "sticky" states of deep focus, while the Holistic group showed more dynamic transitions indicative of exploratory scanning.



---

## 🛠️ Technologies Used

* **Language:** Python 3.x
* **Core Libraries:** Pandas, NumPy, Scikit-learn
* **Sequence Modeling:** HMMlearn, PyTorch (for LSTM)
* **Visualization:** Matplotlib, Seaborn
* **Environment:** Jupyter Notebook / Google Colab

















