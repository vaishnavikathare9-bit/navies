# MAGIC Gamma Telescope Particle Classification

## Overview
This repository contains a machine learning workflow for classifying high-energy gamma-ray events against hadronic background noise recorded by the **MAGIC (Major Atmospheric Gamma Imaging Cherenkov)** telescope. The project covers data preprocessing, feature standardization, class rebalancing using oversampling techniques, and comparative model evaluation using Naive Bayes and $k$-Nearest Neighbors ($k$-NN).

---

## Dataset Description
The dataset used is the **MAGIC Gamma Telescope Data Set** (`magic04.data`) sourced from the UCI Machine Learning Repository. It consists of 19,020 instances described by 10 continuous Monte Carlo generated feature parameters and 1 target label.

### Features
| Feature Name | Type | Description |
| :--- | :--- | :--- |
| `fLength` | Continuous | Major axis of ellipse [mm] |
| `fWidth` | Continuous | Minor axis of ellipse [mm] |
| `fSize` | Continuous | 10-log of sum of content of all pixels |
| `fConc` | Continuous | Ratio of sum of two highest pixels over size |
| `fConc1` | Continuous | Ratio of highest pixel over size |
| `fAsym` | Continuous | Distance from highest pixel to center |
| `fM3Long` | Continuous | 3rd root of 3rd moment along major axis |
| `fM3Trans` | Continuous | 3rd root of 3rd moment along minor axis |
| `fAlpha` | Continuous | Angle of major axis with vector to camera center |
| `fDist` | Continuous | Distance from origin to center of ellipse |
| `class` | Binary | Target label: `g` = Gamma signal (1), `h` = Hadron background (0) |

---

## Workflow & Pipeline

1. **Data Preprocessing & Encoding:**
   - Transformed class labels into binary indicators (`g` $\rightarrow$ 1, `h` $\rightarrow$ 0).
   - Partitioned the dataset into **Train (60%)**, **Validation (20%)**, and **Test (20%)** splits.

2. **Feature Scaling & Class Imbalance:**
   - Applied `StandardScaler` to normalize the continuous numerical input features.
   - Utilized `RandomOverSampler` from `imblearn` on the training set to resolve class imbalance between gamma and hadron events.

3. **Model Selection & Evaluation:**
   - **Gaussian Naive Bayes (GaussianNB)**: Probabilistic classifier assuming conditionally independent features based on Bayes' theorem.
   - **$k$-Nearest Neighbors (KNN)**: Distance-based non-parametric classifier ($k=3$).

---

## Installation & Setup

```bash
# Clone the repository
git clone [https://github.com/your-username/magic-gamma-classification.git](https://github.com/your-username/magic-gamma-classification.git)
cd magic-gamma-classification

# Install dependencies
pip install numpy pandas matplotlib scikit-learn imbalanced-learn
