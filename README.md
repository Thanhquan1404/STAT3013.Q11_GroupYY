# Descriptive Analysis for Liver cirrhosis stages by loboratory and clinical Data

_A Statistical Exploration of Clinical Datases: Liver Cirrhosis Dataset_

---

## Overview

This repository contains the descriptive statistical analysis and exploratory data analysis (EDA) for two widely used clinical datasets in hepatology research:

   **Liver Cirrhosis Dataset**  
   → Multiclass classification: _Predict the stage of cirrhosis (1–3)._

The goal of this analysis is to **understand the clinical structure** of the datasets, identify trends, verify statistical hypotheses, and prepare the foundation for later modeling (Logistic Regression, Softmax Regression, KNN, SVM, Random Forest, etc.).

---

## Research Goals

- Perform **exploratory statistical analysis** on liver-related clinical indices.
- Identify **clinical indicators** strongly associated with liver cirrhosis stages conditions.
- Conduct **numerical and categorical statistical tests**:
  - Kruskal-Wallis H-Test
  - Chi-square test
  - Mean/variance comparison
  - Normality test (Q–Q plot)
- Analyze feature distributions using:
  - Histogram
  - Density plot (KDE)
  - Boxplot
  - Violin plot
- Detect and interpret **outliers** that represent abnormal clinical conditions.
- Generate **PCA and LDA projections** for data visualization and separability assessment.
- Summarize **insights essential for predictive modeling**.

---

## Repository Structure

---

**Run Instructions**:

- **Prepare environment (PowerShell)**: create and activate a virtual environment, then install dependencies from `requirements.txt`:

```powershell
# create a venv named .venv
python -m venv .venv
# activate the venv (PowerShell)
.\.venv\Scripts\Activate.ps1
# upgrade pip and install requirements
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

- **Run analyses / experiments**: most analysis and experiment scripts live under the `Data_Analysis/`, `experiment/` and `experiment_PCA/` folders. Example runs (from repository root):

```powershell
# run a descriptive analysis / visualization
python .\Data_Analysis\data_visualization.py

# run a single experiment script (e.g. linear models)
python .\experiment\linear_models_experiment.py

# run PCA-enabled experiment
python .\experiment_PCA\linear_models_PCA_experiment.py
```

Adjust commands as needed — many scripts read CSVs from the `data/` folder and write results to `experiment_result/` or `calculation_result/`.

**Environment / Requirements**:

- **Python**: recommended Python 3.8+ (3.8–3.11 tested).
- **Python packages**: see `requirements.txt` for required packages; key libraries include `numpy`, `pandas`, `scikit-learn`, `imbalanced-learn`, `matplotlib`, and `seaborn`.
- To reproduce results exactly, consider creating a new virtual environment and installing `requirements.txt` as shown above.

**Datasets & Links**:

- Local copies used in this repository are stored under the `data/` folder (e.g. `data/KFold_data/`, `data/data_apply_SMOTE/`).
- Liver cirrhosis / clinical datasets (commonly available on Kaggle or institutional repositories). If you have a specific source, replace or augment the link below:
  -Origin (417 samples): https://www.kaggle.com/datasets/fedesoriano/cirrhosis-prediction-dataset
  -Synthetic (25k samples): https://www.kaggle.com/datasets/aadarshvelu/liver-cirrhosis-stage-classification

**Demo Video / Presentation**:

- If a demo video or presentation exists, add its URL here. (No demo video linked yet — to add, update this section with the video URL or YouTube/Vimeo link.)

**License**:

- This project is provided under the **MIT License**. You can replace this with another license if required.

```
MIT License

Copyright (c) 2025 [Project Authors]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
