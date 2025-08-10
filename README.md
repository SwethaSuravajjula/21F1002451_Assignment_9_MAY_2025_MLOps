# Iris Dataset – Data Drift, Fairness, and Explainability Analysis

## 📌 Overview

This project demonstrates a complete workflow for:

* Detecting **data drift** between reference and current datasets.
* Evaluating **model fairness** across a sensitive attribute.
* Providing **model explainability** using SHAP.

It uses the **Iris dataset** with an added synthetic `location` feature to simulate sensitive attributes and an artificial data drift scenario.

---

## 🚀 Features

1. **Data Drift Detection** – Using [Evidently AI](https://github.com/evidentlyai/evidently) to compare reference vs. current datasets.
2. **Fairness Evaluation** – Using [Fairlearn](https://fairlearn.org/) to check model performance across different sensitive groups.
3. **Model Training** – Decision Tree classifier trained on clean reference data.
4. **Model Explainability** – Using [SHAP](https://shap.readthedocs.io/en/latest/) to understand feature importance for predictions.

---

## 📂 Project Structure

```
train.ipynb       # Main notebook
iris.csv          # Dataset (optional, auto-generated if not found)
```

---

## 🛠 Installation

```bash
# Clone this repository
git clone <repo-url>
cd <repo-folder>

# Install dependencies
pip install pandas numpy scikit-learn evidently fairlearn shap
```

---

## 📊 Workflow

1. **Data Preparation**

   * Load Iris dataset from `iris.csv` or sklearn.
   * Add a synthetic `location` column (0 or 1).
   * Create a *current* dataset with artificial noise to simulate drift.

2. **Data Drift Analysis**

   * Compare reference (clean) dataset with current (noisy) dataset.
   * Generate drift and data summary reports using Evidently.

3. **Fairness Analysis**

   * Train a Decision Tree classifier.
   * Evaluate accuracy overall and per sensitive group (`location`).

4. **Model Explainability**

   * Use SHAP to explain predictions for the `virginica` class.
   * Visualize feature contributions with force plots.

---

## 📈 Example Outputs

* **Drift Report** – Identifies numerical & categorical feature drift.
* **Fairness Report** – Shows performance disparities across `location`.
* **SHAP Plots** – Highlights feature importance for predictions.

