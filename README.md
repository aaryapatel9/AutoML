# 🤖 AutoML Mini-Framework with Optuna

![Python](https://img.shields.io/badge/Python-3.7%2B-3776AB?style=flat-square&logo=python&logoColor=white)
![Optuna](https://img.shields.io/badge/Optuna-TPE%20Sampler-6C63FF?style=flat-square&logo=optuna&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-%E2%89%A51.0.0-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-22C55E?style=flat-square)
![Status](https://img.shields.io/badge/Status-Active-22C55E?style=flat-square)

An automated machine learning framework that uses **Bayesian Optimization (Optuna TPE)** to search through multiple machine learning models and their hyperparameters simultaneously — finding the best configuration for your dataset without manual tuning.

---

## 📋 Table of Contents

1. [Overview](#1-overview)
2. [Project Structure](#2-project-structure)
3. [Installation](#3-installation)
4. [Classes & Architecture](#5-classes--architecture)
5. [Supported Models & Hyperparameter Search Spaces](#10-supported-models--hyperparameter-search-spaces)
6. [How Bayesian Optimization Works](#11-how-bayesian-optimization-works)
7. [Step-by-Step Usage Guide](#12-step-by-step-usage-guide)
8. [Running the Provided Scripts](#13-running-the-provided-scripts)
9. [Visualizations](#14-visualizations)
10. [Output & Results Explained](#15-output--results-explained)
11. [Troubleshooting](#19-troubleshooting)

---

## 1. 🔍 Overview

### ⚙️ What This Framework Does

Most ML projects require tedious manual work — trying different models, tuning hyperparameters one by one, comparing results. This framework automates the entire process:

1. You provide a dataset and task type (classification or regression)
2. The framework tries dozens of model + hyperparameter combinations
3. Each trial is guided by Bayesian optimization — smarter than random or grid search
4. After all trials, it returns the single best model, ready to make predictions

### 🧠 Why Bayesian Optimization Instead of Grid Search

| Approach | How It Works | Trials Needed | Quality |
|---|---|---|---|
| Grid Search | Tries every combination exhaustively | 100s – 1000s | Baseline |
| Random Search | Picks combinations randomly | 100s | Slightly better |
| **Bayesian (Optuna TPE)** | Learns from past trials to pick better ones | **30 – 100** | **Best** |

Bayesian optimization builds a probabilistic model of which hyperparameter regions produce good scores and focuses new trials there — getting better results in far fewer iterations.

### ✨ Key Features

- **7 classification algorithms** with individual search spaces
- **10 regression algorithms** with individual search spaces
- **Optuna TPE sampler** for smart hyperparameter selection
- **MedianPruner** to kill unpromising trials early
- **k-fold cross-validation** on training data for reliable scoring
- **StandardScaler** applied automatically to all features
- **3 built-in visualizations** (optimization history, model comparison, hyperparameter distribution)
- Fully extensible — add your own models or metrics by subclassing

---

## 2. 📁 Project Structure

```
automl_project/
│
├── automl_optuna_framework.py   # Core framework — all classes live here
├── automl_examples.py           # Five detailed examples with saving and comparison
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

### 📄 What Each File Does

**`automl_optuna_framework.py`** — The only file you need to import. Contains:
- `ModelResult` dataclass
- `BaseAutoML` abstract base class
- `ClassificationAutoML` concrete class
- `RegressionAutoML` concrete class


**`automl_examples.py`** — Five separate example functions covering different datasets and sizes. Includes a comparison function that ranks all experiments against each other and saves a bar chart.

---

## 3. ⚙️ Installation

### 🐍 Step 1 — Check Python version

This framework requires **Python 3.7 or higher**.

```bash
python --version
```

### 🏗️ Step 2 — Create a virtual environment (recommended)

A virtual environment keeps this project's packages isolated from other projects.

```bash
# Create the environment
python -m venv venv

# Activate it — Windows
venv\Scripts\activate

# Activate it — macOS / Linux
source venv/bin/activate
```

### 📦 Step 3 — Install dependencies

```bash
pip install -r requirements.txt
```

**What gets installed:**

| Package | Version | Purpose |
|---|---|---|
| `numpy`  | >= 1.21.0 | Array operations |
| `pandas`  | >= 1.3.0 | DataFrames and CSV handling |
| `matplotlib`  | >= 3.4.0 | Plotting optimization history and comparisons |
| `seaborn`  | >= 0.11.0 | Plot styling |
| `scikit-learn`  | >= 1.0.0 | All ML models and metrics |
| `optuna`  | >= 3.0.0 | Bayesian hyperparameter optimization |

### ✅ Step 4 — Verify installation

```bash
python -c "import numpy, pandas, matplotlib, seaborn, sklearn, optuna; print('All packages installed successfully')"
```

If no errors appear, you are ready.

### 🚀 Step 5 — Verify the framework imports correctly

```bash
python -c "from automl_optuna_framework import ClassificationAutoML, RegressionAutoML; print('Framework ready')"
```

---

## 4. 🏛️ Classes & Architecture

```
BaseAutoML  (Abstract Base Class)
├── Implements : fit(), predict(), predict_proba()
├── Implements : get_summary(), get_top_models()
├── Implements : plot_optimization_history(), plot_model_comparison(), plot_hyperparameter_importance()
├── Abstract   : get_models(), define_search_space(), get_scoring_metric(), _evaluate()
│
├── ClassificationAutoML
│   ├── get_models()           →  7 classifiers
│   ├── define_search_space()  →  classification hyperparameter ranges
│   ├── get_scoring_metric()   →  'f1_weighted'
│   └── _evaluate()            →  f1_score(..., average='weighted')
│
└── RegressionAutoML
    ├── get_models()           →  10 regressors
    ├── define_search_space()  →  regression hyperparameter ranges
    ├── get_scoring_metric()   →  'r2'
    └── _evaluate()            →  r2_score()
```

The abstract base class handles all the heavy lifting — data splitting, scaling, running Optuna, collecting results, sorting, and plotting. Concrete subclasses only define which models to use, what their hyperparameter ranges are, and which metric to optimize.

---

## 5.  Supported Models & Hyperparameter Search Spaces

### 🎯 Classification Models

####  Logistic Regression
A fast linear classifier. Works well for linearly separable data and provides probability estimates natively.

| Hyperparameter | Type | Search Range | Scale | Notes |
|---|---|---|---|---|
| `C` | float | 0.001 → 1000 | log | Inverse regularization strength. Higher = less regularization. |
| `max_iter` | int | 100 → 1000 | linear | Max solver iterations before stopping. |

####  Decision Tree

| Hyperparameter | Type | Search Range | Notes |
|---|---|---|---|
| `max_depth` | int | 2 → 20 | Maximum depth of the tree. Lower = simpler model. |
| `min_samples_split` | int | 2 → 20 | Minimum samples required to split a node. |
| `min_samples_leaf` | int | 1 → 10 | Minimum samples required to be a leaf node. |

####  Random Forest

| Hyperparameter | Type | Search Range | Notes |
|---|---|---|---|
| `n_estimators` | int | 50 → 300 | Number of trees. More = better but slower. |
| `max_depth` | int | 5 → 30 | Max depth per tree. |
| `min_samples_split` | int | 2 → 15 | Min samples to split a node. |
| `min_samples_leaf` | int | 1 → 10 | Min samples per leaf. |

####  Gradient Boosting

| Hyperparameter | Type | Search Range | Scale | Notes |
|---|---|---|---|---|
| `n_estimators` | int | 50 → 300 | linear | Number of boosting stages. |
| `learning_rate` | float | 0.001 → 0.5 | log | Shrinkage per tree. Lower = more robust, needs more trees. |
| `max_depth` | int | 2 → 10 | linear | Depth of each individual tree. |
| `subsample` | float | 0.5 → 1.0 | linear | Fraction of samples used per tree. |

####  AdaBoost

| Hyperparameter | Type | Search Range | Notes |
|---|---|---|---|
| `n_estimators` | int | 50 → 200 | Number of estimators. |
| `learning_rate` | float | 0.5 → 2.0 | Contribution weight of each classifier. |

####  SVM

| Hyperparameter | Type | Search Range | Scale | Notes |
|---|---|---|---|---|
| `C` | float | 0.01 → 1000 | log | Regularization. Higher = tighter fit, higher overfitting risk. |
| `kernel` | categorical | rbf, poly, linear | — | `rbf` is the default and works well in most cases. |
| `gamma` | categorical | scale, auto | — | Kernel coefficient. `scale` = 1 / (n_features x X.var()). |

####  KNN

| Hyperparameter | Type | Search Range | Notes |
|---|---|---|---|
| `n_neighbors` | int | 3 → 30 | Number of neighbors to consider. |
| `weights` | categorical | uniform, distance | `distance` weights closer neighbors more heavily. |
| `metric` | categorical | euclidean, manhattan | Distance metric for finding neighbors. |

---

### 📈 Regression Models

####  Linear Regression
No hyperparameters — the OLS solution is unique. Used as a fast baseline.

####  Ridge
Linear regression with L2 regularization. Shrinks coefficients toward zero but keeps all features active.

| Hyperparameter | Type | Search Range | Scale | Notes |
|---|---|---|---|---|
| `alpha` | float | 0.001 → 1000 | log | Regularization strength. Higher = stronger shrinkage. |

####  Lasso
Linear regression with L1 regularization. Can zero out irrelevant features entirely (automatic feature selection).

| Hyperparameter | Type | Search Range | Scale | Notes |
|---|---|---|---|---|
| `alpha` | float | 0.0001 → 1.0 | log | Regularization strength. |
| `max_iter` | int | 1000 → 5000 | linear | Max iterations for coordinate descent solver. |

####  ElasticNet
Combines L1 and L2 regularization. `l1_ratio` controls the mix between them.

| Hyperparameter | Type | Search Range | Notes |
|---|---|---|---|
| `alpha` | float | 0.0001 → 1.0 (log) | Overall regularization strength. |
| `l1_ratio` | float | 0.0 → 1.0 | 0 = Ridge, 1 = Lasso, values in between = blend. |
| `max_iter` | int | 1000 → 5000 | Solver iterations. |

####  Decision Tree (Regressor)
Same structure as classifier version but predicts continuous values. Same hyperparameters as Classification Decision Tree.

####  Random Forest (Regressor)
Same structure as classifier. Averages predictions across all trees. Same hyperparameters as Classification Random Forest.

####  Gradient Boosting (Regressor)
Same structure as classifier. Minimizes a regression loss. Same hyperparameters as Classification Gradient Boosting.

####  AdaBoost (Regressor)
Same structure as classifier. Uses regression base estimators. Same hyperparameters as Classification AdaBoost.

####  SVR (Support Vector Regression)
Fits a tube of width `epsilon` around the data. Points inside the tube contribute zero loss.

| Hyperparameter | Type | Search Range | Scale | Notes |
|---|---|---|---|---|
| `C` | float | 0.01 → 1000 | log | Regularization. |
| `kernel` | categorical | rbf, poly, linear | — | Kernel function. |
| `gamma` | categorical | scale, auto | — | Kernel coefficient. |
| `epsilon` | float | 0.01 → 1.0 | linear | Width of the insensitive tube. |

####  KNN (Regressor)
Predicts the average target value of the k nearest neighbors. Same hyperparameters as Classification KNN.

---

## 6. 🧠 How Bayesian Optimization Works

### ❌ The Problem with Grid Search

If you have 5 hyperparameters each with 5 possible values, grid search requires 5^5 = 3,125 combinations. Double the values and it becomes 10^5 = 100,000. It scales exponentially — the curse of dimensionality.

### 🔮 What Optuna TPE Does

Optuna uses the **Tree-structured Parzen Estimator (TPE)** algorithm:

1. **First 10 trials** — explore randomly to build an initial picture of the search space
2. **All subsequent trials** — build two probability density models:
   - `l(x)` — density of hyperparameter values that produced good scores
   - `g(x)` — density of hyperparameter values that produced poor scores
3. **Next trial** — sample hyperparameters that maximize `l(x) / g(x)`, i.e., configurations that resemble good results and differ from bad ones
4. After each trial, both models are updated with the new data point

The optimizer genuinely learns which regions of the search space are promising and focuses there, rather than wasting trials on regions that have already proven bad.

### ✂️ MedianPruner

The `MedianPruner` is an additional optimization layer. It monitors intermediate trial results and prunes (stops early) a trial if its intermediate score is below the median of all completed trials at the same evaluation step. This avoids spending full training time on clearly bad configurations.

Pruning activates after 10 startup trials (`n_startup_trials=10`).

### 📊 Practical Impact

On the breast cancer dataset with 40 trials, the optimizer typically finds a model scoring above 0.97 within the first 15–20 trials and spends the remaining trials fine-tuning hyperparameters in that neighborhood. A random search achieving the same quality would typically require 80–120 trials.

---

## 7. 📖 Step-by-Step Usage Guide

###  Scenario A — Classification with your own CSV file

```python
import pandas as pd
from automl_optuna_framework import ClassificationAutoML

# Step 1 — Load your data
df = pd.read_csv('your_data.csv')

# Step 2 — Separate features and target
# Replace 'target' with the actual name of your target column
X = df.drop(columns=['target'])
y = df['target']

# Step 3 — Handle categorical features in X (if any)
# The framework expects numeric input; encode string columns first
for col in X.select_dtypes(include='object').columns:
    X[col] = pd.Categorical(X[col]).codes

# Step 4 — Create the AutoML instance
automl = ClassificationAutoML(
    verbose      = True,
    n_jobs       = -1,
    random_state = 42
)

# Step 5 — Run the search
automl.fit(
    X, y,
    test_size = 0.2,
    cv_folds  = 5,
    n_trials  = 50
)

# Step 6 — Inspect the best model
best = automl.best_result
print(f"Best model : {best.model_name}")
print(f"CV Score   : {best.cv_score:.4f} +/- {best.cv_std:.4f}")
print(f"Test Score : {best.test_score:.4f}")
print(f"Params     : {best.hyperparameters}")

# Step 7 — View all results ranked by CV score
summary = automl.get_summary()
print(summary.head(10).to_string())

# Step 8 — Make predictions on new data
X_new = pd.read_csv('new_data.csv').drop(columns=['target'])
predictions = automl.predict(X_new)

# Step 9 — Save results to CSV
summary.to_csv('automl_results.csv', index=False)

# Step 10 — Visualize
import matplotlib.pyplot as plt
automl.plot_optimization_history()
automl.plot_model_comparison(top_n=10)
automl.plot_hyperparameter_importance(top_n=8)
plt.show()
```

---

###  Scenario B — Regression with a sklearn dataset

```python
from sklearn.datasets import load_diabetes
from automl_optuna_framework import RegressionAutoML
import matplotlib.pyplot as plt

# Step 1 — Load data
X, y = load_diabetes(return_X_y=True)

# Step 2 — Create instance
automl = RegressionAutoML(verbose=True, random_state=42)

# Step 3 — Run
automl.fit(X, y, n_trials=40, cv_folds=5)

# Step 4 — Results
print(f"Best  : {automl.best_result.model_name}")
print(f"R2 CV : {automl.best_result.cv_score:.4f}")
print(f"R2 Test: {automl.best_result.test_score:.4f}")

# Step 5 — Predict
predictions = automl.predict(X)

# Step 6 — Plots
automl.plot_optimization_history()
automl.plot_model_comparison()
plt.show()
```

---

###  Scenario C — Time-limited search

When you have a fixed time budget rather than a fixed trial count:

```python
automl = ClassificationAutoML(verbose=True)

# Run as many trials as possible within 10 minutes
automl.fit(X, y, n_trials=10000, timeout=600)

print(f"Completed {len(automl.results)} trials in 10 minutes")
print(automl.best_result)
```

---

###  Scenario D — Inspecting individual fold scores

```python
automl.fit(X, y, n_trials=30, cv_folds=5)

best = automl.best_result

print("Individual fold scores:")
for i, score in enumerate(best.cv_scores):
    print(f"  Fold {i+1}: {score:.4f}")

print(f"Mean : {best.cv_score:.4f}")
print(f"Std  : {best.cv_std:.4f}")
```

---

###  Scenario E — Using the fitted model object directly

`best_result.model` is a fully fitted sklearn estimator and can be used like any sklearn model:

```python
automl.fit(X_train, y_train, n_trials=30)

model = automl.best_result.model

# Feature importance — works for tree-based models
if hasattr(model, 'feature_importances_'):
    import pandas as pd
    importance = pd.Series(model.feature_importances_, index=feature_names)
    print(importance.sort_values(ascending=False))

# Coefficients — works for linear models
if hasattr(model, 'coef_'):
    print(model.coef_)

# Run additional cross-validation on the best model with different folds
from sklearn.model_selection import cross_val_score
scores = cross_val_score(model, X_train, y_train, cv=10)
print(f"10-fold CV: {scores.mean():.4f} +/- {scores.std():.4f}")
```

---

## 8. 🖥️ Running the Provided Scripts

###  automl_examples.py

Runs 5 separate experiments, prints detailed results for each, then compares all of them in a final ranked table and saves CSV files.

**📁 Step 1 — Create output directories**

```bash
mkdir outputs
mkdir outputs\examples
```

**✏️ Step 2 — Update save paths**

Open `automl_examples.py` and update two locations:

In `compare_results()` (around line 156):
```python
plt.savefig('outputs/examples/automl_comparison.png', dpi=300, bbox_inches='tight')
```

In `save_results_to_csv()` (around line 168):
```python
filename = f'outputs/examples/automl_results_{name.replace(" ", "_").lower()}.csv'
```

**▶️ Step 3 — Run all 5 examples**

```bash
python automl_examples.py
```

**🎯 Step 4 — Run a single example without running all five**

```python
from automl_examples import example_2_breast_cancer_classification

automl, summary = example_2_breast_cancer_classification()
print(summary.head())
```

**📋 The 5 examples:**

| Function | Dataset | Task | Trials | Samples | Features |
|---|---|---|---|---|---|
| `example_1_iris_classification()` | Iris | Classification | 25 | 150 | 4 |
| `example_2_breast_cancer_classification()` | Breast Cancer | Classification | 40 | 569 | 30 |
| `example_3_diabetes_regression()` | Diabetes | Regression | 40 | 442 | 10 |
| `example_4_synthetic_large_classification()` | Synthetic | Classification | 50 | 5000 | 50 |
| `example_5_synthetic_large_regression()` | Synthetic | Regression | 50 | 2000 | 30 |

---

## 9. 📊 Visualizations

### 📉 Optimization History

Generated by `automl.plot_optimization_history()`.

**Left panel — Trial Scores:** CV score for every trial. Points scattered widely early on show the optimizer is still exploring. Points clustering near the top later show it has found a good region.

**Right panel — Best Score Progress:** Monotonically increasing line showing the best score found so far at each trial. If this line flattens after trial 15, no improvement was found in the remaining trials — you could reduce `n_trials`. If it is still rising at the final trial, increasing `n_trials` would likely help.

### 📊 Model Comparison

Generated by `automl.plot_model_comparison(top_n=10)`.

Grouped bar chart with two bars per model: CV Score and Test Score. Numeric values printed on top of each bar.

**What to look for:** A large gap between CV Score and Test Score (more than ~0.05) indicates overfitting. Models where both scores are close and high are the most trustworthy.

### 🔬 Hyperparameter Importance

Generated by `automl.plot_hyperparameter_importance(top_n=10)`.

Scatter plots of numeric hyperparameter values across the top 10 models (up to 3 hyperparameters shown). Clustered points mean the optimizer converged on a specific value range — that range is important. Widely scattered points mean that hyperparameter has little effect on the score.

### 💾 Saving all plots

```python
import matplotlib.pyplot as plt
import os

os.makedirs('outputs', exist_ok=True)

fig1 = automl.plot_optimization_history()
fig1.savefig('outputs/history.png', dpi=300, bbox_inches='tight')

fig2 = automl.plot_model_comparison(top_n=10)
fig2.savefig('outputs/comparison.png', dpi=300, bbox_inches='tight')

fig3 = automl.plot_hyperparameter_importance(top_n=8)
if fig3:
    fig3.savefig('outputs/hyperparams.png', dpi=300, bbox_inches='tight')

plt.close('all')  # free memory
```

---

## 10. 📋 Output & Results Explained

### 🖨️ Reading the trial-by-trial log

```
Trial 23: AdaBoost | CV: 0.9822 | Test: 0.9736 | Time: 0.31s
```

- `Trial 23` — this is the 24th trial (0-indexed)
- `AdaBoost` — which model was tested
- `CV: 0.9822` — mean F1 (classification) or R² (regression) across all CV folds
- `Test: 0.9736` — score on the held-out test set
- `Time: 0.31s` — total wall-clock time for this trial including CV

### 🏆 Reading the final summary block

```
================================================================================
BEST MODEL: AdaBoost
Hyperparameters:
{
  "n_estimators": 106,
  "learning_rate": 1.07496901142235
}
CV Score: 0.9822 +/- 0.0208
Test Score: 0.9736
Train Score: 1.0000
================================================================================
```

**CV Score: 0.9822 +/- 0.0208** — mean of 0.9822 with std deviation of 0.0208 across the 5 folds. A low std (< 0.03) means consistent performance across folds. A high std (> 0.05) means performance varies a lot between folds, which can indicate the dataset is too small or the model is unstable.

**Test Score: 0.9736** — this is the number to report. It was computed on data the model never saw during training or optimization.

**Train Score: 1.0000** — perfect on training data. Combined with Test Score 0.9736, this shows mild overfitting, which is typical for boosting algorithms.

### 🗂️ Reading the summary DataFrame

```
                 Model  CV Score  Test Score  Train Score    CV Std  Train Time (s)  Trial
0             AdaBoost  0.982177    0.973621     1.000000  0.020762        0.306156     23
1             AdaBoost  0.980108    0.973621     1.000000  0.008361        0.266015     26
2  Logistic Regression  0.977937    0.973742     0.989004  0.018454        0.021617      9
```

- Rows are sorted by CV Score descending
- Multiple rows with the same model name are different hyperparameter configurations of that model
- The `Trial` column shows which Optuna trial this result came from
- `CV Std` — lower is better; indicates consistency across folds

---

## 11. 🔧 Troubleshooting

### ❌ `ModuleNotFoundError: No module named 'optuna'`

```bash
pip install optuna
```

### ❌ `ModuleNotFoundError: No module named 'sklearn'`

```bash
pip install scikit-learn
```

### 📂 `FileNotFoundError` when saving plots

The output directory must exist before saving. Create it first:

```bash
mkdir outputs
```

Or handle it in code:

```python
import os
os.makedirs('outputs', exist_ok=True)
fig.savefig('outputs/plot.png')
```

### 🐢 Trials are very slow

Gradient Boosting and Random Forest with many trees are the main culprits on large datasets.

```python
# Option 1 — Reduce total trials
automl.fit(X, y, n_trials=20)

# Option 2 — Add a time limit
automl.fit(X, y, n_trials=1000, timeout=120)

# Option 3 — Use fewer CV folds
automl.fit(X, y, n_trials=30, cv_folds=3)

# Option 4 — Disable parallelism (reduces memory pressure)
automl = ClassificationAutoML(n_jobs=1)
```

### 🎲 Results are different every time I run

Both `random_state` arguments must be set:

```python
automl = ClassificationAutoML(random_state=42)   # seeds Optuna + model init
automl.fit(X, y, random_state=42, n_trials=30)   # seeds train/test split
```

### ⚠️ `predict_proba` raises `ValueError`

SVM does not support `predict_proba` by default. If the best model is an SVM, either use `predict()` instead, or find the best model that supports probabilities:

```python
for result in automl.results:
    if hasattr(result.model, 'predict_proba'):
        proba_model = result.model
        break

probabilities = proba_model.predict_proba(automl.scaler.transform(X_new))
```

### 📉 `No numeric hyperparameters to plot`

`plot_hyperparameter_importance()` only plots numeric hyperparameters. If your top models all use only categorical hyperparameters (e.g., only SVM configurations), the function returns `None` and prints this message. This is not an error. Use the other two plot methods instead.

### 💾 Memory error on large datasets

```python
# Reduce parallelism
automl = ClassificationAutoML(n_jobs=1)

# Reduce trials and folds
automl.fit(X, y, n_trials=20, cv_folds=3)
```

---

## 📄 License

MIT License — free for academic and commercial use.
