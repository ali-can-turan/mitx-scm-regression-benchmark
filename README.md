# 📦 Domain-Constrained SCM Regression Benchmark
> **Evaluating Domain-Informed Linear Models vs. Hyperparameter-Driven, Domain-Excluded Lasso Pipelines in Small-Sample Supply Chain Analytics ($N=100$)**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.2+-orange.svg)](https://scikit-learn.org/)
[![Course Benchmark](https://img.shields.io/badge/Benchmark-MITx--SC0x-red.svg)](https://www.edx.org/course/supply-chain-analytics)

---

## 📌 Project Overview

This repository reproduces and expands upon the baseline linear regression framework introduced in the **MITx MicroMasters in Supply Chain Management (SC0x: Supply Chain Analytics)** course. 

Using the course's underlying dataset ($N=100$), this benchmark compares two distinct modeling philosophies:

1. **ML-1 (Domain-Informed Linear Model):** Built on baseline course conditions (unscaled features, native units) enhanced with targeted domain extensions:
   * Exploration of 2nd-degree polynomial structures and feature interactions.
   * Transformation of lead time into ordinal/binary feature representations.
   * Enforced non-negativity constraints (`positive=True`) aligned with physical business logic.
2. **ML-2 (Domain-Excluded Lasso Pipeline):** A fully automated hyperparameter pipeline leveraging `StandardScaler`, polynomial expansion ($degree \in [1, 2, 3]$), interaction terms, and $L_1$ regularization via nested `LassoCV`.

### 💡 Core Takeaway
> **Domain-informed regularization outperforms automated algorithmic regularization in small-sample contexts.** 
> Incorporating physical business logic directly into the solver prevents overfitting far more effectively than brute-force grid searches on small datasets ($N=100$).

---

## 📊 Benchmark Results

| Model | Pipeline Architecture | Train (CV) $R^2$ | Validation $R^2$ | Train $R^2$ | Test $R^2$ | Test Adj. $R^2$ | Relative Bias (NMB) | Unit Interpretability |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **ML-1 (Winner)** | **Business Logic + Unscaled Features** | **0.84** | **0.80** | **0.83** | **0.72** | **0.71** | **~1.8%** | **Full (Native Units)** |
| ML-2 | Scaled + Poly(d=3) + `LassoCV` | 0.84 | 0.80 | 0.86 | 0.58 | 0.55 | ~2.3% | Lost (Standardized) |

*Note: Models were evaluated on a 90/10 split ($N_{\text{train}}=90, N_{\text{test}}=10$) with 5-fold cross-validation during training.*

---

## 🧠 Methodology & Key Insights

### 1. Business Logic Mapping (ML-1)
Standard regression algorithms can yield contradictory coefficients (e.g., higher prices leading to increased demand). To enforce physical plausibility without artificial penalty parameters:
* Variables with expected inverse relationships (e.g., lead time) were sign-inverted prior to estimation.
* A strict non-negativity constraint (`positive=True`) was applied directly to the regression solver.
* Features remained **unscaled** to maintain direct physical interpretation ($y = w_1 X_1 + w_2 X_2 + \dots$).

### 2. GridSearch Hyperparameter Exploration (ML-2)
A brute-force hyperparameter search across **1,250 execution runs** was implemented, covering:
* Polynomial expansion up to $degree = 3$.
* Feature interaction terms.
* Scaled (`StandardScaler`) vs. Unscaled (`passthrough`) pipeline paths.
* Multi-grid $L_1$ regularization penalties ($\alpha$ parameters).

> **Key Discovery:** Unconstrained hyperparameter tuning resulted in severe overfitting—causing a **22% performance drop** in Test Adjusted $R^2$ (falling from **0.71** in ML-1 down to **0.55** in ML-2).

### 3. Residual Diagnostics ($N_{\text{test}}=10$)
Due to low statistical power at $N_{\text{test}}=10$, formal hypothesis testing (e.g., Shapiro-Wilk, Breusch-Pagan) was substituted with qualitative diagnostic evaluations:
* **Normality:** No extreme skewness or heavy-tailed outliers observed visually.
* **Linearity:** Residuals are evenly dispersed across zero (5 positive, 5 negative). Minor residual trendline slope was identified as an artifact of high sample variance in small $N$, rather than true systemic non-linearity.
* **Independence:** Slight negative autocorrelation between residuals is visually observable.
* **Homoscedasticity:** Dispersion of residual values remained uniform across predicted Cost-Per-Lead (CPL) ranges.

---
