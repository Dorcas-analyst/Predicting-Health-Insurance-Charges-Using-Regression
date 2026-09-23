# Predicting-Health-Insurance-Charges-Using-Regression-2026
Using Python, pandas, NumPy, Matplotlib, Seaborn, and scikit-learn, I explored how factors such as age, BMI, smoking status, number of children, sex, and residential region relate to medical insurance charges and compared multiple regression algorithms to determine which model best captured the underlying structure of the data.

## Project Overview
This project analysed **1,338 health insurance policyholder records** to investigate which demographic and lifestyle factors were associated with medical insurance charges and to build machine learning models capable of predicting those charges.
The analysis was completed independently using **Python, pandas, NumPy, Matplotlib, Seaborn, and scikit-learn**.

## Objective
The project aimed to:
* Explore demographic, lifestyle, and insurance-cost variables.
* Assess data quality and distributions.
* Encode categorical variables appropriately.
* Identify the strongest predictors of insurance charges.
* Build and compare regression models.
* Validate model performance using test-set evaluation, cross-validation, and residual analysis.
* Investigate whether non-linear relationships or feature interactions affected model performance.

## Dataset
The dataset contained seven original variables:
* `age`
* `sex`
* `bmi`
* `children`
* `smoker`
* `region`
* `charges` — prediction target
There were **no missing values**, so no imputation was required.

## Data Preparation
Categorical variables were encoded according to their structure:
* Binary encoding for `sex` and `smoker`
* One-hot encoding for `region`
* `drop_first=True` to establish a baseline category
The final modelling dataset contained eight predictors. Features were standardized using `StandardScaler`, followed by an **80/20 train-test split** with `random_state=42`.

## Exploratory Data Analysis
Correlation analysis identified smoking status as the strongest linear predictor of insurance charges:
* **Smoking status:** +0.79
* **Age:** +0.30
* **BMI:** +0.20
* **Region Southeast:** +0.07
* **Children:** +0.07
* **Sex:** +0.06
The analysis also showed that insurance charges were strongly right-skewed.

## Models Compared
Four regression algorithms were evaluated:
| Model             |   Test R² | Corrected RMSE |
| ----------------- | --------: | -------------: |
| Linear Regression |     0.784 |      $5,796.28 |
| Ridge Regression  |     0.784 |      $5,796.98 |
| Lasso Regression  |     0.784 |      $5,796.32 |
| **Random Forest** | **0.873** |  **$4,436.24** |

Random Forest produced the strongest performance on the held-out test set.
Five-fold cross-validation of Linear Regression produced a mean R² of **0.747**, providing a more conservative estimate than its single test-set score of 0.784.

## Key Analytical Finding
The most important finding was not simply the strong association between smoking status and charges.
Visual analysis of **BMI versus charges, separated by smoking status**, revealed a clear interaction between the two variables. At higher BMI levels, smokers occupied a substantially higher charge band than non-smokers.

This interaction helped explain:
* why the linear model's residuals showed systematic patterns;
* why the charges distribution had a long upper tail;
* and why Random Forest outperformed the linear-based models.

## Skills Demonstrated
**Python · pandas · NumPy · Matplotlib · Seaborn · scikit-learn · Data Cleaning · Exploratory Data Analysis · Feature Engineering · Categorical Encoding · Feature Scaling · Regression · Random Forest · Model Evaluation · Cross-Validation · Residual Analysis · Data Visualisation · Statistical Interpretation**

## Methodological Improvements
The report identified several improvements for future iterations:
* Build the preprocessing stage with `Pipeline`.
* Fit the scaler only on training data to eliminate scaling leakage.
* Engineer an explicit `smoker × BMI` interaction term.
* Test a log transformation of `charges`.
* Tune Random Forest hyperparameters using `GridSearchCV`.
* Compare Random Forest feature importance against correlation-based rankings.

## Key Takeaway
This project demonstrated that **model selection should reflect the underlying structure of the data**. The combination of correlation analysis, targeted visualisation, residual diagnostics, and cross-validation provided evidence that the insurance-charge relationship was not purely additive or linear.

The project therefore went beyond simply asking *which model scored highest?* and instead investigated **why one model was better suited to the data**.
