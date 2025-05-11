# E-commerce Shoppers’ Behaviour Understanding

## 📄 Competition Link

[E-commerce Shoppers’ Behaviour Understanding on Kaggle](https://www.kaggle.com/competitions/e-commerce-shoppers-behaviour-understanding/overview)

## 🏆 Overview

Your client runs an e-commerce platform and has collected one year of user session data to understand shopping behavior. Each row corresponds to a unique user session, with a binary target `Made_Purchase` indicating whether the user purchased anything during the year. The data may contain errors, as it was gathered by non-experts.

* **Task:** Predict `Made_Purchase` (True/False) per session
* **Metric:** Mean F1-Score (harmonic mean of precision & recall)

## 📂 Dataset Description

* **Train Set:** `train.csv` (sessions with `Made_Purchase` labels)
* **Test Set:** `test.csv` (sessions without labels)
* **Submission:** `sample_submission.csv` with `id,Made_Purchase`

### Key Features (21 total)

| Feature                           | Description                                             |
| --------------------------------- | ------------------------------------------------------- |
| `HomePage`                        | Times visited home page                                 |
| `HomePage_Duration`               | Total time spent on home page                           |
| `LandingPage`                     | Times entered via landing page                          |
| `LandingPage_Duration`            | Total time on landing page                              |
| `ProductDescriptionPage`          | Times visited product description page                  |
| `ProductDescriptionPage_Duration` | Total time on product description page                  |
| `GoogleMetric-BounceRate`         | % sessions with single-page visits (bounce)             |
| `GoogleMetric-ExitRate`           | % exits from pages other than first landing             |
| `GoogleMetric-PageValue`          | Average value of pages leading to a transaction         |
| `SeasonalPurchase`                | Indicator for purchase during special seasonal period   |
| `Month_SeasonalPurchase`          | Month number for seasonal purchase (e.g., Diwali month) |
| `OS`                              | Operating system of the user’s device                   |
| `SearchEngine`                    | Source search engine                                    |
| `Zone`                            | Geographic or site zone identifier                      |
| `TypeOfTraffic`                   | Traffic source type (e.g., organic, paid)               |
| `CustomerType`                    | New or returning customer                               |
| `Gender`                          | User gender                                             |
| `CookiesSetting`                  | User’s cookie consent configuration                     |
| `Education`                       | Education level                                         |
| `MaritalStatus`                   | Marital status                                          |
| `WeekendPurchase`                 | Indicator if session occurred on weekend                |
| `Made_Purchase` (target)          | True if user purchased, False otherwise (train only)    |

## 🛠️ Environment & Dependencies

* **Python 3.8+**
* **pandas**, **numpy**, **matplotlib**, **seaborn**
* **scikit-learn** (`LogisticRegression`, metrics, model\_selection)
* **XGBoost**, **LightGBM** (optional boosting)
* **joblib** (model serialization)
* **Jupyter Notebook** / **Kaggle Notebook**

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost lightgbm joblib
```

## 🔍 Exploratory Data Analysis (EDA)

1. **Missing & Erroneous Values:** identify nulls, unrealistic durations, out-of-range metrics
2. **Distribution Plots:** histograms for numeric features (durations, counts)
3. **Categorical Counts:** bar plots for OS, Traffic Type, Customer Type, etc.
4. **Correlation Heatmap:** numeric feature correlations & target relationships
5. **Feature Engineering Ideas:** session-level aggregates, time-based flags, error-correction heuristics

## 🏗️ Modeling Approaches

### 1. Baseline: Logistic Regression

* **Preprocessing:** one-hot encode categorical vars, scale numeric features
* **Model:** `LogisticRegression(class_weight='balanced', solver='saga')`
* **Pros:** interpretable, fast
* **Cons:** limited nonlinearity capture

### 2. Tree-Based Models

* **Random Forest** (`n_estimators=100`, `max_depth=10`)
* **XGBoost** (`XGBClassifier` with `scale_pos_weight`)
* **LightGBM** (`LGBMClassifier` with `class_weight`)
* **Pros:** handle nonlinear interactions, robust to mixed data types
* **Cons:** slower on large data

### 3. Neural Network (Optional)

* **Architecture:** 2–3 dense layers with ReLU, dropout, batch normalization
* **Loss:** binary cross-entropy
* **Optimizer:** Adam

#### Feature Engineering & Selection

* Correct or clip abnormal durations
* Create ratios (e.g., `Duration/Visits`)
* Encode seasonality: `Month_SeasonalPurchase` → cyclic features (sine, cosine)
* Combine related metrics: bounce × exit rates

## 🎓 Training & Validation Pipeline

1. **Split Data:** stratified K-fold (e.g., K=5) on `Made_Purchase`
2. **Preprocessing Pipeline:** sklearn `ColumnTransformer` for numeric & categorical
3. **Model Training:** per-fold training with early stopping (for boosting)
4. **Evaluation:** compute F1 per fold, average as validation score
5. **Ensembling:** average or majority vote across models/folds (optional)
6. **Hyperparameter Tuning:** grid search or Bayesian optimization on F1

## 📊 Evaluation

* **Metric:** Mean F1-Score across folds or full test set
* Use `sklearn.metrics.f1_score(y_true, y_pred, average='binary')`
* **Confusion Matrix:** inspect false positives vs. false negatives

## 🚀 Inference & Submission

1. **Train Final Model:** on full training set or ensemble of best-fold models
2. **Predict Test Set:** output boolean or 0/1 values for `Made_Purchase`
3. **Format CSV:**

   ```csv
   id,Made_Purchase
   1,False
   2,True
   3,False
   ...
   ```
4. **Submit:** upload to Kaggle competition page

## 💡 Skills & Techniques Demonstrated

* Data cleaning & error handling
* Feature engineering & encoding mixed data types
* Model selection & ensembling
* Stratified K-fold cross-validation
* Hyperparameter optimization for F1 metric

## 🏃 How to Reproduce

1. Clone this repo and navigate to `notebooks/`.
2. Install dependencies.
3. Place `train.csv` & `test.csv` under `data/`.
4. Run EDA notebook: `notebooks/eda_ecommerce.ipynb`.
5. Train models in `notebooks/train_models.ipynb`.
6. Execute `notebooks/inference.ipynb` or `submission.py` to generate submission file.

---

**Deep dive into shopper behavior and drive smarter e-commerce decisions!**
