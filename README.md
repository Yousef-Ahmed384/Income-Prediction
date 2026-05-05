# Income-Prediction
FCIS ML Project in AI Course that Classify based on prediction the income of Employees based on some Features.

---

# 📊 Adult Income Prediction: End-to-End Machine Learning Project

Welcome to the documentation for the **Income Prediction Project**. This project aims to build a machine learning pipeline that predicts whether an individual earns more than $50,000 a year based on census data (like age, occupation, and education). 

This README provides a clear, step-by-step explanation of the data preprocessing, the machine learning models used, and the evaluation techniques applied.

---


## 🧹 1. Data Preprocessing & Techniques

Before feeding data to an AI model, it must be cleaned and formatted. Think of this as preparing the ingredients before cooking. Here are the steps and techniques we used:

* **Handling Missing Values & Duplicates:** * We identified missing values marked as `?` and dropped them because they accounted for a small percentage (about 7.5%) of the data. We also removed duplicate rows to prevent the model from learning redundant patterns.
  * *Redundancy check:* We dropped the `education` column because the `education-num` column provides the exact same information but in a more model-friendly numerical format.

* **Encoding Categorical Data:** Machine learning models only understand numbers, not text.
  * **One-Hot Encoding (`pd.get_dummies`):** Used for categories without a specific order (like `workclass`, `occupation`, `race`). It creates a new binary (0 or 1) column for each category.
  * **Binary Mapping:** For columns with only two options, we mapped them manually using dictionaries (e.g., `Male: 1, Female: 0` and `>50K: 1, <=50K: 0`).

* **Outlier Handling (Winsorizing/Clipping):** * We calculated the **Interquartile Range (IQR)** for the `fnlwgt` (final weight) column. Extreme values (too high or too low) were "clipped" to a maximum and minimum boundary. This prevents wild outliers from confusing the models.

* **Feature Selection (Correlation):**
  * We calculated how strongly each feature relates to the `Income` target. We kept only the "Top Features" that had an absolute correlation greater than `0.05`. This removes noisy, unhelpful data.

* **Feature Scaling (`StandardScaler`):**
  * **What it is:** It shrinks all numerical columns so they share the same scale (mean of 0).
  * **How we used it:** We used `.fit()` on the training data to learn the rules, and `.transform()` on the train, validation, and test sets to apply the rules. This prevents "Data Leakage" (cheating by looking at the test test before the exam).

* **Test Data Alignment:**
  * We used `.reindex()` on the test dataset to ensure it had the exact same columns as the training dataset.

---

## 🤖 2. Classification Models: General Overview

**What is Classification?**
Classification is a type of Supervised Machine Learning where the goal is to predict a category. Since our target is binary (`>50K` or `<=50K`), this is a **Binary Classification** problem.

**Models We Used:**
1. Support Vector Machine (SVM)
2. Decision Tree
3. Random Forest
4. Logistic Regression

**Models We Did NOT Use (For Context):**
* *K-Nearest Neighbors (KNN):* Looks at the closest data points to make a guess. It can be very slow with large datasets like ours.
* *Naive Bayes:* Based on probability. It assumes all features are completely independent, which isn't true for our data (e.g., education and occupation are highly related).

---

## 🛠️ 3. Model Implementation Details

Here is how we trained and tested our chosen models. For every model, we trained it on `x_train_scaled`, validated it on `x_val_scaled`, and did a final check on `x_test_scaled`.

### A. Support Vector Machine (SVM)
* **Concept:** SVM tries to draw a "line" (or a complex boundary) in space that best separates the low-income and high-income groups.
* **How we used it:** * We used the **RBF (Radial Basis Function)** kernel, which is excellent for data that cannot be separated by a simple straight line.
  * We set hyperparameters `C = 1` and `gamma = 0.01` to balance the model so it doesn't overthink the training data (overfitting).

### B. Decision Tree
* **Concept:** It works like a flowchart, asking a series of True/False questions (e.g., "Is age > 30?") to split the data until it reaches a prediction.
* **How we used it:**
  * If a tree is allowed to grow too deep, it memorizes the data perfectly but fails on new data.
  * We tested depths from 1 to 9. By plotting the training vs. validation accuracy, we discovered that a **`max_depth` of 8** provided the best accuracy without overfitting. We used the `entropy` criterion to measure the quality of the splits.

### C. Random Forest
* **Concept:** An "ensemble" method. Instead of relying on one Decision Tree, it builds hundreds of them and lets them "vote" on the final prediction. It is very powerful and accurate.
* **How we used it:**
  * We used `GridSearchCV`, a technique that automatically tests hundreds of different parameter combinations to find the absolute best setup.
  * **Best Parameters Found:** `criterion = 'gini'`, `max_depth = 17`, and `n_estimators = 250` (meaning 250 trees voting together).

### D. Logistic Regression
* **Concept:** Despite the word "regression," this is a classification model. It uses a mathematical formula to calculate the probability (from 0% to 100%) that a person makes >50K.
* **How we used it:**
  * It is a fantastic, fast baseline model. We used the `liblinear` solver, which works very well for binary datasets.
  * We set the regularization strength to `C = 10`.

---

## 📈 4. Evaluation Metrics
To know if our models are actually good, we used these tools:
* **Accuracy Score:** The overall percentage of correct predictions.
* **Classification Report:** Shows deeper metrics:
  * *Precision:* When the model guesses >50K, how often is it right?
  * *Recall:* Out of all the people who actually make >50K, how many did the model find?
  * *F1-Score:* A balanced average of Precision and Recall.
* **Confusion Matrix:** A heat-map visualization showing the exact number of True Positives, True Negatives, False Positives, and False Negatives.
