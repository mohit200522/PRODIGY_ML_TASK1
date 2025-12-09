**PRODIGY INFOTECH_ML_TASK_01**
# 🏡 House Price Prediction using Linear Regression

A machine learning project that predicts **house prices** based on features such as **square footage**, **number of bedrooms**, and **bathrooms** using **Linear Regression**.

---

## 📌 **Project Overview**

This project implements a **Linear Regression model** to predict house prices.
It is part of the **Prodigy Infotech Machine Learning Internship – Task 01**.

The goal is to:

* Load and understand the dataset
* Preprocess the data
* Build a linear regression model
* Evaluate performance
* Visualize results

---

## 📂 **Dataset**

You can use the dataset provided in the repository:

### Files:

```
data.csv       → training data (features)
output.csv     → target values / actual house prices
data.dat       → raw/extra data (not required for training)
```

### Important Columns Used:

* **SquareFootage**
* **Bedrooms**
* **Bathrooms**
* **Price** (target)

---

## 🚀 **Technologies Used**

* Python
* NumPy
* Pandas
* Matplotlib
* Seaborn
* Scikit-Learn

---

## 🧠 **Model Used**

### **Linear Regression**

A supervised ML algorithm used to model the relationship between dependent and independent variables.

**Why Linear Regression?**

* Easy to interpret
* Works well for continuous numerical prediction
* Fast and efficient

---

## 🛠️ **Project Workflow**

1. Import libraries
2. Load dataset
3. Clean & preprocess data
4. Select important features
5. Split training and testing sets
6. Train the Linear Regression model
7. Evaluate performance using:

   * Mean Absolute Error (MAE)
   * Mean Squared Error (MSE)
   * R² Score
8. Visualize predictions vs. actual values

---

## 📈 **Results**

The model predicts house prices based on the given features.
Visualizations include:

* Scatter plots
* Regression line
* Model performance comparison

---

## 📊 **Example Code Snippet**

```python
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import pandas as pd

# Load data
X = pd.read_csv("data.csv")
y = pd.read_csv("output.csv")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model
model = LinearRegression()
model.fit(X_train, y_train)

# Prediction
predictions = model.predict(X_test)
```

---

## 📦 **How to Run**

```bash
1. Clone the repo
2. Install dependencies → pip install -r requirements.txt
3. Run the notebook or script
```

---

## 🤝 **Contributing**

Contributions, issues, and suggestions are welcome!

---

## 🧑‍💻 **Author**

**Mohit Vishwakarma**
Machine Learning Intern – Prodigy Infotech

---

## ⭐ If you like this project, don't forget to star the repository!

---
