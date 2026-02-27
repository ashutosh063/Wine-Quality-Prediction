GROUP - 1
BRANCH - CSE
SEM - 4TH 
TOPIC - Wine Quality Classification

TEAM MEMBERS -

ASHUTOSH BEHERA - 2401020063
SATYAJIT PRADHAN - 2401020011
DILSON KUMAR JENA - 2401020022
AMIT KUMAR - 2401020040





# 🍷 Wine Quality Classification & Prediction App

## Overview
This is an end-to-end Machine Learning mini-project that predicts the quality of wine (Low, Medium, or High) based on its physicochemical properties. This project encompasses the entire ML pipeline: Exploratory Data Analysis (EDA), Data Preprocessing, Model Training, Model Evaluation, and finally, deployment using a Streamlit web interface.

## 🎯 Features
* **Exploratory Data Analysis:** Visualized feature distributions and correlations using Seaborn and Matplotlib.
* **Data Preprocessing:** Handled class imbalances, grouped target variables, and applied `StandardScaler` for feature scaling.
* **Machine Learning Models:** Trained and compared **Logistic Regression** and **Random Forest Classifier**.
* **Model Evaluation:** Analyzed models using Accuracy, Precision, Recall, F1-Score, and Confusion Matrices.
* **Interactive UI:** A user-friendly web application built with Streamlit that allows users to input chemical parameters and receive real-time wine quality predictions.

## 🛠️ Tech Stack
* **Language:** Python 3
* **Data Manipulation:** Pandas, NumPy
* **Machine Learning:** Scikit-Learn
* **Data Visualization:** Matplotlib, Seaborn
* **Web Deployment:** Streamlit

## 📂 Dataset
The dataset used for this project is the **Wine Quality Dataset**. It contains 11 chemical features (like pH, alcohol, and acidity) and a target variable (`quality`).

## 🚀 How to Run Locally
1. **Clone the repository:**
   `git clone https://github.com/ashutosh063/Wine-Quality-Prediction.git`
2. **Navigate to the project directory:**
   `cd Wine-Quality-Prediction`
3. **Install the required dependencies:**
   `pip install streamlit pandas numpy scikit-learn joblib`
4. **Run the Streamlit app:**
   `streamlit run app.py`

## 🏆 Results

After evaluating the models, the **Random Forest Classifier** outperformed Logistic Regression, making it the final model chosen for the web application deployment.

