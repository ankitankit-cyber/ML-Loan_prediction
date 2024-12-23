# Loan Prediction Machine Learning Project

This repository contains a machine learning project aimed at predicting loan approvals based on applicant data. The project follows a step-by-step approach, covering data preprocessing, model training, evaluation, and results.

---

## **1. Objective**
The primary goal is to create a predictive model that determines whether a loan application should be approved or rejected based on specific features of the applicant.

---

## **2. Project Workflow**
The project workflow is divided into the following steps:

### **Step 1: Data Collection**
- Load the dataset containing features such as income, credit history, loan amount, etc.
- Ensure the dataset is available in CSV format and placed in the project directory.

### **Step 2: Data Preprocessing**
- Handle missing values.
- Perform feature engineering to encode categorical variables.
- Normalize or scale numeric features for model compatibility.

### **Step 3: Train-Test Split**
- Split the dataset into training (80%) and testing (20%) subsets.
- Use the training set to train the model and the testing set to evaluate its performance.

### **Step 4: Exploratory Data Analysis (EDA)**
- Analyze the distribution of data using:
  - Summary statistics.
  - Visualizations like histograms, box plots, and scatter plots.
- Investigate relationships between features and the target variable.

### **Step 5: Model Training**
- Train various machine learning models:
  - Logistic Regression
  - Decision Trees
  - Random Forests
  - Support Vector Machines (SVM)
- Use cross-validation to fine-tune hyperparameters.

### **Step 6: Model Evaluation**
- Evaluate models using metrics such as:
  - Accuracy
  - Precision
  - Recall
  - F1 Score
- Select the model with the best balance of metrics.

### **Step 7: Predictions**
- Test the final model with unseen data.
- Generate predictions for loan approvals.

---

## **3. Technologies Used**
- **Languages:** Python
- **Tools:**
  - Jupyter Notebook, google colab
- **Libraries:**
  - `NumPy` and `Pandas` for data manipulation
  - `Matplotlib` and `Seaborn` for data visualization
  - `Scikit-learn` for machine learning models


