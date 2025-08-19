# Loan-Approval-Prediction
Predict loan approval using Kaggle data. Clean and encode features, handle imbalance with SMOTE, train Logistic Regression, Decision Tree, and Random Forest, and evaluate with precision, recall, F1-score, and confusion matrices. Bonus: compare LR vs. Decision Tree.


This task predicts whether a loan application will be approved using the Loan Approval Prediction dataset from Kaggle. The dataset contains demographic and financial attributes such as education, employment status, income, and loan details.

The pipeline starts with data cleaning, including handling missing values and normalizing column names. Categorical variables like education, self_employment, and loan status are encoded using LabelEncoder. Features are then separated from the target variable (loan_status).

Since loan approval data is often imbalanced (fewer approvals than rejections), the task applies SMOTE (Synthetic Minority Oversampling Technique) to balance the classes. Features are also scaled using StandardScaler for better model performance.

The dataset is split into training and testing sets. Different classification models are trained:

Random Forest Classifier

Logistic Regression

Decision Tree Classifier

Model performance is evaluated with precision, recall, F1-score, and confusion matrices, providing insights into both false positives and false negatives.

As a bonus, comparisons are made between Logistic Regression vs. Decision Tree, showing trade-offs in interpretability and performance. Visualizations of confusion matrices allow for a clearer assessment of each model’s predictions.

This task emphasizes binary classification with imbalanced data and demonstrates effective handling of real-world challenges in predictive modeling.
