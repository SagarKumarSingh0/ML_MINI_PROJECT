import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# -------------------------------
# Step 1: Create Synthetic Dataset
# -------------------------------
np.random.seed(42)

data = pd.DataFrame({
    'CIBIL_Score': np.random.randint(300, 900, 1000),
    'Annual_Income': np.random.randint(100000, 2000000, 1000),
    'Loan_Amount': np.random.randint(50000, 1000000, 1000),
    'Tenure_Years': np.random.randint(1, 30, 1000),
    'Existing_Loans': np.random.randint(0, 5, 1000),
    'Loan_Approved': np.random.choice([0, 1], 1000, p=[0.4, 0.6])
})

# -------------------------------
# Step 2: Exploratory Data Analysis
# -------------------------------
print("Dataset Overview:")
print(data.head())

# Correlation heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
plt.title("Feature Correlation Heatmap")
plt.show()

# CIBIL Score vs Loan Approval
sns.boxplot(x='Loan_Approved', y='CIBIL_Score', data=data)
plt.title('CIBIL Score vs Loan Approval')
plt.show()

# -------------------------------
# Step 3: Preprocessing
# -------------------------------
X = data.drop('Loan_Approved', axis=1)
y = data['Loan_Approved']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# -------------------------------
# Step 4: Train Model
# -------------------------------
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# -------------------------------
# Step 5: Evaluate Model
# -------------------------------
y_pred = model.predict(X_test)

print("\nModel Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# -------------------------------
# Step 6: Predict New Application
# -------------------------------
new_applicant = pd.DataFrame({
    'CIBIL_Score': [750],
    'Annual_Income': [1200000],
    'Loan_Amount': [500000],
    'Tenure_Years': [10],
    'Existing_Loans': [1]
})

new_applicant_scaled = scaler.transform(new_applicant)
loan_status = model.predict(new_applicant_scaled)

print("\nNew Applicant Prediction:")
print("✅ Loan Approved" if loan_status[0] == 1 else "❌ Loan Rejected")