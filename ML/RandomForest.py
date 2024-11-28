import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
import numpy as np

# Load the dataset
file_path = 'C:/Users/USER/Desktop/matching-automatique/ML/data.csv'
data = pd.read_csv(file_path, encoding='ISO-8859-1')

# Drop rows with missing target values
data = data.dropna(subset=['Correspondence'])

# Separate features and target
X = data.drop(columns=['Correspondence'])
y = data['Correspondence']

# Fill missing values for non-target columns and encode categorical data
# Using SimpleImputer to replace NaNs with 'missing' for categorical data
imputer = SimpleImputer(strategy="constant", fill_value="missing")
X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

# Label encoding for categorical features
label_encoders = {col: LabelEncoder() for col in X_imputed.columns}
for col, encoder in label_encoders.items():
    X_imputed[col] = encoder.fit_transform(X_imputed[col])

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X_imputed, y, test_size=0.2, random_state=42)

# Initialize and train the Random Forest Classifier
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)

# Predict on test set
y_pred = rf_model.predict(X_test)
# Probabilities for the positive class
y_proba = rf_model.predict_proba(X_test)[:, 1]

# Calculate evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=1)
recall = recall_score(y_test, y_pred, zero_division=1)
f1 = f1_score(y_test, y_pred, zero_division=1)
roc_auc = roc_auc_score(y_test, y_proba)

# Print metrics
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
print("ROC-AUC:", roc_auc)
