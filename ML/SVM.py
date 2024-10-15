import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix

# Load the dataset
data_path = 'C:/Users/USER/Desktop/matching-automatique/ML/data.csv'
data = pd.read_csv(data_path, encoding='ISO-8859-1')

# Rename columns for clarity
data.columns = ['Metamodel1', 'Element1', 'Description1',
                'Metamodel2', 'Element2', 'Description2', 'Correspondence']

data = data.dropna(subset=['Correspondence'])

# Separate features and target variable
X = data[['Metamodel1', 'Element1', 'Description1',
          'Metamodel2', 'Element2', 'Description2']]
y = data['Correspondence']

# Encode categorical features using one-hot encoding
X_encoded = pd.get_dummies(X)

# Split into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y, test_size=0.2, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Initialize the SVM model (using the 'rbf' kernel as an example)
svm_model = SVC(kernel='rbf', C=1, gamma='scale', random_state=42)

# Train the model on the training set
svm_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = svm_model.predict(X_test)


# Check for missing values in the target variable
missing_values = y.isnull().sum()
print(f"Number of missing values in the target variable: {missing_values}")


# Calculate Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# Calculate Precision
# Use 'binary' for binary classification if needed
precision = precision_score(y_test, y_pred, average='weighted')
print(f"Precision: {precision:.2f}")

# Calculate Recall
recall = recall_score(y_test, y_pred, average='weighted')
print(f"Recall: {recall:.2f}")

# Calculate F1 Score
f1 = f1_score(y_test, y_pred, average='weighted')
print(f"F1-Score: {f1:.2f}")

# Print a detailed classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Confusion Matrix
print("\nConfusion Matrix:")
conf_matrix = confusion_matrix(y_test, y_pred)
print(conf_matrix)
