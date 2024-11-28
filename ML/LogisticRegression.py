import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

# Load the dataset
file_path = 'C:/Users/USER/Desktop/matching-automatique/ML/data.csv'
data = pd.read_csv(file_path, encoding='ISO-8859-1')

# Drop rows with missing target values
data_cleaned = data.dropna(subset=['Correspondence']).copy()

# Separate features and target
X = data_cleaned[['Metamodel1', 'Element1', 'Description1',
                  'Metamodel2', 'Element2', 'Description2']]
y = data_cleaned['Correspondence']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42)

# Column transformer to handle categorical and text data
preprocessor = ColumnTransformer(
    transformers=[
        ('metamodel', OneHotEncoder(), ['Metamodel1', 'Metamodel2']),
        ('elements', TfidfVectorizer(), 'Element1'),
        ('descriptions', TfidfVectorizer(), 'Description1')
    ],
    remainder='drop'
)

# Create pipeline with logistic regression
pipeline = make_pipeline(preprocessor, LogisticRegression(
    max_iter=1000, random_state=42))

# Train the model
pipeline.fit(X_train, y_train)

# Make predictions
y_pred = pipeline.predict(X_test)
y_pred_proba = pipeline.predict_proba(
    X_test)[:, 1]  # Probabilities for ROC AUC

# Calculate quality metrics
metrics = {
    'accuracy': accuracy_score(y_test, y_pred),
    'precision': precision_score(y_test, y_pred),
    'recall': recall_score(y_test, y_pred),
    'f1_score': f1_score(y_test, y_pred),
    'roc_auc': roc_auc_score(y_test, y_pred_proba)
}

print(metrics)
