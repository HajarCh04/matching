import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Charger le fichier CSV avec encodage ISO-8859-1
file_path = 'C:/Users/USER/Desktop/matching-automatique/ML/data.csv'
data = pd.read_csv(file_path, encoding='ISO-8859-1')

# Nettoyer les données en supprimant les lignes avec des valeurs manquantes dans les colonnes nécessaires
data_cleaned = data.dropna(
    subset=['Metamodel1', 'Element1', 'Metamodel2', 'Element2', 'Correspondence'])

# Convertir les colonnes catégorielles en numériques avec l'encodage one-hot
data_encoded = pd.get_dummies(
    data_cleaned, columns=['Metamodel1', 'Element1', 'Metamodel2', 'Element2'])

# Séparer les caractéristiques (X) et la cible (y)
X = data_encoded.drop(
    columns=['Correspondence', 'Description1', 'Description2'])
y = data_encoded['Correspondence']

# Diviser le jeu de données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Initialiser et entraîner le classificateur Gradient Boosting
gbc = GradientBoostingClassifier(random_state=42)
gbc.fit(X_train, y_train)

# Faire des prédictions sur l'ensemble de test
y_pred = gbc.predict(X_test)

# Calculer les métriques de qualité
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# Afficher les résultats
print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")
