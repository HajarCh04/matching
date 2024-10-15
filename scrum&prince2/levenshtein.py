import numpy as np
import pandas as pd
from Levenshtein import distance

# Étape 1: Extraire les éléments
scrum_elements = ["ProductBacklog", "Sprint", "ScrumTeam",
                  "SprintBacklog", "Increment", "DailyScrum"]
prince2_elements = ["BusinessCase", "ProjectPlan",
                    "ProjectBoard", "StagePlan", "WorkPackage", "EndStageReport"]

# Étape 2: Correspondances réelles
real_matches = {
    "ProductBacklog": "BusinessCase",
    "Sprint": "ProjectPlan",
    "ScrumTeam": "ProjectBoard",
    "SprintBacklog": "WorkPackage",
    "Increment": "EndStageReport",
    "DailyScrum": "StagePlan"
}

# Étape 3: Calculer la similarité de Levenshtein
similarity_matrix = np.zeros((len(scrum_elements), len(prince2_elements)))

for i, scrum_elem in enumerate(scrum_elements):
    for j, prince2_elem in enumerate(prince2_elements):
        max_len = max(len(scrum_elem), len(prince2_elem))
        similarity_matrix[i, j] = 1 - \
            distance(scrum_elem, prince2_elem) / max_len

# Afficher la matrice de similarité
similarity_df = pd.DataFrame(
    similarity_matrix, index=scrum_elements, columns=prince2_elements)
print("Matrice de Similarité:")
print(similarity_df)

# Sauvegarder la matrice dans un fichier CSV pour consultation
# similarity_df.to_csv('/mnt/data/similarity_matrix.csv')

# Étape 5 et 6: Essayer différents seuils


def find_matches(threshold):
    matches = []
    for i, scrum_elem in enumerate(scrum_elements):
        for j, prince2_elem in enumerate(prince2_elements):
            if similarity_matrix[i, j] >= threshold:
                matches.append(
                    (scrum_elem, prince2_elem, similarity_matrix[i, j]))
    return matches


# Seuil choisi
threshold = 0.3
matches = find_matches(threshold)

print(f"Correspondances trouvées avec le seuil {threshold}:")
for match in matches:
    print(f"{match[0]} ↔ {match[1]} : Similarité = {match[2]}")

# Étape 7: Calculer les métriques de qualité


def calculate_metrics(matches):
    tp = sum(1 for m in matches if real_matches.get(m[0]) == m[1])
    fp = len(matches) - tp
    fn = len(real_matches) - tp
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f_measure = 2 * (precision * recall) / (precision +
                                            recall) if (precision + recall) > 0 else 0
    return precision, recall, f_measure


precision, recall, f_measure = calculate_metrics(matches)

print("Métriques de qualité:")
print(f"Précision: {precision}")
print(f"Rappel: {recall}")
print(f"Mesure F: {f_measure}")
