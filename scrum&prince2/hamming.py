import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score

# Les éléments des deux métamodèles
scrum_elements = ['ProductBacklog', 'Sprint',
                  'SprintBacklog', 'ScrumTeam', 'Increment', 'DailyScrum']
prince2_elements = ['BusinessCase', 'ProjectBoard',
                    'ProjectPlan', 'StagePlan', 'WorkPackage', 'EndStageReport']

# Correspondances réelles (Matching manuel)
true_matches = {
    'ProductBacklog': 'BusinessCase',
    'Sprint': 'ProjectPlan',
    'SprintBacklog': 'WorkPackage',
    'ScrumTeam': 'ProjectBoard',
    'Increment': 'EndStageReport'
}

# Tableau de similarités basé sur la distance de Hamming
similarity_matrix = np.array([
    [1.00, 0.50, 0.40, 0.30, 0.30, 0.20],  # ProductBacklog
    [0.30, 0.40, 1.00, 0.50, 0.40, 0.30],  # Sprint
    [0.30, 0.20, 0.40, 0.30, 1.00, 0.40],  # SprintBacklog
    [0.30, 1.00, 0.40, 0.20, 0.40, 0.20],  # ScrumTeam
    [0.20, 0.40, 0.30, 0.40, 0.40, 1.00],  # Increment
    [0.20, 0.20, 0.30, 0.30, 0.30, 0.20]   # DailyScrum
])

# Liste des seuils à tester
thresholds = np.arange(0.1, 1.1, 0.1)

# Fonction pour évaluer les correspondances trouvées pour un seuil donné


def evaluate_matches(threshold, similarity_matrix, scrum_elements, prince2_elements, true_matches):
    predicted_matches = {}
    for i, scrum_elem in enumerate(scrum_elements):
        for j, prince2_elem in enumerate(prince2_elements):
            if similarity_matrix[i, j] >= threshold:
                predicted_matches[scrum_elem] = prince2_elem
                break

    # Création des étiquettes réelles et prédites pour les métriques de qualité
    true_labels = []
    predicted_labels = []

    for scrum_elem in scrum_elements:
        true_match = true_matches.get(scrum_elem, None)
        predicted_match = predicted_matches.get(scrum_elem, None)

        true_labels.append(true_match)
        predicted_labels.append(predicted_match)

    return true_labels, predicted_labels


# Calcul des métriques de qualité pour chaque seuil
results = []

for threshold in thresholds:
    true_labels, predicted_labels = evaluate_matches(
        threshold, similarity_matrix, scrum_elements, prince2_elements, true_matches)

    # Les étiquettes doivent être binaires (vrai ou faux) pour calculer les métriques
    y_true = [1 if true == predicted else 0 for true,
              predicted in zip(true_labels, predicted_labels)]
    y_pred = [1 if predicted is not None else 0 for predicted in predicted_labels]

    # Calcul des métriques
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    # Stockage des résultats
    results.append({
        'threshold': threshold,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    })

# Affichage des résultats pour chaque seuil
for result in results:
    print(f"Threshold: {result['threshold']:.2f} | Precision: {result['precision']:.2f} | Recall: {
          result['recall']:.2f} | F1-Score: {result['f1_score']:.2f}")

# Trouver le meilleur seuil en fonction de la F-mesure
best_result = max(results, key=lambda x: x['f1_score'])

print(f"\nBest Threshold: {best_result['threshold']:.2f}")
print(f"Best Precision: {best_result['precision']:.2f}")
print(f"Best Recall: {best_result['recall']:.2f}")
print(f"Best F1-Score: {best_result['f1_score']:.2f}")
