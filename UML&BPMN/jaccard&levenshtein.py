from difflib import SequenceMatcher
import numpy as np
from sklearn.metrics import jaccard_score
from Levenshtein import distance as levenshtein_distance

# Étape 1 : Extraction des éléments des métamodèles
uml_elements = [
    "ControlFlow", "DecisionNode", "MergeNode", "ForkNode",
    "JoinNode", "InitialNode", "FinalNode", "Action",
    "Activity", "Partition", "ObjectNode"
]

bpmn_elements = [
    "SequenceFlow", "ExclusiveGateway", "ParallelGateway",
    "StartEvent", "EndEvent", "Task", "Process", "Lane",
    "DataObject"
]

# Étape 2 : Correspondances réelles (définies manuellement)
real_matches = [
    ("ControlFlow", "SequenceFlow"),
    ("DecisionNode", "ExclusiveGateway"),
    ("MergeNode", "ExclusiveGateway"),
    ("ForkNode", "ParallelGateway"),
    ("JoinNode", "ParallelGateway"),
    ("InitialNode", "StartEvent"),
    ("FinalNode", "EndEvent"),
    ("Action", "Task"),
    ("Activity", "Process"),
    ("Partition", "Lane"),
    ("ObjectNode", "DataObject")
]

# Étape 3 : Calcul des similarités Jaccard et Levenshtein


def jaccard_similarity(a, b):
    a_set = set(a)
    b_set = set(b)
    intersection = len(a_set.intersection(b_set))
    union = len(a_set.union(b_set))
    return intersection / union


def levenshtein_similarity(a, b):
    max_len = max(len(a), len(b))
    return 1 - (levenshtein_distance(a, b) / max_len)


# Matrices de similarité
jaccard_similarity_matrix = np.zeros((len(uml_elements), len(bpmn_elements)))
levenshtein_similarity_matrix = np.zeros(
    (len(uml_elements), len(bpmn_elements)))

for i, uml_elem in enumerate(uml_elements):
    for j, bpmn_elem in enumerate(bpmn_elements):
        jaccard_similarity_matrix[i, j] = jaccard_similarity(
            uml_elem, bpmn_elem)
        levenshtein_similarity_matrix[i, j] = levenshtein_similarity(
            uml_elem, bpmn_elem)

# Étape 4 : Calcul des métriques de qualité
thresholds = np.arange(0.1, 0.6, 0.1)


def calculate_metrics(similarity_matrix, real_matches, thresholds):
    precision_scores = []
    recall_scores = []
    f1_scores = []

    for threshold in thresholds:
        true_positives = 0
        false_positives = 0
        false_negatives = 0

        for i, uml_elem in enumerate(uml_elements):
            for j, bpmn_elem in enumerate(bpmn_elements):
                if similarity_matrix[i, j] >= threshold:
                    if (uml_elem, bpmn_elem) in real_matches:
                        true_positives += 1
                    else:
                        false_positives += 1

        # Les faux négatifs sont les correspondances réelles non trouvées
        for (uml_elem, bpmn_elem) in real_matches:
            if similarity_matrix[uml_elements.index(uml_elem), bpmn_elements.index(bpmn_elem)] < threshold:
                false_negatives += 1

        # Calcul des métriques
        precision = true_positives / \
            (true_positives + false_positives) if (true_positives +
                                                   false_positives) > 0 else 0
        recall = true_positives / \
            (true_positives + false_negatives) if (true_positives +
                                                   false_negatives) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision +
                                               recall) if (precision + recall) > 0 else 0

        precision_scores.append(precision)
        recall_scores.append(recall)
        f1_scores.append(f1_score)

    return precision_scores, recall_scores, f1_scores


# Calcul des métriques pour Jaccard
precision_jaccard, recall_jaccard, f1_jaccard = calculate_metrics(
    jaccard_similarity_matrix, real_matches, thresholds)

# Calcul des métriques pour Levenshtein
precision_levenshtein, recall_levenshtein, f1_levenshtein = calculate_metrics(
    levenshtein_similarity_matrix, real_matches, thresholds)

# Affichage des résultats
for i, threshold in enumerate(thresholds):
    print(f"Threshold: {threshold}")
    print(f"Jaccard - Precision: {precision_jaccard[i]:.2%}, Recall: {
          recall_jaccard[i]:.2%}, F1-Score: {f1_jaccard[i]:.2%}")
    print(f"Levenshtein - Precision: {precision_levenshtein[i]:.2%}, Recall: {
          recall_levenshtein[i]:.2%}, F1-Score: {f1_levenshtein[i]:.2%}")
    print()


# Éléments extraits des métamodèles UML et BPMN
uml_elements = ["Action", "ControlFlow", "DecisionNode", "MergeNode", "ForkNode",
                "JoinNode", "InitialNode", "FinalNode", "Activity", "Partition", "ObjectNode"]
bpmn_elements = ["Task", "SequenceFlow", "ExclusiveGateway",
                 "ParallelGateway", "StartEvent", "EndEvent", "Process", "Lane", "DataObject"]

# Correspondances réelles
real_matches = [
    ("Action", "Task"),
    ("ControlFlow", "SequenceFlow"),
    ("DecisionNode", "ExclusiveGateway"),
    ("MergeNode", "ExclusiveGateway"),
    ("ForkNode", "ParallelGateway"),
    ("JoinNode", "ParallelGateway"),
    ("InitialNode", "StartEvent"),
    ("FinalNode", "EndEvent"),
    ("Activity", "Process"),
    ("Partition", "Lane"),
    ("ObjectNode", "DataObject")
]

# Calcul de la similarité de Jaccard (trivial dans ce cas car les attributs sont similaires)
# et de la distance de Levenshtein entre chaque paire d'éléments
jaccard_similarity = np.zeros((len(uml_elements), len(bpmn_elements)))
levenshtein_distance = np.zeros((len(uml_elements), len(bpmn_elements)))

for i, uml_elem in enumerate(uml_elements):
    for j, bpmn_elem in enumerate(bpmn_elements):
        # Similarité de Jaccard
        jaccard_similarity[i, j] = len(set(uml_elem).intersection(
            set(bpmn_elem))) / len(set(uml_elem).union(set(bpmn_elem)))

        # Distance de Levenshtein
        levenshtein_distance[i, j] = SequenceMatcher(
            None, uml_elem, bpmn_elem).ratio()

# Convertir la distance de Levenshtein en similarité en divisant par la longueur max des éléments
levenshtein_similarity = levenshtein_distance

# Création du tableau de similarité pour Jaccard et Levenshtein
jaccard_similarity, levenshtein_similarity
print(jaccard_similarity)
print(levenshtein_similarity)
