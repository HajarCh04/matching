import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Élément des métamodèles UML et BPMN
uml_elements = [
    "FinalNode", "ControlFlow", "DecisionNode", "MergeNode", "ForkNode",
    "JoinNode", "InitialNode", "Action", "Activity", "Partition", "ObjectNode"
]

bpmn_elements = [
    "EndEvent", "SequenceFlow", "ExclusiveGateway", "ParallelGateway",
    "StartEvent", "Task", "Process", "Lane", "DataObject"
]

# Correspondances réelles
real_matches = {
    "Action": "Task",
    "Activity": "Process",
    "ControlFlow": "SequenceFlow",
    "FinalNode": "EndEvent",
    "InitialNode": "StartEvent",
    "DecisionNode": "ExclusiveGateway",
    "MergeNode": "ParallelGateway",
    "ObjectNode": "DataObject",
    "Partition": "Lane"
}

# Préparation des données
all_elements = uml_elements + bpmn_elements

# Calcul de la similarité de cosinus
vectorizer = TfidfVectorizer().fit_transform(all_elements)
vectors = vectorizer.toarray()
cosine_matrix = cosine_similarity(
    vectors[:len(uml_elements)], vectors[len(uml_elements):])

# Création d'une DataFrame pour stocker les similarités
similarity_df = pd.DataFrame(
    cosine_matrix, index=uml_elements, columns=bpmn_elements)

# Définir un seuil de similarité
threshold = 0.5

# Trouver les correspondances basées sur le seuil
matches = []
for uml_element in uml_elements:
    for bpmn_element in bpmn_elements:
        similarity = similarity_df.loc[uml_element, bpmn_element]
        if similarity >= threshold:
            matches.append((uml_element, bpmn_element, similarity))

# Calcul des métriques de qualité
true_positives = sum(
    1 for match in matches if real_matches.get(match[0]) == match[1])
false_positives = len(matches) - true_positives
false_negatives = len(real_matches) - true_positives

# Vérification pour éviter la division par zéro
if true_positives + false_positives > 0:
    precision = true_positives / (true_positives + false_positives)
else:
    precision = 0.0

if true_positives + false_negatives > 0:
    recall = true_positives / (true_positives + false_negatives)
else:
    recall = 0.0

if precision + recall > 0:
    f_measure = 2 * (precision * recall) / (precision + recall)
else:
    f_measure = 0.0

# Affichage des résultats
print("Matrice de similarité:")
print(similarity_df)
print("\nCorrespondances trouvées (avec similarité ≥ seuil de {}):".format(threshold))
for match in matches:
    print(match)

print("\nMétriques de qualité:")
print("Précision: {:.2f}".format(precision))
print("Rappel: {:.2f}".format(recall))
print("F-mesure: {:.2f}".format(f_measure))
