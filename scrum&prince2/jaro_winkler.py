from jellyfish import jaro_winkler_similarity
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import numpy as np

# Elements of the Scrum and PRINCE2 metamodels
scrum_elements = [
    "ProductBacklog", "Sprint", "ScrumTeam",
    "SprintBacklog", "Increment", "DailyScrum"
]

prince2_elements = [
    "BusinessCase", "ProjectBoard", "ProjectPlan",
    "StagePlan", "WorkPackage", "EndStageReport"
]


def calculate_similarity_matrix(scrum_elements, prince2_elements):
    """Calculate the Jaro-Winkler similarity matrix between elements of two metamodels."""
    similarity_matrix = np.zeros((len(scrum_elements), len(prince2_elements)))
    for i, scrum_element in enumerate(scrum_elements):
        for j, prince2_element in enumerate(prince2_elements):
            similarity_score = jaro_winkler_similarity(
                scrum_element, prince2_element)
            similarity_matrix[i, j] = similarity_score
    return pd.DataFrame(similarity_matrix, index=scrum_elements, columns=prince2_elements)


# Calculate the similarity matrix
similarity_df = calculate_similarity_matrix(scrum_elements, prince2_elements)

# Display the similarity table
print("Jaro-Winkler Similarity Table:")
print(similarity_df)

# Define the ground truth matches (real correspondences)
real_matches = [
    (0, 0),  # ProductBacklog ↔ BusinessCase
    (1, 2),  # Sprint ↔ ProjectPlan
    (2, 1),  # ScrumTeam ↔ ProjectBoard
    (3, 4),  # SprintBacklog ↔ WorkPackage
    (4, 5),  # Increment ↔ EndStageReport
    (5, 3)   # DailyScrum ↔ StagePlan
]


def compute_metrics_for_threshold(similarity_df, real_matches, threshold):
    """Compute precision, recall, F1-score, and accuracy for a given threshold."""
    matches = (similarity_df >= threshold).astype(int)

    # Create the true labels vector
    num_elements = len(scrum_elements) * len(prince2_elements)
    true_labels = np.zeros(num_elements, dtype=int)
    for x, y in real_matches:
        true_labels[x * len(prince2_elements) + y] = 1

    # Flatten predictions
    pred_labels = matches.to_numpy().flatten()

    # Calculate metrics
    precision, recall, f1, _ = precision_recall_fscore_support(
        true_labels, pred_labels, average='binary', zero_division=0)
    accuracy = accuracy_score(true_labels, pred_labels)
    return precision, recall, f1, accuracy


# Test all thresholds from 0 to 1 with a step of 0.1
thresholds = np.arange(0.0, 1.1, 0.1)
results = []

for threshold in thresholds:
    precision, recall, f1, accuracy = compute_metrics_for_threshold(
        similarity_df, real_matches, threshold)
    results.append({
        'Threshold': threshold,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1,
        'Accuracy': accuracy
    })

# Convert results to DataFrame for visualization
results_df = pd.DataFrame(results)

# Display results for each threshold
print("\nResults for Each Threshold:")
print(results_df)

# Find the threshold that maximizes the F1-Score
best_threshold = results_df.loc[results_df['F1-Score'].idxmax()]
print("\nBest Threshold:")
print(best_threshold)
