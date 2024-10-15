from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import pandas as pd

# Elements from Scrum Metamodel
scrum_elements = [
    "ProductBacklog", "Sprint", "ScrumTeam",
    "SprintBacklog", "Increment", "DailyScrum"
]

# Elements from PRINCE2 Metamodel
prince2_elements = [
    "BusinessCase", "ProjectPlan", "ProjectBoard",
    "StagePlan", "WorkPackage", "EndStageReport"
]

# Define the real correspondences as ground truth
real_correspondences = {
    "ProductBacklog": "BusinessCase",
    "Sprint": "ProjectPlan",
    "ScrumTeam": "ProjectBoard",
    "SprintBacklog": "WorkPackage",
    "Increment": "EndStageReport",
    "DailyScrum": "StagePlan"
}

# Create a combined list of elements for generating embeddings
combined_elements = scrum_elements + prince2_elements

# Generate TF-IDF Vectorizer to simulate word embeddings (for simplicity in this demo)
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(combined_elements)

# Calculate the cosine similarity between all elements
similarity_matrix = cosine_similarity(X)

# Create a DataFrame for better visualization of the similarity matrix
similarity_df = pd.DataFrame(
    similarity_matrix[:len(scrum_elements), len(scrum_elements):],
    index=scrum_elements,
    columns=prince2_elements
)

# Set different threshold values for analysis
thresholds = [0.3, 0.5, 0.7, 0.8]

# Function to calculate metrics: precision, recall, F1-score


def calculate_metrics(similarity_df, threshold, real_correspondences):
    true_positives = 0
    false_positives = 0
    false_negatives = 0

    # Check for real matches above the threshold
    for scrum_element, prince2_element in real_correspondences.items():
        if similarity_df.loc[scrum_element, prince2_element] >= threshold:
            true_positives += 1
        else:
            false_negatives += 1

    # Check for incorrect matches (false positives)
    for scrum_element in similarity_df.index:
        for prince2_element in similarity_df.columns:
            if similarity_df.loc[scrum_element, prince2_element] >= threshold and (scrum_element, prince2_element) not in real_correspondences.items():
                false_positives += 1

    # Calculate precision, recall, F1-score
    precision = true_positives / \
        (true_positives + false_positives) if (true_positives +
                                               false_positives) > 0 else 0
    recall = true_positives / \
        (true_positives + false_negatives) if (true_positives +
                                               false_negatives) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision +
                                           recall) if (precision + recall) > 0 else 0

    return precision, recall, f1_score


# Calculate metrics for each threshold
metrics = {threshold: calculate_metrics(
    similarity_df, threshold, real_correspondences) for threshold in thresholds}

# Find the best threshold based on the maximum F1-score
best_threshold = max(metrics, key=lambda t: metrics[t][2])

# Calculate the average similarity score across all elements
average_similarity = np.mean(similarity_df.values)

# Print the similarity matrix, metrics, best threshold, and average similarity
print("Similarity Matrix:")
print(similarity_df)

print("\nMetrics for each threshold:")
for threshold, (precision, recall, f1_score) in metrics.items():
    print(f"Threshold {threshold}: Precision={
          precision}, Recall={recall}, F1-score={f1_score}")

print(f"\nBest Threshold: {best_threshold}")
print(f"\nAverage Similarity: {average_similarity}")
