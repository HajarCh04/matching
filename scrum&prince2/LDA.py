import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import numpy as np
from sklearn.metrics import precision_recall_fscore_support

# Elements from Scrum and PRINCE2 metamodels
scrum_elements = [
    "ProductBacklog: ID, Name, Description",
    "Sprint: ID, Name, Goal, StartDate, EndDate",
    "ScrumTeam: ID, Name, Members",
    "SprintBacklog: ID, Name, Tasks",
    "Increment: ID, Name, Description, Version",
    "DailyScrum: ID, Name, Date, Notes"
]

prince2_elements = [
    "BusinessCase: ID, Title, Description",
    "ProjectBoard: ID, Name, Members",
    "ProjectPlan: ID, Name, StartDate, EndDate",
    "StagePlan: ID, Name, StageObjective",
    "WorkPackage: ID, Name, Tasks",
    "EndStageReport: ID, Name, Summary"
]

# Combine elements for LDA
documents = scrum_elements + prince2_elements

# Vectorize the documents
vectorizer = CountVectorizer()
dtm = vectorizer.fit_transform(documents)

# Apply LDA
lda = LatentDirichletAllocation(n_components=5, random_state=42)
lda_matrix = lda.fit_transform(dtm)

# Compute similarity (dot product of topic distributions)
similarity_matrix = np.dot(
    lda_matrix[:len(scrum_elements)], lda_matrix[len(scrum_elements):].T)

# Convert to DataFrame for easier manipulation
similarity_df = pd.DataFrame(
    similarity_matrix, index=scrum_elements, columns=prince2_elements)

# Define real matches (ground truth)
real_matches = np.zeros(similarity_matrix.shape)
real_indices = [
    (0, 0),  # ProductBacklog ↔ BusinessCase
    (1, 2),  # Sprint ↔ ProjectPlan
    (2, 1),  # ScrumTeam ↔ ProjectBoard
    (3, 4),  # SprintBacklog ↔ WorkPackage
    (4, 5),  # Increment ↔ EndStageReport
    (5, 3)   # DailyScrum ↔ StagePlan
]
for x, y in real_indices:
    real_matches[x, y] = 1

# Flatten the ground truth array for metric calculation
true_labels = real_matches.flatten()

# Evaluate metrics over a range of thresholds
thresholds = np.arange(0.1, 1.0, 0.1)
best_threshold = 0
best_f1 = 0

for threshold in thresholds:
    # Convert similarity scores to binary matches based on the threshold
    pred_labels = (similarity_matrix >= threshold).astype(int).flatten()

    # Calculate metrics
    precision, recall, f1, _ = precision_recall_fscore_support(
        true_labels, pred_labels, average='binary')

    # Print metrics for each threshold
    print(f"Threshold: {
          threshold:.1f} - Precision: {precision:.2f}, Recall: {recall:.2f}, F1-Score: {f1:.2f}")

    # Determine the best threshold based on F1-score
    if f1 > best_f1:
        best_f1 = f1
        best_threshold = threshold

print(f"\nBest Threshold: {best_threshold:.1f} - Best F1-Score: {best_f1:.2f}")

# Calculate maximum and average similarity scores
max_similarity = np.max(similarity_matrix)
mean_similarity = np.mean(similarity_matrix)

print(f"Maximum Similarity Score: {max_similarity:.2f}")
print(f"Average Similarity Score: {mean_similarity:.2f}")

# Display the similarity matrix
print("Similarity Matrix:")
print(similarity_df)
