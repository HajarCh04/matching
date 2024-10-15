import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score

# Elements of each metamodel
scrum_elements = ["ProductBacklog", "Sprint", "ScrumTeam",
                  "SprintBacklog", "Increment", "DailyScrum"]
prince2_elements = ["BusinessCase", "ProjectPlan",
                    "ProjectBoard", "WorkPackage", "StagePlan", "EndStageReport"]

# Generate random similarities for illustration
np.random.seed(0)
similarities = np.random.rand(len(scrum_elements), len(prince2_elements))

# Define true matches (ground truth)
true_matches = {
    "ProductBacklog": "BusinessCase",
    "Sprint": "ProjectPlan",
    "ScrumTeam": "ProjectBoard",
    "SprintBacklog": "WorkPackage",
    "Increment": "StagePlan",
    "DailyScrum": "EndStageReport"
}

# Create a DataFrame for similarities
similarity_df = pd.DataFrame(
    similarities, index=scrum_elements, columns=prince2_elements)

# Function to calculate quality metrics


def calculate_metrics(threshold):
    predictions = []
    truths = []

    for scrum in scrum_elements:
        for prince2 in prince2_elements:
            similarity = similarity_df.loc[scrum, prince2]
            match = similarity >= threshold
            true_match = true_matches.get(scrum) == prince2

            predictions.append(match)
            truths.append(true_match)

    precision = precision_score(truths, predictions)
    recall = recall_score(truths, predictions)
    f1 = f1_score(truths, predictions)

    return precision, recall, f1


# Iterate through thresholds and find the best one
thresholds = np.linspace(0, 1, 101)
best_threshold = 0
best_f1 = 0
metrics_dict = {}

for threshold in thresholds:
    precision, recall, f1 = calculate_metrics(threshold)
    metrics_dict[threshold] = (precision, recall, f1)
    if f1 > best_f1:
        best_f1 = f1
        best_threshold = threshold

best_metrics = metrics_dict[best_threshold]

# Display the results
print(f"Best Threshold: {best_threshold}")
print(f"Precision: {best_metrics[0]}")
print(f"Recall: {best_metrics[1]}")
print(f"F1 Score: {best_metrics[2]}")
print("\nSimilarity Table:")
print(similarity_df)

# Format results into a readable table
metrics_table = pd.DataFrame(
    metrics_dict, index=["Precision", "Recall", "F1"]).T
