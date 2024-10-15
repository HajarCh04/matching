import numpy as np

# Define the elements of each metamodel
prince2_elements = ["BusinessCase", "ProjectBoard",
                    "ProjectPlan", "StagePlan", "WorkPackage", "EndStageReport"]
scrum_elements = ["ProductBacklog", "Sprint", "ScrumTeam",
                  "SprintBacklog", "Increment", "DailyScrum"]

# Define the actual correspondences for evaluation
actual_correspondences = {
    "BusinessCase": "ProductBacklog",
    "ProjectBoard": "ScrumTeam",
    "ProjectPlan": "Sprint",
    "StagePlan": "SprintBacklog",
    "WorkPackage": "SprintBacklog",
    "EndStageReport": "Increment"
}

# Initialize a similarity matrix with zeros
similarity_matrix = np.zeros((len(prince2_elements), len(scrum_elements)))

# Example similarity function based on predefined correspondences


def calculate_similarity(element1, element2):
    return 1.0 if actual_correspondences.get(element1) == element2 else 0.0


# Fill the similarity matrix
for i, prince2_element in enumerate(prince2_elements):
    for j, scrum_element in enumerate(scrum_elements):
        similarity_matrix[i, j] = calculate_similarity(
            prince2_element, scrum_element)

# Define a function to evaluate metrics


def evaluate_metrics(similarity_matrix, threshold):
    predictions = similarity_matrix >= threshold
    true_positives = 0
    false_positives = 0
    false_negatives = 0

    for i, prince2_element in enumerate(prince2_elements):
        for j, scrum_element in enumerate(scrum_elements):
            predicted = predictions[i, j]
            actual = actual_correspondences.get(
                prince2_element) == scrum_element

            if predicted and actual:
                true_positives += 1
            elif predicted and not actual:
                false_positives += 1
            elif not predicted and actual:
                false_negatives += 1

    precision = true_positives / \
        (true_positives + false_positives) if (true_positives +
                                               false_positives) > 0 else 0
    recall = true_positives / \
        (true_positives + false_negatives) if (true_positives +
                                               false_negatives) > 0 else 0
    f_measure = 2 * (precision * recall) / (precision +
                                            recall) if (precision + recall) > 0 else 0

    return precision, recall, f_measure


# Evaluate for different thresholds
thresholds = [0.3, 0.5, 0.7, 0.8]
results = {}

for threshold in thresholds:
    precision, recall, f_measure = evaluate_metrics(
        similarity_matrix, threshold)
    results[threshold] = {"Precision": precision,
                          "Recall": recall, "F-Measure": f_measure}

# Print the similarity matrix
print("Similarity Matrix:")
print(similarity_matrix)

# Print the results
print("\nEvaluation Metrics:")
for threshold, metrics in results.items():
    print(f"Threshold: {threshold}")
    print(f"  Precision: {metrics['Precision']:.2f}")
    print(f"  Recall: {metrics['Recall']:.2f}")
    print(f"  F-Measure: {metrics['F-Measure']:.2f}")

# Find the best threshold based on maximum F-Measure
best_threshold = max(results, key=lambda t: results[t]['F-Measure'])
best_metrics = results[best_threshold]

# Calculate max and average metrics
max_precision = max(metrics['Precision'] for metrics in results.values())
max_recall = max(metrics['Recall'] for metrics in results.values())
max_f_measure = max(metrics['F-Measure'] for metrics in results.values())

average_precision = np.mean([metrics['Precision']
                            for metrics in results.values()])
average_recall = np.mean([metrics['Recall'] for metrics in results.values()])
average_f_measure = np.mean([metrics['F-Measure']
                            for metrics in results.values()])

# Print the best metrics and overall statistics
print("\nBest Metrics:")
print(f"  Best Threshold: {best_threshold}")
print(f"  Precision: {best_metrics['Precision']:.2f}")
print(f"  Recall: {best_metrics['Recall']:.2f}")
print(f"  F-Measure: {best_metrics['F-Measure']:.2f}")

print("\nOverall Statistics:")
print(f"  Max Precision: {max_precision:.2f}")
print(f"  Max Recall: {max_recall:.2f}")
print(f"  Max F-Measure: {max_f_measure:.2f}")
print(f"  Average Precision: {average_precision:.2f}")
print(f"  Average Recall: {average_recall:.2f}")
print(f"  Average F-Measure: {average_f_measure:.2f}")
