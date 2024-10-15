import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Step 1: Define the elements and their descriptions for TF-IDF vectorization

# Elements with their attributes (acting as descriptions for TF-IDF vectorization)
scrum_elements_with_descriptions = {
    "ProductBacklog": "ID: String, Name: String, Description: String",
    "Sprint": "ID: String, Name: String, Goal: String, StartDate: Date, EndDate: Date",
    "ScrumTeam": "ID: String, Name: String, Members: List",
    "SprintBacklog": "ID: String, Name: String, Tasks: List",
    "Increment": "ID: String, Description: String, Version: String",
    "DailyScrum": "ID: String, Date: Date, Notes: String",
}

prince2_elements_with_descriptions = {
    "BusinessCase": "ID: String, Title: String, Description: String",
    "ProjectPlan": "ID: String, Name: String, StartDate: Date, EndDate: Date",
    "ProjectBoard": "ID: String, Name: String, Members: List",
    "StagePlan": "ID: String, Name: String, StageObjective: String",
    "WorkPackage": "ID: String, Name: String, Tasks: List",
    "EndStageReport": "ID: String, Name: String, Summary: String",
}

# Combine all elements and their descriptions for vectorization
combined_elements = {
    **scrum_elements_with_descriptions,
    **prince2_elements_with_descriptions
}

# Prepare data for TF-IDF Vectorization
element_names = list(combined_elements.keys())
descriptions = list(combined_elements.values())

# Create TF-IDF Vectorizer
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(descriptions)

# Calculate cosine similarity between all elements
similarity_matrix = cosine_similarity(X)

# Create a DataFrame for better visualization of the similarity matrix
similarity_df = pd.DataFrame(
    similarity_matrix,
    index=element_names,
    columns=element_names
)

# Define real correspondences as ground truth
real_correspondences = {
    "ProductBacklog": "BusinessCase",
    "Sprint": "ProjectPlan",
    "ScrumTeam": "ProjectBoard",
    "SprintBacklog": "WorkPackage",
    "Increment": "EndStageReport",
    "DailyScrum": "StagePlan"
}

# Function to calculate quality metrics


def calculate_quality_metrics(similarity_df, threshold, real_correspondences):
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


# Step 4: Evaluate metrics at various thresholds
thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
results = []

for threshold in thresholds:
    precision, recall, f1_score = calculate_quality_metrics(
        similarity_df, threshold, real_correspondences)
    results.append({
        "Threshold": threshold,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1_score
    })

# Create a DataFrame for results
results_df = pd.DataFrame(results)

# Find the best threshold based on the highest F1 score
best_threshold = results_df.loc[results_df["F1 Score"].idxmax()]

# Step 5: Print results

# Print the similarity matrix
print("Similarity Matrix:")
print(similarity_df)
print("\n")

# Print the results of metrics for each threshold
print("Quality Metrics for Each Threshold:")
print(results_df)
print("\n")

# Print the best threshold and its metrics
print("Best Threshold based on F1 Score:")
print(best_threshold)
