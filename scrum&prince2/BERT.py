import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from transformers import BertTokenizer, BertModel
import torch

# Step 1: Extract elements from the metamodels
scrum_elements = [
    "ProductBacklog",
    "Sprint",
    "SprintBacklog",
    "ScrumTeam",
    "Increment",
    "DailyScrum"
]

prince2_elements = [
    "BusinessCase",
    "ProjectBoard",
    "ProjectPlan",
    "StagePlan",
    "WorkPackage",
    "EndStageReport"
]

# Step 2: Generate BERT embeddings


def get_bert_embeddings(elements):
    # Load pre-trained BERT model and tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')

    embeddings = []
    for element in elements:
        # Encode the element and convert to tensor
        input_ids = tokenizer.encode(element, return_tensors='pt')
        with torch.no_grad():
            outputs = model(input_ids)
            # Use the output from the last hidden layer
            last_hidden_states = outputs.last_hidden_state
            # Take the mean of the last hidden states to get a single vector representation
            embeddings.append(last_hidden_states.mean(dim=1).squeeze().numpy())
    return np.array(embeddings)


# Get embeddings for both sets of elements
scrum_embeddings = get_bert_embeddings(scrum_elements)
prince2_embeddings = get_bert_embeddings(prince2_elements)

# Step 3: Calculate cosine similarity
similarity_matrix = cosine_similarity(scrum_embeddings, prince2_embeddings)

# Convert the similarity matrix to a DataFrame for better visualization
similarity_df = pd.DataFrame(similarity_matrix,
                             index=scrum_elements,
                             columns=prince2_elements)

# Step 4: Present the similarity matrix
print("Similarity Matrix (Scrum vs PRINCE2):")
print(similarity_df)

# Optional: Define true matches for metrics (for example)
true_matches = np.array([
    [1, 0, 0, 0, 0, 0],  # ProductBacklog ↔ BusinessCase
    [0, 0, 1, 0, 0, 0],  # Sprint ↔ ProjectPlan
    [0, 0, 0, 0, 1, 0],  # SprintBacklog ↔ WorkPackage
    [0, 1, 0, 0, 0, 0],  # ScrumTeam ↔ ProjectBoard
    [0, 0, 0, 0, 0, 1],  # Increment ↔ EndStageReport
    [0, 0, 0, 0, 0, 0],  # DailyScrum ↔ Aucun équivalent
])

# Function to calculate metrics


def calculate_metrics(similarity_matrix, threshold, true_matches):
    predicted_matches = similarity_matrix >= threshold

    # Recall
    true_positive = np.sum(np.logical_and(predicted_matches, true_matches))
    recall = true_positive / np.sum(true_matches)

    # Precision
    precision = true_positive / np.sum(predicted_matches)

    # F1-score
    if precision + recall == 0:
        f1_score = 0
    else:
        f1_score = 2 * (precision * recall) / (precision + recall)

    return recall, precision, f1_score


# Testing different thresholds
thresholds = [0.3, 0.5, 0.8]
metrics_results = {}

for threshold in thresholds:
    recall, precision, f1_score = calculate_metrics(
        similarity_matrix, threshold, true_matches)
    metrics_results[threshold] = (recall, precision, f1_score)

# Output metrics results
print("\nMetrics for different thresholds:")
for threshold, metrics in metrics_results.items():
    print(f"Threshold: {threshold}, Recall: {metrics[0]:.2f}, Precision: {
          metrics[1]:.2f}, F1-score: {metrics[2]:.2f}")

# Maximum and mean similarity
max_similarity = np.max(similarity_matrix)
mean_similarity = np.mean(similarity_matrix)

print(f"\nMaximum similarity: {max_similarity:.2f}")
print(f"Mean similarity: {mean_similarity:.2f}")
