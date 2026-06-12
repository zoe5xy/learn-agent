import numpy as np

# Assume we have learned simplified 2D word vectors
embeddings = {
    "king": np.array([0.9, 0.8]),
    "queen": np.array([0.9, 0.2]),
    "man": np.array([0.7, 0.9]),
    "woman": np.array([0.7, 0.3])
}

def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm_product = np.linalg.norm(vec1) * np.linalg.norm(vec2)
    return dot_product / norm_product

# king - man + woman
result_vec = embeddings["king"] - embeddings["man"] + embeddings["woman"]

# Calculate similarity between result vector and "queen"
sim = cosine_similarity(result_vec, embeddings["queen"])

print(f"Result vector of king - man + woman: {result_vec}")
print(f"Similarity of this result with 'queen': {sim:.4f}")

