from sentence_transformers import SentenceTransformer, util

# Load a pre-trained Sentence Transformer model
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# List of sentences you want to convert to embeddings
sentences = [
    "This is the first sentence.",
    "Here's the second sentence.",
    "And this is the third sentence."
]

# Compute embeddings for the sentences
sentence_embeddings = model.encode(sentences)

# Calculate cosine similarity between sentence pairs
similarities = util.pytorch_cos_sim(sentence_embeddings, sentence_embeddings)

# Print similarities between sentences
for i in range(len(sentences)):
    for j in range(len(sentences)):
        if i != j:
            similarity = similarities[i][j]
            print(f"Similarity between Sentence {i+1} and Sentence {j+1}: {similarity:.4f}")
