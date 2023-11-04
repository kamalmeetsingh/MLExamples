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

# Print the embeddings for each sentence
for sentence, embedding in zip(sentences, sentence_embeddings):
    print(f"Sentence: {sentence}")
    print(f"Embedding: {embedding}")
    print("\n")
