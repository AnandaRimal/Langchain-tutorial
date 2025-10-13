from langchain_huggingface import HuggingFaceEmbeddings

embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
documents = [
    "Kathamndu is capital of nepal",
    "New Delhi is capital of India"
    "Madrid is capital of Spain"
]
vectors = embedding.embed_documents(documents)
print(str(vectors))