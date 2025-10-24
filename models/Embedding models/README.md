# Embedding Models

This folder contains implementations and applications of various embedding models for text vectorization and similarity analysis.

## 📖 Overview

Embedding models convert text into numerical vectors (embeddings) that capture semantic meaning. These vectors enable semantic search, similarity comparison, and clustering of text documents.

## 🎯 Concepts Covered

### 1. **Text Embeddings**
- Converting text to high-dimensional vectors
- Semantic representation of documents
- Vector operations for text analysis

### 2. **Similarity Measurement**
- Cosine similarity calculation
- Comparing semantic similarity between texts
- Percentage-based similarity scoring

### 3. **Provider Implementations**
- OpenAI embeddings
- Google Gemini embeddings
- HuggingFace embeddings

## 📁 Files Description

### `openai_embedding_documents.py`
- **Provider**: OpenAI
- **Model**: text-embedding-3-large
- **Features**:
  - Configurable vector dimensions (32 in example)
  - Batch document embedding
  - Multiple document processing

### `geminiembedding_models.py`
- **Provider**: Google Generative AI
- **Model**: gemini-embedding-001
- **Features**:
  - Task-specific embeddings (RETRIEVAL_DOCUMENT)
  - Single query embedding
  - High-quality semantic vectors

### `huggingface_embedding.py`
- **Provider**: HuggingFace
- **Model**: sentence-transformers/all-MiniLM-L6-v2
- **Features**:
  - Open-source embedding model
  - Lightweight and fast
  - Multi-document embedding

### `query_similarity.py`
- **Purpose**: Cricket player text similarity analysis
- **Key Features**:
  - Compares two cricket player biographies
  - Cosine similarity calculation using NumPy
  - Percentage-based similarity output
- **Use Case**: Demonstrating semantic similarity between related texts

### `text_similarity_projects.py`
- **Purpose**: Interactive Streamlit web application
- **Key Features**:
  - User-friendly text comparison interface
  - Real-time similarity calculation
  - Visual percentage display
  - Two text input areas for comparison
- **Technology**: Streamlit + Gemini Embeddings
- **Use Case**: General-purpose text similarity tool

## 🔑 Key Mathematical Concepts

### Cosine Similarity Formula
```
similarity = (A · B) / (||A|| × ||B||)
```

Where:
- A, B are embedding vectors
- · represents dot product
- ||A|| represents vector magnitude (L2 norm)

### Similarity Interpretation
- **0.9-1.0**: Highly similar (90-100%)
- **0.7-0.9**: Moderately similar (70-90%)
- **0.5-0.7**: Somewhat similar (50-70%)
- **0.0-0.5**: Low similarity (0-50%)

## 💡 Usage Examples

### Basic Embedding
```python
from langchain_openai import OpenAIEmbeddings

embedding = OpenAIEmbeddings(model='text-embedding-3-large')
documents = ["Text 1", "Text 2"]
vectors = embedding.embed_documents(documents)
```

### Similarity Calculation
```python
import numpy as np

def cosine_similarity(a, b):
    a = np.array(a)
    b = np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

similarity_score = cosine_similarity(vector1, vector2)
```

## 🛠️ Dependencies

- `langchain_openai`
- `langchain_google_genai`
- `langchain_huggingface`
- `numpy`
- `streamlit` (for web app)
- `python-dotenv`

## 🎯 Use Cases

1. **Semantic Search**: Find similar documents in a database
2. **Duplicate Detection**: Identify similar or duplicate content
3. **Text Clustering**: Group related documents together
4. **Question-Answer Matching**: Match questions with relevant answers
5. **Content Recommendation**: Suggest similar articles or products

## 🎓 Learning Outcomes

After reviewing these examples, you should understand:
- How embedding models work
- Different embedding providers and their characteristics
- Calculating and interpreting similarity scores
- Building text similarity applications
- Choosing appropriate embedding dimensions
- Practical applications of semantic similarity
