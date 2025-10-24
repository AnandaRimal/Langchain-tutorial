# Models

This folder contains implementations and examples of various Language Models (LLMs) and Embedding Models in LangChain.

## 📖 Overview

This directory demonstrates how to work with two fundamental types of AI models:
1. **Chat Models**: Conversational AI models for generating text responses
2. **Embedding Models**: Models that convert text into numerical vectors for semantic analysis

## 📂 Folder Structure

### `chat_models/`
Contains implementations of various chat model providers:
- **OpenAI** (GPT-3.5-turbo)
- **Google Gemini** (Gemini-2.5-flash)
- **Anthropic Claude** (Claude-3-7-sonnet)
- **HuggingFace** (DeepSeek-R1)

### `Embedding models/`
Contains embedding model implementations and similarity analysis tools:
- **OpenAI Embeddings** (text-embedding-3-large)
- **Google Gemini Embeddings** (gemini-embedding-001)
- **HuggingFace Embeddings** (sentence-transformers)
- **Similarity Projects** (Text comparison applications)

## 🎯 Key Concepts

### Chat Models
- Generate human-like text responses
- Support conversational context
- Configurable creativity (temperature)
- Multi-provider support

### Embedding Models
- Convert text to numerical vectors
- Enable semantic similarity comparison
- Support semantic search
- Dimensionality reduction options

## 🔑 Core Features

1. **Multi-Provider Support**: Work with different AI providers seamlessly
2. **Unified Interface**: Consistent API across different models
3. **Environment-based Configuration**: Secure API key management
4. **Practical Examples**: Real-world use cases and implementations

## 💡 Common Usage Patterns

### Chat Model Pattern
```python
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()
model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)
response = model.invoke("Your question")
print(response.content)
```

### Embedding Model Pattern
```python
from langchain_openai import OpenAIEmbeddings

embedding = OpenAIEmbeddings(model='text-embedding-3-large')
vectors = embedding.embed_documents(["Text 1", "Text 2"])
```

## 🛠️ Dependencies

- `langchain_openai`
- `langchain_google_genai`
- `langchain_huggingface`
- `langchain`
- `numpy`
- `streamlit`
- `python-dotenv`

## 🎓 Learning Path

1. Start with **chat_models/** to understand basic LLM integration
2. Explore **Embedding models/** for semantic analysis
3. Compare different providers for your use case
4. Build applications combining both chat and embedding models

## 📚 Further Reading

- [LangChain Documentation](https://python.langchain.com/)
- [OpenAI API Documentation](https://platform.openai.com/docs)
- [Google AI Documentation](https://ai.google.dev/)
- [HuggingFace Documentation](https://huggingface.co/docs)
