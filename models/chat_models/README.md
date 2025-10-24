# Chat Models

This folder contains implementations of various chat model providers in LangChain.

## 📖 Overview

Chat Models are AI language models designed for conversational interactions. This folder demonstrates how to integrate and use different LLM providers (OpenAI, Google Gemini, Anthropic Claude, and HuggingFace) with LangChain.

## 🎯 Supported Providers

### 1. **OpenAI** (`openai.py`)
- **Model**: GPT-3.5-turbo
- **Provider**: OpenAI
- **Key Features**:
  - Temperature control (0.7 for creative responses)
  - Simple invoke method for queries
  - Environment-based API key management

### 2. **Google Gemini** (`geminichatmodel.py`, `geminichat2.py`)
- **Model**: Gemini-2.5-flash
- **Provider**: Google Generative AI
- **Key Features**:
  - Fast response times
  - Temperature configuration
  - Content extraction from responses

### 3. **Anthropic Claude** (`anthropicchatmodel.py`)
- **Model**: Claude-3-7-sonnet
- **Provider**: Anthropic
- **Key Features**:
  - Advanced reasoning capabilities
  - Unified model initialization with `init_chat_model`

### 4. **HuggingFace** (`huggingfaceapi.py`)
- **Model**: DeepSeek-R1
- **Provider**: HuggingFace Inference API
- **Key Features**:
  - Text generation task
  - Open-source model access
  - ChatHuggingFace wrapper for conversational interface

## 🔑 Key Concepts

### Model Invocation
All models use the `.invoke()` method to send queries:
```python
response = model.invoke("What is the capital of India")
print(response.content)
```

### Temperature Control
- **Temperature**: Controls randomness in responses (0.0 to 1.0)
  - Lower (0.0-0.3): More deterministic, factual
  - Medium (0.4-0.7): Balanced creativity
  - Higher (0.8-1.0): More creative, varied

### Environment Variables
All implementations use `.env` file for API key management:
- `OPENAI_API_KEY`
- `GOOGLE_API_KEY`
- `ANTHROPIC_API_KEY`
- `HUGGINGFACEHUB_API_TOKEN`

## 📁 Files Description

| File | Model | Provider | Use Case |
|------|-------|----------|----------|
| `openai.py` | GPT-3.5-turbo | OpenAI | General-purpose chat |
| `geminichatmodel.py` | Gemini-2.5-flash | Google | Fast responses |
| `geminichat2.py` | Gemini-2.5-flash | Google | Alternative implementation |
| `anthropicchatmodel.py` | Claude-3-7-sonnet | Anthropic | Advanced reasoning |
| `huggingfaceapi.py` | DeepSeek-R1 | HuggingFace | Open-source models |

## 💡 Usage Pattern

```python
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)
response = model.invoke("Your question here")
print(response.content)
```

## 🛠️ Dependencies

- `langchain_openai`
- `langchain_google_genai`
- `langchain_huggingface`
- `langchain` (for init_chat_model)
- `python-dotenv`

## 🎓 Learning Outcomes

After reviewing these examples, you should understand:
- How to integrate multiple LLM providers in LangChain
- Differences between various AI models
- API key management and security best practices
- Model parameter tuning (temperature, etc.)
- Unified interface for different chat models
