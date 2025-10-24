# Chat Prompt Template

This folder contains examples and implementations of Chat Prompt Templates in LangChain.

## 📖 Overview

Chat Prompt Templates are structured ways to create conversations with AI models by defining system messages, user messages, and managing chat history. They help create consistent and well-formatted prompts for chat-based language models.

## 🎯 Concepts Covered

### 1. **ChatPromptTemplate**
- Creating structured prompts with system and human messages
- Using variables in prompts with `{variable_name}` syntax
- Formatting messages with dynamic content

### 2. **MessagesPlaceholder**
- Managing chat history in conversations
- Inserting previous conversation context into prompts
- Building context-aware chatbots

## 📁 Files

### `chatproompt.py`
- **Purpose**: Basic ChatPromptTemplate example
- **Key Features**:
  - System message defining assistant role
  - Human message with dynamic topic variable
  - Message formatting with `.format_messages()`
- **Use Case**: Simple question-answering with topic-based queries

### `messageplaceholder.py`
- **Purpose**: Chat history management
- **Key Features**:
  - MessagesPlaceholder for dynamic chat history
  - Loading chat history from external file (`chat_history.txt`)
  - Building customer support agent with context
- **Use Case**: Customer support chatbot with conversation memory

### `chat_history.txt`
- **Purpose**: Sample chat history data
- **Use Case**: Demonstrates how to load and use previous conversation context

## 🔑 Key Concepts

1. **System Message**: Defines the AI's role and behavior
2. **Human Message**: Represents user input
3. **Message Variables**: Dynamic placeholders for runtime values
4. **Chat History**: Maintains conversation context across multiple turns

## 💡 Usage Example

```python
from langchain_core.prompts import ChatPromptTemplate

# Basic template
chat_prompt = ChatPromptTemplate([
    ("system", "you are a helpful assistant explain about {topic} in simple terms"),
    ("human", "explain in simple term what is {topic}?")
])

# Format with specific topic
final_prompt = chat_prompt.format_messages(topic="AI")
```

## 🛠️ Dependencies

- `langchain_core`
- `python-dotenv`

## 🎓 Learning Outcomes

After reviewing these examples, you should understand:
- How to structure chat prompts for better AI responses
- Managing conversation history in chatbots
- Using variables to create reusable prompt templates
- Building context-aware conversational AI applications
