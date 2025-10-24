# Prompt Templates

This folder contains various implementations of Prompt Templates in LangChain with practical applications.

## 📖 Overview

Prompt Templates are reusable structures for creating dynamic prompts with variable inputs. They help maintain consistency, reduce code duplication, and enable easy modification of AI interactions.

## 🎯 Concepts Covered

### 1. **PromptTemplate**
- Creating reusable prompt structures
- Dynamic variable substitution
- Template formatting with `.format()`

### 2. **ChatPromptTemplate**
- Structured chat-based prompts
- System and user message separation
- Context-aware conversations

### 3. **Practical Applications**
- Real-world use cases of prompt engineering
- Streamlit-based interactive applications
- Domain-specific implementations

## 📁 Files Description

### `prompt_template.py`
- **Purpose**: Research assistant tool
- **Features**:
  - Multi-variable templates (paper, language, format)
  - Dropdown selections for user input
  - Multiple output formats (summarization, key points, facts)
- **Use Case**: Academic research paper analysis

### `chatprompt_template.py`
- **Purpose**: Basic ChatPromptTemplate setup
- **Use Case**: Foundation for chat-based applications

### `email_spam_classifier.py`
- **Purpose**: Email spam detection
- **Features**:
  - Binary classification (Spam/Not Spam)
  - Text area input for email content
  - Real-time classification with Gemini model
- **Technology**: Streamlit + Gemini AI
- **Use Case**: Email filtering and security

### `movie_recommendation_system.py`
- **Purpose**: Personalized movie recommendations
- **Features**:
  - User preference-based recommendations
  - Generates 3 movie suggestions with descriptions
  - Interactive text input interface
- **Technology**: Streamlit + Gemini AI
- **Use Case**: Content recommendation system

### `simple_usser_prompt.py`
- **Purpose**: Basic chatbot implementation
- **Features**:
  - Simple text input/output
  - Direct model invocation
  - Minimal prompt engineering
- **Use Case**: General-purpose chatbot

## 🔑 Key Concepts

### Template Variables
```python
prompt = PromptTemplate(
    template="Explain {topic} in {language}",
    input_variables=["topic", "language"]
)
```

### Template Formatting
```python
formatted_prompt = prompt.format(topic="AI", language="simple terms")
```

### System vs Human Messages
- **System Message**: Defines AI behavior and role
- **Human Message**: User's query or request

## 💡 Usage Pattern

```python
from langchain import PromptTemplate

prompt = PromptTemplate(
    template="You are a {role}. {instruction}",
    input_variables=["role", "instruction"]
)

user_input = prompt.format(
    role="teacher", 
    instruction="Explain quantum physics"
)
```

## 🛠️ Dependencies

- `langchain`
- `langchain_core`
- `langchain_google_genai`
- `streamlit`
- `python-dotenv`

## 🎯 Applications Showcase

| Application | Input | Output | Technology |
|-------------|-------|--------|------------|
| Research Tool | Paper title, language, format | Summary/Key points/Facts | Gemini + Streamlit |
| Spam Classifier | Email content | Spam/Not Spam | Gemini + Streamlit |
| Movie Recommender | User preferences | 3 movie recommendations | Gemini + Streamlit |
| Simple Chatbot | Text query | AI response | Gemini + Streamlit |

## 🎨 Prompt Engineering Best Practices

1. **Be Specific**: Clearly define the AI's role and task
2. **Use Variables**: Make templates reusable with placeholders
3. **Provide Context**: Include relevant background information
4. **Set Format**: Specify desired output structure
5. **Test Iterations**: Refine prompts based on outputs

## 🎓 Learning Outcomes

After reviewing these examples, you should understand:
- Creating and using prompt templates
- Building interactive AI applications with Streamlit
- Applying prompt engineering to real-world problems
- Difference between simple prompts and chat prompts
- Variable substitution and dynamic content generation
- Domain-specific AI applications (classification, recommendation, research)
