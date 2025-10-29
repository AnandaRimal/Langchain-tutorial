# 🛠️ Tools in LangChain

## 📋 Overview

This folder contains comprehensive implementations of Tools in LangChain. Tools allow LLMs to interact with external functions, APIs, and services, transforming them from text generators into agents that can take actions and retrieve real-time information.

## 🎯 What is a Tool in LangChain?

A **Tool** is a Python function or external API wrapped in a structure that a LangChain LLM can understand and use.

**Every Tool has:**
- **Name:** Unique identifier for the tool
- **Description:** Explains what the tool does (LLM uses this to decide when to use it)
- **Input Schema:** Defines the arguments it accepts (with types and descriptions)
- **Function:** The actual code that executes when the tool is called

**Why Tools Matter:**
- 🔍 Enable LLMs to access real-time information (search, databases, APIs)
- 🧮 Perform calculations and data processing
- 🌐 Interact with external services (Wikipedia, Google Search, etc.)
- 🤖 Build autonomous agents that can use multiple tools
- 📊 Query databases and process structured data

## 📁 Files in this Folder

| File | Description |
|------|-------------|
| `tool.ipynb` | Complete notebook with tool creation methods and integrations |

## 🔧 Types of Tools

### 1️⃣ Python Functions
Simple sync/async functions wrapped with `@tool` decorator or `StructuredTool`

**Examples:** Add, multiply, data processing functions

### 2️⃣ External APIs
Tools that call external services and APIs

**Examples:** Google Search, Weather API, Wikipedia, DuckDuckGo

### 3️⃣ Custom Agents / Complex Workflows
Multi-step chains, database queries, or scraping pipelines

**Examples:** RAG systems, database tools, web scrapers

---

## 🚀 Creating Tools in LangChain

### Method 1: @tool Decorator (Simplest)

**Best for:** Quick tool creation, simple functions

```python
from langchain_core.tools import tool

@tool
def multiply(a: int, b: int) -> int:
    """Multiply two numbers."""
    return a * b

@tool
def add(a: int, b: int) -> int:
    """Adds a and b."""
    return a + b

# Inspect tool attributes
print(multiply.name)         # Output: 'multiply'
print(multiply.description)  # Output: 'Multiply two numbers.'
print(multiply.args)         # Shows argument schema
```

**Features:**
- ✅ Minimal boilerplate
- ✅ Automatic schema inference from type hints
- ✅ Uses docstring as description
- ⚠️ Limited customization

---

### Method 2: StructuredTool (Explicit, Supports Async)

**Best for:** Async operations, dynamic tool creation, more control

```python
from langchain_core.tools import StructuredTool

def multiply(a: int, b: int) -> int:
    """Multiply two numbers."""
    return a * b

async def amultiply(a: int, b: int) -> int:
    """Multiply two numbers asynchronously."""
    return a * b

calculator = StructuredTool.from_function(
    func=multiply,      # Sync version
    coroutine=amultiply # Async version
)

# Sync call
print(calculator.invoke({"a": 2, "b": 3}))  # Output: 6

# Async call
print(await calculator.ainvoke({"a": 2, "b": 5}))  # Output: 10
```

**Features:**
- ✅ Supports both sync and async in one tool
- ✅ Better for API calls and I/O operations
- ✅ Non-blocking execution
- ✅ Can be dynamically created

**Sync vs Async:**
- **Sync Tool** (`invoke()`): Blocks execution until done
- **Async Tool** (`ainvoke()`): Runs without blocking, can handle concurrent operations

**Use Case:** Tools calling external APIs where waiting for response shouldn't block other tasks

---

### Method 3: StructuredTool with Pydantic (Industry Standard)

**Best for:** Production applications, strict validation, complex schemas

```python
from langchain.tools import StructuredTool
from pydantic import BaseModel, Field

class MultiplyInput(BaseModel):
    a: int = Field(required=True, description="The first number to multiply")
    b: int = Field(required=True, description="The second number to multiply")

def multiply_func(a: int, b: int) -> int:
    return a * b

multiply_tool = StructuredTool.from_function(
    func=multiply_func,
    name="multiply",
    description="Multiply two numbers",
    args_schema=MultiplyInput
)
```

**Features:**
- ✅ Explicit input validation
- ✅ Better error messages
- ✅ Self-documenting schemas
- ✅ Industry-standard approach
- ✅ Type safety guaranteed

---

### Method 4: BaseTool Class (Full Control)

**Best for:** Complex tools, custom validation, stateful tools

```python
from langchain.tools import BaseTool
from pydantic import BaseModel, Field
from typing import Type

class MultiplyInput(BaseModel):
    a: int = Field(required=True, description="The first number to multiply")
    b: int = Field(required=True, description="The second number to multiply")

class MultiplyTool(BaseTool):
    name: str = "multiply"
    description: str = "Multiply two numbers"
    args_schema: Type[BaseModel] = MultiplyInput

    def _run(self, a: int, b: int) -> int:
        """Synchronous execution"""
        return a * b
    
    async def _arun(self, a: int, b: int) -> int:
        """Asynchronous execution"""
        return a * b

# Create instance
multiply_tool = MultiplyTool()

# Use it
result = multiply_tool.invoke({'a': 3, 'b': 3})  # Output: 9
```

**Features:**
- ✅ Maximum flexibility and control
- ✅ Can maintain state
- ✅ Custom validation logic
- ✅ Both sync and async support
- ✅ Can override behavior methods

---

## 🌐 Built-in Tools & Integrations

LangChain provides ready-to-use tools for popular services:

### 1️⃣ Wikipedia Tool

```python
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper

api_wrapper = WikipediaAPIWrapper(
    top_k_results=5,
    doc_content_chars_max=500
)
wiki_tool = WikipediaQueryRun(api_wrapper=api_wrapper)

result = wiki_tool.invoke({"query": "artificial intelligence"})
print(result)
```

**Use Cases:** Research, fact-checking, knowledge retrieval

---

### 2️⃣ DuckDuckGo Search Tool

```python
from langchain_community.tools import DuckDuckGoSearchRun

search_tool = DuckDuckGoSearchRun()

results = search_tool.invoke('latest AI news 2025')
print(results)
```

**Use Cases:** Real-time information, news, current events

---

### 3️⃣ ArXiv Tool (Research Papers)

```python
from langchain_community.utilities import ArxivAPIWrapper

arxiv_tool = ArxivAPIWrapper()

# Search by paper ID
result = arxiv_tool.run("1706.03762")  # "Attention is All You Need" paper
print(result)
```

**Use Cases:** Academic research, paper summaries, citations

---

### Other Available Integrations:

| Tool | Purpose | Package |
|------|---------|---------|
| **Gmail** | Email management | `langchain_community` |
| **Google Search** | Web search | `langchain_google_community` |
| **Google Drive** | File management | `langchain_google_community` |
| **Slack** | Team communication | `langchain_community` |
| **GitHub** | Code repository access | `langchain_community` |
| **SQL Database** | Database queries | `langchain_community` |
| **Python REPL** | Execute Python code | `langchain_experimental` |
| **Calculator** | Math operations | `langchain_community` |

📚 **Full List:** [LangChain Tools Documentation](https://python.langchain.com/docs/integrations/tools/)

---

## 🤖 Using Tools with LLMs

### Basic Tool Binding

```python
from langchain.chat_models import init_chat_model
from langchain_core.tools import tool

# Define tools
@tool
def multiply(a: int, b: int) -> int:
    """Multiply two numbers."""
    return a * b

@tool
def add(a: int, b: int) -> int:
    """Adds a and b."""
    return a + b

# Initialize LLM
llm = init_chat_model("gemini-2.5-flash", model_provider="google_genai")

# Bind tools to LLM
tools = [multiply, add, wiki_tool]
llm_with_tools = llm.bind_tools(tools)
```

---

### Single Tool Call Example

```python
from langchain_core.messages import HumanMessage

query = "What is the sum of 2 and 3?"
messages = [HumanMessage(query)]

# LLM decides to use the 'add' tool
response = llm_with_tools.invoke(query)
print(response.tool_calls)
# Output: [{'name': 'add', 'args': {'a': 2, 'b': 3}, 'id': '...'}]

# Execute the tool
for tool_call in response.tool_calls:
    selected_tool = {"add": add, "multiply": multiply}[tool_call["name"].lower()]
    tool_result = selected_tool.invoke(tool_call)
    print(tool_result)  # Output: 5
```

---

### Multiple Tool Calls Example

```python
from langchain_core.messages import HumanMessage

query = "What is LangChain and what is 5*15?"
messages = [HumanMessage(query)]

ai_msg = llm_with_tools.invoke(messages)
print(ai_msg.tool_calls)
# Output: [
#   {'name': 'wikipedia', 'args': {'query': 'LangChain'}, ...},
#   {'name': 'multiply', 'args': {'a': 5, 'b': 15}, ...}
# ]

# Execute all tools
for tool_call in ai_msg.tool_calls:
    selected_tool = {
        "add": add,
        "multiply": multiply,
        "wikipedia": wiki_tool
    }[tool_call["name"].lower()]
    
    tool_msg = selected_tool.invoke(tool_call)
    messages.append(tool_msg)

# Get final response
final_response = llm_with_tools.invoke(messages)
print(final_response.content)
```

---

## 🔄 Tool Execution Flow

```
User Query
    ↓
LLM analyzes query
    ↓
LLM decides which tool(s) to use
    ↓
LLM generates tool calls with arguments
    ↓
Tool(s) execute and return results
    ↓
Results added to message history
    ↓
LLM synthesizes final response
    ↓
User receives answer
```

---

## 📊 Tool Comparison Table

| Method | Complexity | Validation | Async Support | Best For |
|--------|-----------|------------|---------------|----------|
| **@tool Decorator** | Low | Automatic | ❌ | Simple functions, quick prototyping |
| **StructuredTool** | Medium | Automatic | ✅ | API calls, I/O operations |
| **StructuredTool + Pydantic** | Medium | Explicit | ✅ | Production apps, strict validation |
| **BaseTool Class** | High | Custom | ✅ | Complex logic, stateful tools |

---

## 💡 Best Practices

### 1. Write Clear Descriptions
```python
@tool
def search_database(query: str) -> str:
    """
    Search the customer database for relevant information.
    Use this when the user asks about customer data, orders, or history.
    """
    return database.search(query)
```

### 2. Use Type Hints
```python
def calculate(a: int, b: int, operation: str) -> int:
    """Type hints help LLM understand expected inputs"""
    pass
```

### 3. Handle Errors Gracefully
```python
@tool
def api_call(endpoint: str) -> str:
    """Call external API with error handling"""
    try:
        response = requests.get(endpoint)
        return response.json()
    except Exception as e:
        return f"Error: {str(e)}"
```

### 4. Keep Tools Focused
```python
# ✅ Good - Single responsibility
@tool
def get_weather(city: str) -> str:
    """Get current weather for a city"""
    pass

# ❌ Bad - Too many responsibilities
@tool
def get_weather_and_forecast_and_alerts(city: str) -> str:
    """Get weather, forecast, and alerts"""
    pass
```

---

## 🎓 Common Use Cases

### 1. Research Assistant
```python
tools = [wiki_tool, arxiv_tool, search_tool]
# LLM can search Wikipedia, academic papers, and web
```

### 2. Data Analyst Agent
```python
tools = [sql_query_tool, calculator_tool, plot_tool]
# LLM can query databases, calculate, and visualize
```

### 3. Customer Support Bot
```python
tools = [knowledge_base_tool, order_lookup_tool, ticket_creation_tool]
# LLM can search KB, check orders, create tickets
```

### 4. Code Assistant
```python
tools = [github_search_tool, code_executor_tool, documentation_tool]
# LLM can search repos, run code, find docs
```

---

## 🚀 Getting Started

### Prerequisites
```bash
pip install langchain langchain-community langchain-google-genai
pip install wikipedia duckduckgo-search arxiv
```

### Quick Start
```python
from langchain_core.tools import tool
from langchain.chat_models import init_chat_model

# 1. Define a tool
@tool
def calculator(expression: str) -> float:
    """Evaluate a mathematical expression"""
    return eval(expression)

# 2. Initialize LLM
llm = init_chat_model("gemini-2.5-flash", model_provider="google_genai")

# 3. Bind tools
llm_with_tools = llm.bind_tools([calculator])

# 4. Use it
response = llm_with_tools.invoke("What is 15% of 200?")
print(response.tool_calls)
```

---

## 🔍 Advanced Topics

### Tool Routing
```python
# LLM automatically selects the right tool based on query
tools = [weather_tool, stock_tool, news_tool]
# Query: "What's the weather?" → uses weather_tool
# Query: "AAPL stock price?" → uses stock_tool
```

### Tool Chaining
```python
# Output of one tool becomes input to another
search_result = search_tool.invoke("AI news")
summary = summarize_tool.invoke(search_result)
```

### Custom Tool Return Types
```python
@tool
def get_data() -> dict:
    """Return structured data"""
    return {"status": "success", "data": [...]}
```

---

## 🎯 Key Takeaways

✅ Tools extend LLMs beyond text generation to real-world actions

✅ Four methods: @tool, StructuredTool, Pydantic, BaseTool (choose based on complexity)

✅ Built-in integrations for Wikipedia, Search, ArXiv, and 100+ services

✅ LLMs automatically decide when and how to use tools

✅ Tools enable autonomous agents and multi-step reasoning

✅ Always write clear descriptions - LLM uses them to choose tools

✅ Use async tools for I/O operations and API calls

---

## 🔗 Related Concepts

- **Agents:** Autonomous systems that use multiple tools
- **Chains:** Sequential tool execution pipelines
- **Memory:** Tools can store and retrieve information
- **RAG:** Tools can retrieve documents and data sources

---

## 📚 Resources

- [LangChain Tools Documentation](https://python.langchain.com/docs/modules/tools/)
- [Tool Integrations](https://python.langchain.com/docs/integrations/tools/)
- [Building Custom Tools](https://python.langchain.com/docs/how_to/custom_tools/)
- [Async Tools Guide](https://python.langchain.com/docs/how_to/async_tools/)

---

**Created for LangChain Tutorial Repository**  
Demonstrates practical implementations of Tools for building LLM-powered applications and agents.
