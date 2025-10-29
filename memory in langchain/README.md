# 🧠 Memory in LangChain

## 📋 Overview

This folder contains implementations and examples of different memory types in LangChain. Memory allows LLMs to remember information across multiple interactions, enabling contextual conversations and multi-step reasoning.

## 🎯 What is Memory in LangChain?

Memory in LangChain is a mechanism to store and retrieve information across multiple interactions with an LLM.

**Without Memory** → Every prompt is independent. The LLM "forgets" everything after responding.

**With Memory** → The LLM can remember previous messages, facts, or context, allowing for:
- Conversational AI with context awareness
- RAG systems with multi-step reasoning
- Agents that track tool usage and outputs
- Persistent user preferences and history

## 📁 Files in this Folder

| File | Description |
|------|-------------|
| `memory.ipynb` | Comprehensive notebook demonstrating all memory types |

## 🔧 Memory Types Implemented

### 1️⃣ ConversationBufferMemory

**What it does:** Stores ALL previous messages in a simple buffer

**Best for:**
- Short conversations
- When you need complete conversation history
- Applications with sufficient token budget

**Key Features:**
- ✅ Complete conversation context
- ✅ Simple implementation
- ❌ Can consume many tokens in long conversations

```python
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferMemory

memory = ConversationBufferMemory()
conversation = ConversationChain(llm=llm, memory=memory, verbose=True)

response = conversation.predict(input="Hi there! I am Sam")
# LLM remembers: "User said their name is Sam"
```

---

### 2️⃣ ConversationSummaryMemory

**What it does:** Summarizes past interactions to save tokens

**Best for:**
- Long conversations
- Token-limited applications
- When you need context without full history

**Key Features:**
- ✅ Token-efficient for long conversations
- ✅ Maintains essential context
- ❌ May lose some details in summarization
- ⚠️ Requires additional LLM calls for summarization

```python
from langchain.chains.conversation.memory import ConversationSummaryMemory

memory = ConversationSummaryMemory(llm=llm)
conversation = ConversationChain(llm=llm, memory=memory, verbose=True)

response = conversation.predict(input="Hi there! I am Sam")
# Creates summary: "User introduced themselves as Sam"
```

---

### 3️⃣ ConversationBufferWindowMemory

**What it does:** Stores only the last `k` messages for short-term memory

**Best for:**
- Recent context tracking
- Memory-constrained applications
- When only recent interactions matter

**Key Features:**
- ✅ Fixed memory size
- ✅ Predictable token usage
- ❌ Forgets older interactions
- ⚙️ Configurable window size with `k` parameter

```python
from langchain.chains.conversation.memory import ConversationBufferWindowMemory

memory = ConversationBufferWindowMemory(k=2)  # Remember last 2 exchanges
conversation = ConversationChain(llm=llm, memory=memory, verbose=True)

# After 3 exchanges, only the last 2 are retained
```

---

### 4️⃣ ConversationSummaryBufferMemory

**What it does:** Combines buffer and summary approaches - keeps recent messages in full and summarizes older ones

**Best for:**
- Balanced approach between context and efficiency
- Long conversations needing recent precision
- Production chatbots

**Key Features:**
- ✅ Recent messages in full detail
- ✅ Older messages summarized
- ✅ Good balance of context vs. tokens
- ⚙️ Configurable with `max_token_limit`

```python
from langchain.chains.conversation.memory import ConversationSummaryBufferMemory

memory = ConversationSummaryBufferMemory(llm=llm, max_token_limit=100)
conversation = ConversationChain(llm=llm, memory=memory, verbose=True)
```

---

### 5️⃣ ConversationKGMemory (Knowledge Graph Memory)

**What it does:** Stores information as a knowledge graph with entities and relationships

**Best for:**
- Complex entity tracking
- Relationship-based reasoning
- Domain-specific knowledge extraction
- Multi-entity conversations

**Key Features:**
- ✅ Structured knowledge representation
- ✅ Entity relationship mapping
- ✅ Efficient querying of facts
- ✅ Better reasoning capabilities
- 🔗 Can integrate with graph databases (Neo4j, NetworkX)

**How it works:**
1. **Entity Extraction:** Identifies entities (people, places, concepts) from conversation
2. **Relationship Mapping:** Connects entities with relationships
3. **Graph Traversal:** Queries the graph to answer questions

**Example:**
```
User: "Alice went to Paris and met Bob there"
Graph Structure:
- Node: Alice (person)
- Node: Paris (location)
- Node: Bob (person)
- Edge: Alice --[visited]--> Paris
- Edge: Alice --[met]--> Bob
- Edge: Bob --[located_in]--> Paris

Next Query: "Where did Alice go?"
Answer: "Paris" (by traversing Alice → visited → Paris)
```

```python
from langchain.chains.conversation.memory import ConversationKGMemory

# Initialize Knowledge Graph Memory
kg_memory = ConversationKGMemory(llm=llm)
conversation = ConversationChain(llm=llm, memory=kg_memory, verbose=True)

# Build knowledge graph through conversation
conversation.predict(input="Alice works at Google")
conversation.predict(input="Bob is Alice's manager")
conversation.predict(input="Google is located in Mountain View")

# Query relationships
conversation.predict(input="Who is Alice's manager?")
# LLM can traverse: Alice → managed_by → Bob
```

---

## 📊 Memory Comparison Table

| Memory Type | Token Usage | Context Retention | Best Use Case | Complexity |
|-------------|-------------|-------------------|---------------|------------|
| **ConversationBufferMemory** | High (all messages) | Complete | Short chats | Low |
| **ConversationSummaryMemory** | Low (summarized) | High-level only | Long conversations | Medium |
| **ConversationBufferWindowMemory** | Fixed (k messages) | Recent only | Recent context | Low |
| **ConversationSummaryBufferMemory** | Medium (balanced) | Recent + summary | Production apps | Medium |
| **ConversationKGMemory** | Low (entities only) | Structured facts | Entity tracking | High |

## 🔑 Key Concepts

### Memory Persistence
- **Default:** Memory is temporary and in-process only
- **For Production:** Use file or database-backed memory for persistence across sessions
- **Important for:** Chatbots, agents, or apps requiring continuity

### When to Use Which Memory?

```
📝 Short conversations + full context needed
   → ConversationBufferMemory

📚 Long conversations + token optimization
   → ConversationSummaryMemory

⏰ Only recent context matters
   → ConversationBufferWindowMemory

⚖️ Balance between recency and history
   → ConversationSummaryBufferMemory

🕸️ Complex entity relationships + structured knowledge
   → ConversationKGMemory (Knowledge Graph)
```

## 💡 Use Cases

### 1. Customer Support Chatbot
```python
# Remember customer name, issue, and previous interactions
memory = ConversationSummaryBufferMemory(llm=llm, max_token_limit=200)
```

### 2. Research Assistant
```python
# Track entities, papers, authors, and their relationships
memory = ConversationKGMemory(llm=llm)
```

### 3. Quick Q&A Bot
```python
# Only need last few exchanges
memory = ConversationBufferWindowMemory(k=3)
```

### 4. Personal AI Assistant
```python
# Remember everything about the user
memory = ConversationBufferMemory()
```

## 🚀 Getting Started

### Prerequisites
```bash
pip install langchain langchain-google-genai python-dotenv
```

### Basic Setup
```python
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv

# Load API keys
load_dotenv()

# Initialize LLM
llm = init_chat_model("gemini-2.5-flash", model_provider="google_genai")

# Create memory
memory = ConversationBufferMemory()

# Create conversation chain
conversation = ConversationChain(
    llm=llm,
    memory=memory,
    verbose=True
)

# Start chatting!
response = conversation.predict(input="Hi! I'm learning about LangChain memory")
```

## 🔍 Advanced Features

### Combining Multiple Memory Types
```python
from langchain.memory import CombinedMemory

# Use multiple memory strategies together
combined_memory = CombinedMemory(memories=[
    ConversationBufferWindowMemory(k=3),
    ConversationKGMemory(llm=llm)
])
```

### Custom Memory Storage
```python
# Store memory in external database
# Useful for persistent conversations across sessions
```

### Memory with Tools/Agents
```python
# Track tool calls and their results
# Remember which APIs were called and their outputs
```

## 📖 Learning Path

1. **Start with:** `ConversationBufferMemory` (simplest)
2. **Experiment with:** `ConversationBufferWindowMemory` (understand windowing)
3. **Optimize with:** `ConversationSummaryMemory` (token efficiency)
4. **Balance with:** `ConversationSummaryBufferMemory` (production-ready)
5. **Advanced:** `ConversationKGMemory` (structured knowledge)

## 🎓 Key Takeaways

✅ Memory enables contextual conversations and multi-step reasoning

✅ Different memory types trade-off between context retention and token usage

✅ ConversationKGMemory provides structured knowledge representation

✅ Production apps should use persistent storage for memory

✅ Choose memory type based on conversation length and use case

## 🔗 Related Concepts

- **Chains:** Memory is often used with ConversationChain
- **Agents:** Agents use memory to track tool usage
- **RAG Systems:** Memory helps maintain context across document retrievals
- **Prompt Templates:** Memory injects conversation history into prompts

## 📚 Resources

- [LangChain Memory Documentation](https://python.langchain.com/docs/modules/memory/)
- [Knowledge Graphs in AI](https://neo4j.com/developer/knowledge-graph/)
- [Conversation Design Best Practices](https://developers.google.com/assistant/conversational/design)

---

**Created for LangChain Tutorial Repository**  
Demonstrates practical implementations of conversation memory in LangChain applications.
