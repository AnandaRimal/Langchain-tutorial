# Chains

This folder contains implementations and examples of various Chain types in LangChain.

## 📖 Overview

A **Chain** is an end-to-end wrapper around multiple components, executed in a specific sequence to accomplish complex tasks. Think of it like a recipe where you follow steps in order rather than throwing everything together at once.

**Core Concept**: Chaining different operations together, where the output of one step becomes the input to the next.

## 🎯 Why Use Chains?

1. **Modularity**: Break complex tasks into manageable steps
2. **Reusability**: Create reusable workflows for common patterns
3. **Maintainability**: Easier to debug and modify individual steps
4. **Flexibility**: Combine LLMs, data retrieval, code execution, and more
5. **Efficiency**: Enable parallel processing for independent tasks

## 📁 Files Description

### `chain.ipynb`
- **Purpose**: Comprehensive guide to LangChain chains
- **Chain Types Covered**: LLMChain, SequentialChain, Parallel Chains, Conditional Chains
- **Format**: Jupyter Notebook with interactive examples

## 🔑 Chain Types Covered

### 1. LLMChain (Basic Chain)

**Purpose**: Simplest chain combining a prompt template with an LLM

**Features**:
- Single prompt template
- Single LLM call
- Direct input/output mapping

**Use Case**: Basic prompt → LLM → response workflow

**Example**:
```python
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

model = ChatGoogleGenerativeAI(model="gemini-2.5-pro")

prompt = PromptTemplate(
    input_variables=["topic"],
    template="Write a short poem about {topic}:"
)

chain = LLMChain(llm=model, prompt=prompt)
result = chain.invoke({"topic": "the ocean"})
```

**Flow**: 
```
Input (topic) → Prompt Template → LLM → Output (poem)
```

---

### 2. Sequential Chains

**Purpose**: Run multiple chains in sequence where each step depends on previous outputs

**Types**:
- **SimpleSequentialChain**: Single input → single output (each step)
- **SequentialChain**: Multiple inputs → multiple outputs

**Features**:
- Ordered execution
- Output of one chain becomes input to next
- Multiple intermediate outputs
- Verbose mode for debugging

**Use Case**: Multi-step workflows like generate → refine → validate

**Example**:
```python
from langchain.chains import LLMChain, SequentialChain

# Step 1: Generate startup name
prompt1 = PromptTemplate(
    input_variables=["product"],
    template="Give me a creative name for a {product} startup."
)
chain1 = LLMChain(llm=model, prompt=prompt1, output_key="startup_name")

# Step 2: Generate tagline
prompt2 = PromptTemplate(
    input_variables=["startup_name"],
    template="Write a catchy tagline for a startup named {startup_name}."
)
chain2 = LLMChain(llm=model, prompt=prompt2, output_key="tagline")

# Combine sequentially
overall_chain = SequentialChain(
    chains=[chain1, chain2],
    input_variables=["product"],
    output_variables=["startup_name", "tagline"],
    verbose=True
)

result = overall_chain({"product": "AI-powered language learning app"})
# Output: {'startup_name': '...', 'tagline': '...'}
```

**Flow**:
```
Input (product) → Chain 1 (name) → Chain 2 (tagline) → Output (name + tagline)
```

---

### 3. Parallel Chains

**Purpose**: Execute multiple chains simultaneously for improved efficiency

**Benefits**:
- **Performance**: Reduce total execution time
- **Modularity**: Independent processing of different aspects
- **Flexibility**: Combine diverse operations
- **Error Isolation**: One chain failing doesn't stop others

**Use Case**: When multiple independent operations can run concurrently

**Implementation**: Using `RunnableParallel` (LCEL - Modern Approach)

**Example - Study Material Generator**:
```python
from langchain.schema.runnable import RunnableParallel
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

model1 = ChatGoogleGenerativeAI(model="gemini-2.5-pro")
model2 = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

prompt1 = PromptTemplate(
    template='Generate short and simple notes from the following text \n {text}',
    input_variables=['text']
)

prompt2 = PromptTemplate(
    template='Generate 5 short question answers from the following text \n {text}',
    input_variables=['text']
)

prompt3 = PromptTemplate(
    template='Merge the provided notes and quiz into a single document \n notes -> {notes} and quiz -> {quiz}',
    input_variables=['notes', 'quiz']
)

parser = StrOutputParser()

# Parallel execution: notes and quiz generated simultaneously
parallel_chain = RunnableParallel({
    'notes': prompt1 | model1 | parser,
    'quiz': prompt2 | model2 | parser
})

# Sequential merge of parallel results
merge_chain = prompt3 | model1 | parser

# Complete chain
chain = parallel_chain | merge_chain

result = chain.invoke({'text': "Your educational content here..."})
```

**Flow**:
```
                    ┌─→ Chain A (notes) ─┐
Input (text) ──┬───┤                      ├──→ Merge Chain → Output
                    └─→ Chain B (quiz) ──┘
                    (executed in parallel)
```

**Advantages**:
- Notes and quiz generated simultaneously (faster)
- Can use different models for different tasks
- Results combined in final merge step

**Visualization**:
```python
# View chain structure
chain.get_graph().print_ascii()
```

---

### 4. Conditional Chains (Branching)

**Purpose**: Create dynamic workflows that route execution based on conditions

**Features**:
- Conditional routing based on input or previous outputs
- Different execution paths
- Dynamic decision making
- Integration with classification/sentiment analysis

**Use Case**: Different responses based on sentiment, content type, or user intent

**Implementation**: Using `RunnableBranch` (Recommended)

**Example - Sentiment-Based Customer Feedback**:
```python
from langchain.schema.runnable import RunnableBranch, RunnableLambda
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import Literal

# Define sentiment model
class Feedback(BaseModel):
    sentiment: Literal['positive', 'negative'] = Field(
        description='Give the sentiment of the feedback'
    )

parser2 = PydanticOutputParser(pydantic_object=Feedback)

# Step 1: Classify sentiment
prompt1 = PromptTemplate(
    template='Classify the sentiment of the following feedback text into positive or negative \n {feedback} \n {format_instruction}',
    input_variables=['feedback'],
    partial_variables={'format_instruction': parser2.get_format_instructions()}
)

classifier_chain = prompt1 | model | parser2

# Step 2: Positive response template
prompt2 = PromptTemplate(
    template='Write an appropriate response to this positive feedback \n {feedback}',
    input_variables=['feedback']
)

# Step 3: Negative response template
prompt3 = PromptTemplate(
    template='Write an appropriate response to this negative feedback \n {feedback}',
    input_variables=['feedback']
)

# Branch based on sentiment
branch_chain = RunnableBranch(
    (lambda x: x.sentiment == 'positive', prompt2 | model | parser),
    (lambda x: x.sentiment == 'negative', prompt3 | model | parser),
    RunnableLambda(lambda x: "Could not determine sentiment")  # Default
)

# Complete chain: classify → branch → respond
chain = classifier_chain | branch_chain

# Test
result = chain.invoke({'feedback': "This is a beautiful phone"})
# Routes to positive response template
```

**Flow**:
```
Input (feedback) → Classifier → ┬─→ (if positive) → Positive Response Template → Output
                                   │
                                   └─→ (if negative) → Negative Response Template → Output
```

**Key Components**:
1. **Classifier**: Determines which path to take
2. **Branches**: Different execution paths (positive/negative)
3. **Default**: Fallback if no condition matches
4. **Pydantic Model**: Structured sentiment classification

**Real-World Use Cases**:
- Customer support automation
- Content moderation
- Email routing
- Dynamic response generation

---

## 📊 Chain Comparison Table

| Chain Type | Execution | Dependencies | Performance | Use Case |
|------------|-----------|--------------|-------------|----------|
| **LLMChain** | Single step | None | Fast | Simple prompt-response |
| **Sequential** | One after another | Each step depends on previous | Slower | Multi-step workflows |
| **Parallel** | Simultaneous | Independent steps | Fastest | Multiple independent tasks |
| **Conditional** | Dynamic routing | Based on conditions | Variable | Decision-based workflows |

## 🎯 Choosing the Right Chain

### Use LLMChain when:
- Single prompt-response interaction
- No multi-step logic needed
- Simple, straightforward task

### Use Sequential Chain when:
- Output of one step feeds into the next
- Linear workflow required
- Steps must execute in order

### Use Parallel Chain when:
- Multiple independent operations
- Performance is critical
- Tasks can run simultaneously

### Use Conditional Chain when:
- Different execution paths based on input
- Dynamic decision making required
- Sentiment/classification-based routing

## 💡 LangChain Expression Language (LCEL)

Modern chains use the **pipe operator** (`|`) for clean, readable syntax:

```python
# Old style
chain = LLMChain(llm=model, prompt=prompt)

# New LCEL style
chain = prompt | model | parser
```

**Benefits**:
- More Pythonic and readable
- Better composition
- Built-in parallelization support
- Easier debugging with `.get_graph()`

## 🔧 Advanced Features

### Chain Visualization
```python
# View chain structure
chain.get_graph().print_ascii()
```

### Verbose Mode
```python
# See intermediate outputs
chain = SequentialChain(..., verbose=True)
```

### Custom Lambda Functions
```python
from langchain.schema.runnable import RunnableLambda

custom_step = RunnableLambda(lambda x: x.upper())
chain = prompt | model | custom_step
```

### Error Handling
```python
try:
    result = chain.invoke({"input": "value"})
except Exception as e:
    # Handle chain execution errors
    print(f"Chain failed: {e}")
```

## 🛠️ Dependencies

- `langchain`
- `langchain_core`
- `langchain_google_genai`
- `pydantic`
- `python-dotenv`

## 🎓 Learning Outcomes

After reviewing these examples, you should understand:
- Different chain types and their purposes
- When to use sequential vs parallel execution
- How to implement conditional logic in chains
- LCEL syntax and modern chain composition
- Performance optimization through parallelization
- Building complex multi-step workflows
- Integration of parsers and validators in chains

## 🚀 Real-World Applications

### 1. Content Generation Pipeline
```
Sequential: Research → Draft → Edit → Format
```

### 2. Document Analysis
```
Parallel: Summarize + Extract Keywords + Sentiment Analysis → Combine
```

### 3. Customer Support
```
Conditional: Classify Issue → Route to Department → Generate Response
```

### 4. Educational Tools
```
Parallel: Generate Notes + Create Quiz → Merge into Study Guide
```

### 5. Data Processing
```
Sequential: Load Data → Clean → Analyze → Visualize
```

## 📈 Performance Considerations

### Sequential Chains
- **Time**: Sum of all steps
- **Memory**: One step at a time
- **Best for**: Dependent operations

### Parallel Chains
- **Time**: Max of all parallel steps
- **Memory**: All steps simultaneously
- **Best for**: Independent operations

**Example**:
- Sequential: 3 steps × 2 seconds = 6 seconds total
- Parallel: max(2, 2, 2) = 2 seconds total

## 🎨 Best Practices

1. **Start Simple**: Begin with LLMChain, add complexity as needed
2. **Use LCEL**: Modern pipe syntax for better readability
3. **Parallelize When Possible**: Speed up independent operations
4. **Add Validation**: Use output parsers to ensure data quality
5. **Enable Verbose Mode**: Debug complex chains easily
6. **Visualize Structure**: Use `.get_graph()` to understand flow
7. **Handle Errors**: Add try-catch for production applications
8. **Test Incrementally**: Test each chain step individually

## 📚 Further Reading

- [LangChain Chains Documentation](https://python.langchain.com/docs/modules/chains/)
- [LangChain Expression Language (LCEL)](https://python.langchain.com/docs/expression_language/)
- [RunnableParallel Documentation](https://python.langchain.com/docs/expression_language/primitives/parallel)
- [RunnableBranch Documentation](https://python.langchain.com/docs/expression_language/primitives/branch)

## 🔗 Related Topics

- **Prompt Templates**: Building blocks for chains
- **Output Parsers**: Structuring chain outputs
- **Agents**: More complex decision-making systems
- **Memory**: Adding conversation context to chains
