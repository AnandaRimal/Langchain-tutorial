# Output Parsers

This folder contains examples and implementations of various Output Parsers in LangChain.

## 📖 Overview

Output Parsers are LangChain classes that structure and validate raw text output from LLMs into more usable formats. They ensure the model's response conforms to a specific schema or structure that your application expects, transforming free-form text into structured, type-safe Python objects.

## 🎯 Why Use Output Parsers?

1. **Consistency**: Get structured data instead of free-form text
2. **Validation**: Catch malformed responses early in the pipeline
3. **Type Safety**: Convert strings to proper Python objects
4. **Reliability**: Ensure downstream code receives expected formats
5. **Error Handling**: Automatic retry and repair mechanisms
6. **Integration**: Seamless integration with LangChain chains

## 📁 Files Description

### `outputparsers.ipynb`
- **Purpose**: Comprehensive guide to LangChain output parsers
- **Parsers Covered**: StrOutputParser, JsonOutputParser, StructuredOutputParser, PydanticOutputParser
- **Format**: Jupyter Notebook with interactive examples

## 🔑 Parser Types Covered

### 1. StrOutputParser
**Purpose**: Simplest parser for basic text processing

**Features**:
- Extracts plain string from LLM response
- Converts `AIMessage` object to raw string
- No validation or transformation
- Fastest and most straightforward

**Use Case**: When you need clean text output without structure

**Example**:
```python
from langchain_core.output_parsers import StrOutputParser

parser = StrOutputParser()
model = ChatGoogleGenerativeAI(model="gemini-2.5-pro")

response = model.invoke("Hello, how are you?")
parsed_response = parser.invoke(response)  # Returns: str
```

**Advanced Example - Chain Processing**:
```python
# Generate detailed report
template1 = PromptTemplate(
    template='Write a detailed report on {topic}',
    input_variables=['topic']
)

# Summarize the report
template2 = PromptTemplate(
    template='Write a 5 line summary on the following text. \n {text}',
    input_variables=['text']
)

# First generate, then summarize
prompt1 = template1.invoke({'topic': 'black hole'})
result = model.invoke(prompt1)

prompt2 = template2.invoke({'text': result.content})
result1 = model.invoke(prompt2)

# Parse to string
final_output = parser.invoke(result1.content)
```

---

### 2. JsonOutputParser
**Purpose**: Parse LLM output into JSON/Python dictionary

**Features**:
- Converts text to Python dict or list
- Provides format instructions to the LLM
- Automatic JSON validation
- Handles arrays and nested objects

**Use Case**: When you need structured key-value data without strict schema

**Example**:
```python
from langchain_core.output_parsers import JsonOutputParser

parser = JsonOutputParser()

template = PromptTemplate(
    template="""Give me 5 facts about {topic} as a JSON array of strings.
    
    {format_instructions}
    
    Topic: {topic}""",
    input_variables=['topic'],
    partial_variables={'format_instructions': parser.get_format_instructions()}
)

chain = template | model | parser
result = chain.invoke({'topic': 'black hole'})

# Result: ['fact1', 'fact2', 'fact3', 'fact4', 'fact5']
# Type: list or dict
```

**Format Instructions**:
The parser automatically generates instructions like:
```
Return a JSON object with the requested information.
```

---

### 3. StructuredOutputParser
**Purpose**: Parse output into predefined structure with named fields

**Features**:
- Define schema using `ResponseSchema`
- Named fields with descriptions
- Dictionary output with specific keys
- Better control than JsonOutputParser

**Use Case**: When you need specific named fields in a dictionary

**Example**:
```python
from langchain.output_parsers import StructuredOutputParser, ResponseSchema

# Define schema
schema = [
    ResponseSchema(name='fact_1', description='Fact 1 about the topic'),
    ResponseSchema(name='fact_2', description='Fact 2 about the topic'),
    ResponseSchema(name='fact_3', description='Fact 3 about the topic'),
]

parser = StructuredOutputParser.from_response_schemas(schema)

template = PromptTemplate(
    template='Give 3 facts about {topic} \n {format_instruction}',
    input_variables=['topic'],
    partial_variables={'format_instruction': parser.get_format_instructions()}
)

chain = template | model | parser
result = chain.invoke({'topic': 'black hole'})

# Result: {'fact_1': '...', 'fact_2': '...', 'fact_3': '...'}
```

---

### 4. PydanticOutputParser
**Purpose**: Parse output into Pydantic models with type validation

**Features**:
- Full Pydantic model support
- Type validation and constraints
- Field descriptions and validation rules
- Best type safety and IDE support
- Automatic error messages

**Use Case**: When you need strict type checking and validation

**Example**:
```python
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

# Define Pydantic model
class Person(BaseModel):
    name: str = Field(description='Name of the person')
    age: int = Field(gt=18, description='Age of the person')
    city: str = Field(description='Name of the city the person belongs to')

parser = PydanticOutputParser(pydantic_object=Person)

template = PromptTemplate(
    template='Generate the name, age and city of a fictional {place} person \n {format_instruction}',
    input_variables=['place'],
    partial_variables={'format_instruction': parser.get_format_instructions()}
)

chain = template | model | parser
result = chain.invoke({'place': 'sri lankan'})

# Result: Person(name='...', age=25, city='...')
# Type: Person (Pydantic model)
```

**Benefits**:
- `age` must be > 18 (gt=18)
- Type checking (str, int)
- IDE autocomplete: `result.name`, `result.age`, `result.city`

---

## 📊 Parser Comparison Table

| Parser | Output Type | Validation | Type Safety | Use Case |
|--------|-------------|------------|-------------|----------|
| **StrOutputParser** | `str` | None | Low | Plain text extraction |
| **JsonOutputParser** | `dict`/`list` | JSON format | Medium | Flexible JSON data |
| **StructuredOutputParser** | `dict` | Named fields | Medium | Specific key-value pairs |
| **PydanticOutputParser** | Pydantic Model | Full validation | High | Strict type checking |

## 🔗 LangChain Chains Integration

All parsers work seamlessly with LangChain Expression Language (LCEL):

```python
# Chain: Template → Model → Parser
chain = template | model | parser
result = chain.invoke({'input': 'value'})
```

This creates a pipeline where:
1. **Template** formats the prompt
2. **Model** generates the response
3. **Parser** structures the output

## 🎯 Choosing the Right Parser

### Use StrOutputParser when:
- You need simple text extraction
- No structure required
- Maximum flexibility in output format

### Use JsonOutputParser when:
- You want JSON/dictionary output
- Schema is flexible or dynamic
- Don't need strict validation

### Use StructuredOutputParser when:
- You need specific named fields
- Want clear field descriptions
- Medium level of validation needed

### Use PydanticOutputParser when:
- You need strict type checking
- Want validation constraints (min, max, regex)
- Building production applications
- IDE autocomplete is important
- Integrating with existing Pydantic models

## 💡 Best Practices

1. **Include Format Instructions**: Always use `parser.get_format_instructions()` in your prompts
2. **Start Simple**: Begin with StrOutputParser, move to complex parsers as needed
3. **Validate Early**: Use parsers to catch errors before processing
4. **Use Pydantic for Production**: Most reliable for real applications
5. **Test Parser Output**: Verify the parsed structure matches expectations
6. **Handle Parsing Errors**: Wrap parsing in try-catch for LLM hallucinations

## 🛠️ Dependencies

- `langchain_core`
- `langchain`
- `langchain_google_genai` (or other LLM providers)
- `langchain_huggingface`
- `pydantic`
- `python-dotenv`

## 🔧 Advanced Features

### Format Instructions
Parsers generate instructions for the LLM:
```python
instructions = parser.get_format_instructions()
# Includes JSON schema, field descriptions, validation rules
```

### Error Handling
```python
from langchain.output_parsers import OutputFixingParser

try:
    result = parser.parse(response)
except Exception as e:
    # Use OutputFixingParser to auto-fix malformed output
    fixing_parser = OutputFixingParser.from_llm(parser=parser, llm=model)
    result = fixing_parser.parse(response)
```

### Custom Parsers
You can create custom parsers by extending `BaseOutputParser`:
```python
from langchain_core.output_parsers import BaseOutputParser

class CustomParser(BaseOutputParser):
    def parse(self, text: str):
        # Your custom parsing logic
        return processed_output
```

## 🎓 Learning Outcomes

After reviewing these examples, you should understand:
- Different types of output parsers and their purposes
- How to structure LLM outputs for downstream processing
- Integration of parsers with LangChain chains
- Validation and type safety in LLM applications
- When to use each parser type
- Best practices for reliable LLM output handling

## 🚀 Real-World Applications

1. **Data Extraction**: Extract structured data from documents
2. **Form Filling**: Parse LLM responses into database records
3. **API Integration**: Convert LLM output to API-compatible formats
4. **Chatbots**: Structure conversational responses
5. **Analytics**: Convert insights into queryable data structures
6. **Content Generation**: Structure blog posts, reports, etc.

## 📚 Further Reading

- [LangChain Output Parsers Documentation](https://python.langchain.com/docs/modules/model_io/output_parsers/)
- [Pydantic Documentation](https://docs.pydantic.dev/)
- [LangChain Expression Language (LCEL)](https://python.langchain.com/docs/expression_language/)
- [JSON Schema Specification](https://json-schema.org/)
