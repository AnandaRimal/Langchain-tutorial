# Pydantic

This folder demonstrates the use of Pydantic for data validation and schema enforcement in Python applications.

## 📖 Overview

Pydantic is a data validation library that uses Python type annotations to validate data structures. It's essential for ensuring data integrity, type safety, and automatic error handling in applications, especially when working with LLMs and APIs.

## 🎯 Concepts Covered

### 1. **Data Validation**
- Automatic type checking
- Runtime validation
- Error handling with ValidationError

### 2. **BaseModel**
- Defining data schemas with classes
- Type annotations for fields
- Model validation methods

### 3. **Type Safety**
- Enforcing data types (str, int, email, etc.)
- Preventing invalid data from entering system
- Automatic type conversion where possible

## 📁 Files Description

### `pydanticuse.py`
- **Purpose**: Basic Pydantic usage demonstration
- **Key Features**:
  - `User` model with three fields (name, age, email)
  - `create_user()` function with validation
  - Examples of valid and invalid data
  - Automatic ValidationError raising

## 🔑 Key Concepts

### BaseModel Definition
```python
from pydantic import BaseModel

class User(BaseModel):
    name: str
    age: int
    email: str
```

### Data Validation
```python
# Valid data - passes validation
data = {"name": "Ananda", "age": 22, "email": "ananda@example.com"}
user = User.model_validate(data)  # ✓ Success

# Invalid data - raises ValidationError
data = {"name": "Ram", "age": "twenty", "email": "ram@example.com"}
user = User.model_validate(data)  # ✗ Error: age must be int
```

## 💡 Common Validation Types

| Type | Description | Example |
|------|-------------|---------|
| `str` | String validation | `"John"` |
| `int` | Integer validation | `25` |
| `float` | Float validation | `3.14` |
| `bool` | Boolean validation | `True` |
| `EmailStr` | Email format | `"user@email.com"` |
| `HttpUrl` | URL validation | `"https://example.com"` |
| `datetime` | Date/time validation | `datetime.now()` |

## 🎯 Use Cases in LangChain

1. **LLM Output Validation**: Ensure AI responses match expected schema
2. **API Request Validation**: Validate incoming data before processing
3. **Structured Output**: Force LLMs to return data in specific format
4. **Configuration Management**: Validate settings and parameters
5. **Database Models**: Ensure data integrity before storage

## 🛠️ Advanced Features (Not in examples but useful)

### Optional Fields
```python
from typing import Optional

class User(BaseModel):
    name: str
    age: int
    email: Optional[str] = None  # Optional field with default
```

### Field Validation
```python
from pydantic import Field

class User(BaseModel):
    name: str = Field(..., min_length=1, max_length=50)
    age: int = Field(..., ge=0, le=120)  # ge: >=, le: <=
```

### Custom Validators
```python
from pydantic import validator

class User(BaseModel):
    age: int
    
    @validator('age')
    def age_must_be_positive(cls, v):
        if v < 0:
            raise ValueError('Age must be positive')
        return v
```

## 🔒 Benefits

1. **Type Safety**: Catch type errors at runtime
2. **Automatic Validation**: No manual checking required
3. **Clear Error Messages**: Detailed validation errors
4. **IDE Support**: Better autocomplete and type hints
5. **Serialization**: Easy conversion to/from JSON
6. **Documentation**: Self-documenting code with type hints

## 🎓 Learning Outcomes

After reviewing this example, you should understand:
- How to create Pydantic models with BaseModel
- Automatic data validation with type annotations
- Handling ValidationError exceptions
- Benefits of type safety in Python
- Integration patterns with LangChain and APIs
- Difference between validated and unvalidated data

## 🔗 Integration with LangChain

Pydantic is heavily used in LangChain for:
- **Structured Output Parsers**: Parse LLM responses into Python objects
- **Function Calling**: Validate function arguments
- **Chain Inputs/Outputs**: Ensure data consistency across chain steps
- **Configuration**: Validate LangChain component settings

## 📚 Further Reading

- [Pydantic Documentation](https://docs.pydantic.dev/)
- [LangChain Structured Output](https://python.langchain.com/docs/modules/model_io/output_parsers/structured)
- [Python Type Hints](https://docs.python.org/3/library/typing.html)
