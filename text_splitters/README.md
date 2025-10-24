# Text Splitters

This folder contains implementations of various text splitting strategies in LangChain for processing large documents.

## 📖 Overview

Text Splitters divide large documents into smaller chunks for efficient processing by LLMs. Different splitting strategies preserve different aspects of the text (structure, semantics, code syntax, etc.).

## 🎯 Why Split Text?

1. **Token Limits**: LLMs have maximum context window sizes
2. **Better Retrieval**: Smaller chunks improve semantic search accuracy
3. **Cost Optimization**: Process only relevant sections
4. **Memory Management**: Reduce memory footprint for large documents
5. **Improved Accuracy**: Focused context leads to better answers

## 📁 Files Description

### `length_based.py`
- **Splitter**: CharacterTextSplitter
- **Strategy**: Fixed character length chunks
- **Key Features**:
  - Chunk size: 200 characters
  - No chunk overlap
  - Custom separator support
  - Works with PDF documents (PyPDFLoader)
- **Use Case**: Simple, uniform document splitting

### `text_structure_based.py`
- **Splitter**: RecursiveCharacterTextSplitter
- **Strategy**: Splits on natural text boundaries
- **Key Features**:
  - Chunk size: 500 characters
  - Preserves paragraph structure
  - Recursive splitting for better coherence
- **Use Case**: General text processing, articles, essays

### `markdown_splitting.py`
- **Splitter**: RecursiveCharacterTextSplitter with Language.MARKDOWN
- **Strategy**: Markdown-aware splitting
- **Key Features**:
  - Chunk size: 200 characters
  - Preserves markdown syntax (headers, lists, code blocks)
  - Respects document hierarchy
- **Use Case**: README files, documentation, markdown content

### `python_code_splitting.py`
- **Splitter**: RecursiveCharacterTextSplitter with Language.PYTHON
- **Strategy**: Python syntax-aware splitting
- **Key Features**:
  - Chunk size: 300 characters
  - Preserves class and function definitions
  - Maintains code structure and indentation
- **Use Case**: Python codebases, scripts, documentation

### `semantic_meaning_based.py`
- **Splitter**: SemanticChunker
- **Strategy**: Semantic similarity-based chunking
- **Key Features**:
  - Uses OpenAI embeddings for semantic analysis
  - Groups semantically related content together
  - Breakpoint threshold: 3 standard deviations
  - Creates context-aware chunks
- **Use Case**: Mixed-topic documents, knowledge bases

## 🔑 Key Concepts

### 1. Chunk Size
- Determines maximum characters/tokens per chunk
- Balance between context and granularity
- Typical range: 200-2000 characters

### 2. Chunk Overlap
- Number of overlapping characters between chunks
- Preserves context across boundaries
- Typical range: 0-200 characters

### 3. Separators
- Characters/patterns used to split text
- Examples: `\n\n`, `\n`, `.`, ` `, `""`

### 4. Recursive Splitting
- Tries multiple separators in order
- Falls back to smaller separators if needed
- Preserves natural document structure

## 📊 Comparison Table

| Splitter | Strategy | Best For | Preserves |
|----------|----------|----------|-----------|
| CharacterTextSplitter | Fixed length | Simple splitting | Nothing specific |
| RecursiveCharacterTextSplitter | Structure-based | General text | Paragraphs, sentences |
| Markdown Splitter | Markdown syntax | Documentation | Headers, code blocks |
| Python Code Splitter | Python syntax | Code files | Functions, classes |
| SemanticChunker | Semantic similarity | Mixed content | Topic coherence |

## 💡 Usage Patterns

### Basic Splitting
```python
from langchain.text_splitter import CharacterTextSplitter

splitter = CharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=0
)
chunks = splitter.split_text(text)
```

### Language-Specific Splitting
```python
from langchain.text_splitter import RecursiveCharacterTextSplitter, Language

splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.PYTHON,
    chunk_size=300,
    chunk_overlap=0
)
chunks = splitter.split_text(code)
```

### Semantic Splitting
```python
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai.embeddings import OpenAIEmbeddings

text_splitter = SemanticChunker(
    OpenAIEmbeddings(),
    breakpoint_threshold_type="standard_deviation"
)
docs = text_splitter.create_documents([text])
```

## 🎯 Choosing the Right Splitter

1. **General Text**: RecursiveCharacterTextSplitter
2. **Code**: Language-specific splitter (Python, JavaScript, etc.)
3. **Documentation**: Markdown splitter
4. **Mixed Topics**: SemanticChunker
5. **Simple Cases**: CharacterTextSplitter

## 🧪 Semantic Chunking Deep Dive

### How It Works
1. **Embed Text**: Convert text to vector embeddings
2. **Compute Similarity**: Compare neighboring text pieces
3. **Identify Boundaries**: Split when similarity drops significantly
4. **Group Related Content**: Keep semantically related text together

### Example
```
Paragraph 1: Cricket is a bat-and-ball game...
Paragraph 2: It involves batting strategies...
Paragraph 3: Football is another popular sport...

Result:
- Chunk 1: Paragraphs 1 + 2 (both about cricket)
- Chunk 2: Paragraph 3 (different topic - football)
```

## 🛠️ Dependencies

- `langchain`
- `langchain_community` (for document loaders)
- `langchain_experimental` (for SemanticChunker)
- `langchain_openai` (for embeddings)
- `python-dotenv`

## 🎓 Learning Outcomes

After reviewing these examples, you should understand:
- Different text splitting strategies and when to use them
- Importance of chunk size and overlap parameters
- Language-specific splitting for code and documentation
- Semantic chunking for topic-based splitting
- Integration with document loaders (PDF, etc.)
- Trade-offs between different splitting approaches

## 📚 Best Practices

1. **Match Strategy to Content**: Use language-specific splitters for code
2. **Test Chunk Sizes**: Experiment with different sizes for your use case
3. **Use Overlap for Context**: Add overlap when context is important
4. **Monitor Performance**: Track retrieval accuracy with different strategies
5. **Preserve Structure**: Use appropriate splitters to maintain document hierarchy
