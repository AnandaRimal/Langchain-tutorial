# Document Loaders

This folder contains examples of various document loaders in LangChain for loading data from different file formats.

## 📖 Overview

Document Loaders are LangChain components that load data from various sources (text files, PDFs, CSVs, directories) into a standardized Document format. They extract content and metadata for further processing.

## 🎯 Concepts Covered

### 1. **Document Structure**
- **page_content**: The actual text content
- **metadata**: Information about the document (source, page number, etc.)

### 2. **Loader Types**
- Text file loaders
- PDF loaders
- CSV loaders
- Directory loaders
- Unstructured data loaders

### 3. **Batch Processing**
- Loading multiple files from directories
- Processing entire document collections
- Pattern matching with glob

## 📁 Files Description

### `documentsloader.ipynb`
- **Purpose**: Comprehensive document loader examples in Jupyter Notebook
- **Loaders Covered**:

#### 1. Text Loader
```python
from langchain_community.document_loaders import TextLoader
loader = TextLoader('cricket.txt', encoding='utf-8')
docs = loader.load()
```
- Loads plain text files
- UTF-8 encoding support
- Extracts content and file metadata

#### 2. PDF Loader
```python
from langchain_community.document_loaders import PyPDFLoader
loader = PyPDFLoader('git.pdf')
docs = loader.load()
```
- Loads PDF documents
- Page-by-page extraction
- Metadata includes page numbers

#### 3. Unstructured PDF Loader
- Advanced PDF processing
- Extracts tables and figures
- Better handling of complex layouts

#### 4. Directory Loader
```python
from langchain_community.document_loaders import DirectoryLoader
loader = DirectoryLoader(
    path="./my_documents/",
    glob="*.txt",
    show_progress=True
)
docs = loader.load()
```
- Bulk loading from directories
- Pattern matching (*.txt, *.pdf, etc.)
- Progress tracking
- Recursive directory scanning

#### 5. CSV Loader
```python
from langchain_community.document_loaders import CSVLoader
loader = CSVLoader(file_path='titanic (1).csv')
docs = loader.load()
```
- Loads CSV/Excel data
- Row-by-row document creation
- Header extraction

### Supporting Files

#### `cricket.txt`
- Sample text file for TextLoader demonstration
- Contains cricket-related content

#### `titanic (1).csv`
- Sample CSV dataset (Titanic data)
- Demonstrates structured data loading

## 🔑 Key Concepts

### Document Object
```python
Document(
    page_content="The actual text content...",
    metadata={"source": "file.txt", "page": 1}
)
```

### Loading Pattern
1. **Initialize Loader**: Specify file/directory path
2. **Load Documents**: Call `.load()` method
3. **Access Content**: Use `.page_content` and `.metadata`

## 💡 Usage Examples

### Single File Loading
```python
from langchain_community.document_loaders import TextLoader

loader = TextLoader('document.txt', encoding='utf-8')
docs = loader.load()

print(f"Number of Documents: {len(docs)}")
print(f"Content: {docs[0].page_content}")
print(f"Metadata: {docs[0].metadata}")
```

### Batch Directory Loading
```python
from langchain_community.document_loaders import DirectoryLoader

loader = DirectoryLoader(
    path="./documents/",
    glob="**/*.txt",  # Recursive pattern
    show_progress=True
)

docs = loader.load()
print(f"Loaded {len(docs)} documents")
```

## 📊 Loader Comparison

| Loader | Input Format | Use Case | Special Features |
|--------|--------------|----------|------------------|
| TextLoader | .txt | Plain text | Simple, fast |
| PyPDFLoader | .pdf | PDF documents | Page extraction |
| UnstructuredPDFLoader | .pdf | Complex PDFs | Tables, figures |
| CSVLoader | .csv | Tabular data | Row-based docs |
| DirectoryLoader | Multiple files | Bulk loading | Pattern matching |

## 🎯 Common Use Cases

1. **Knowledge Base Creation**: Load company documents into vector database
2. **Data Analysis**: Process CSV files for insights
3. **Document Search**: Load PDFs for semantic search
4. **Content Processing**: Batch process text files
5. **Research**: Load academic papers and articles

## 🛠️ Dependencies

- `langchain_community`
- `pypdf` (for PDF loading)
- `unstructured` (for advanced PDF processing)
- `python-dotenv`

## 🔧 Advanced Features

### Custom Metadata
```python
loader = TextLoader('file.txt')
docs = loader.load()
# Add custom metadata
docs[0].metadata['author'] = 'John Doe'
docs[0].metadata['category'] = 'Research'
```

### Lazy Loading
```python
# For very large files
loader = TextLoader('huge_file.txt')
for doc in loader.lazy_load():
    process(doc)  # Process one at a time
```

## 🎓 Learning Outcomes

After reviewing these examples, you should understand:
- How to load documents from various sources
- Document structure (content + metadata)
- Differences between loader types
- Batch processing with DirectoryLoader
- Encoding and format handling
- Integration with downstream LangChain components

## 🚀 Next Steps

After loading documents, you typically:
1. **Split Text**: Use text splitters for chunking
2. **Create Embeddings**: Convert to vectors
3. **Store in Vector DB**: For semantic search
4. **Build RAG Applications**: Question-answering systems

## 📚 Further Reading

- [LangChain Document Loaders](https://python.langchain.com/docs/modules/data_connection/document_loaders/)
- [Unstructured Documentation](https://unstructured-io.github.io/unstructured/)
- [PyPDF Documentation](https://pypdf.readthedocs.io/)
