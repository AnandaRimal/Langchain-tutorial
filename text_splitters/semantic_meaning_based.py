from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai.embeddings import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

text_splitter = SemanticChunker(
    OpenAIEmbeddings(), breakpoint_threshold_type="standard_deviation",
    breakpoint_threshold_amount=3
)

sample = """
Farmers were working hard in the fields, preparing the soil and planting seeds for the next season. The sun was bright, and the air smelled of earth and fresh grass. The Indian Premier League (IPL) is the biggest cricket league in the world. People all over the world watch the matches and cheer for their favourite teams.


Terrorism is a big danger to peace and safety. It causes harm to people and creates fear in cities and villages. When such attacks happen, they leave behind pain and sadness. To fight terrorism, we need strong laws, alert security forces, and support from people who care about peace and safety.
"""

docs = text_splitter.create_documents([sample])
print(len(docs))
print(docs)



"""🔹 How it Works (High-Level Steps)

Embed the Text

Convert text into vector representations (embeddings) using models like OpenAI’s text-embedding-3-large.

Each sentence, paragraph, or small piece gets its semantic vector.

Compute Similarity

Compare neighboring pieces using cosine similarity or other distance measures.

Idea: If two pieces are highly related, keep them in the same chunk.

If similarity drops, that’s a natural boundary → start a new chunk.

Chunking with Context

Start with a base size (like 500 tokens)

Add pieces one by one, but monitor semantic similarity.

Stop adding when similarity drops too much or chunk size limit is reached.

Optional Overlap

You can overlap pieces to preserve context between chunks."""



""""Paragraph 1: Cricket is a bat-and-ball game played internationally.  
Paragraph 2: It involves batting and bowling strategies.  
Paragraph 3: Football is another popular sport worldwide.

Semantic splitting might produce:

Chunk 1 → Paragraphs 1 + 2 (because both are about cricket, high semantic similarity)

Chunk 2 → Paragraph 3 (football is unrelated, so a new chunk)

Notice: Even if paragraphs are short, semantic splitting keeps related ideas together.
"""