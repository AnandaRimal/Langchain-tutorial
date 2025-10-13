from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
import numpy as np

# Load environment variables
load_dotenv()

# Initialize the Gemini embeddings model
model = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

# Your two texts
text1 = """Virat Kohli is one of the greatest modern-day cricketers, widely regarded for his exceptional batting skills and leadership on the field. Born in 1988 in Delhi, India, Kohli made his international debut in 2008 and quickly rose to become a mainstay in the Indian cricket team. Known for his aggressive style, remarkable consistency, and hunger for runs, he has broken numerous records across all formats of the game. Kohli also served as the captain of the Indian team, inspiring a new generation of players with his dedication and fitness regimen."""

text2 = """Jasprit Bumrah, born in 1993 in Ahmedabad, India, is one of the finest fast bowlers of his era. Renowned for his unique bowling action, deadly yorkers, and ability to perform under pressure, Bumrah has been a game-changer in both Tests and limited-overs cricket. Since his debut in 2016, he has consistently led India’s pace attack, often turning matches in India’s favor with his accuracy and tactical brilliance. His calm demeanor off the field contrasts with his fiery, aggressive approach on the pitch."""

# Generate embeddings
vector1 = model.embed_query(text1)  # embed_documents returns a list
vector2 = model.embed_query(text2)  # embed_documents returns a list

# Cosine similarity function
def cosine_similarity(a, b):
    a = np.array(a)
    b = np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# Calculate similarity
similarity_score = cosine_similarity(vector1, vector2)
similarity_percentage = similarity_score * 100

print(f"Cosine Similarity: {similarity_score:.4f}")
print(f"Similarity Percentage: {similarity_percentage:.2f}%")
