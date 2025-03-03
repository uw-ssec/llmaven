import textwrap

# API Base URL
API_BASE_URL = "http://localhost:8000/api"

# Embedding model used for retrieval
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L12-v2"

# Generation model
GENERATION_MODEL = "allenai/OLMo-2-1124-7B-Instruct"

EXISTING_COLLECTION = "rubin_telescope"
EXISTING_QDRANT_PATH = "data/vector_stores/rubin_qdrant"

# Retrieval Settings
RETRIEVAL_K = 2  # Number of relevant documents to retrieve

# Expand query with synonyms or additional keywords
def expand_query(query: str) -> str:
    """Modify query for better retrieval."""
    if "Rubin" in query:
        query += " LSST Large Synoptic Survey Telescope"
    return query

# Prompt template for generation
def format_prompt(context: str, question: str) -> str:
    """Format the retrieval context into the final prompt."""
    return textwrap.dedent(f"""
    You are an astrophysics expert with a focus on the Rubin telescope project 
    (formerly known as Large Synoptic Survey Telescope - LSST). Please answer the 
    question on astrophysics based on the following context:

    {context}

    Question: {question}
    """)
