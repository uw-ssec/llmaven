# core/retriever/vector_store.py

import os
from pathlib import Path
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import Qdrant
from qdrant_client import QdrantClient

from core.embeddings.embedding_model import get_embedding_model

# Default configurations.
BASE_DIR = Path(__file__).resolve().parent.parent  # Adjust based on your repo structure.
PDF_FOLDER_PATH = BASE_DIR / "data" / "raw" / "Rubin"
CACHE_VECTOR_STORE_DIR = BASE_DIR / "data" / "vector_stores"  # Visible cache directory.
QDRANT_COLLECTION = "rubin_telescope_exp"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L12-v2"

def load_documents(pdf_folder: Path) -> list:
    """Loads PDFs from the given folder and extracts text."""
    documents = []
    for file in pdf_folder.glob("*.pdf"):
        loader = PyMuPDFLoader(str(file))
        documents.extend(loader.load())
    return documents

def initialize_qdrant(documents, model_name: str = EMBEDDING_MODEL_NAME, 
                       qdrant_path: Path = CACHE_VECTOR_STORE_DIR / "rubin_qdrant_exp", 
                       collection_name: str = QDRANT_COLLECTION) -> Qdrant:
    """Initializes Qdrant with the specified embedding model and documents."""
    embedding = get_embedding_model()
    qdrant_path.mkdir(parents=True, exist_ok=True)
    return Qdrant.from_documents(
        documents=documents,
        embedding=embedding,
        path=str(qdrant_path),
        collection_name=collection_name,
    )

def get_vector_store(config: dict = None):
    """
    Returns a cached vector store. The config dict may include:
       - 'pdf_folder': Path to the PDF folder.
       - 'qdrant_path': Destination path for the vector store.
       - 'collection_name': Qdrant collection name.
       - 'embedding_model': Embedding model name.
    If the vector store does not exist, it is created.
    """
    config = config or {}
    pdf_folder = config.get("pdf_folder", PDF_FOLDER_PATH)
    qdrant_path = config.get("qdrant_path", CACHE_VECTOR_STORE_DIR / "rubin_qdrant_exp")
    collection_name = config.get("collection_name", QDRANT_COLLECTION)
    embedding_model = config.get("embedding_model", EMBEDDING_MODEL_NAME)
    
    documents = load_documents(pdf_folder)
    return initialize_qdrant(documents, model_name=embedding_model, qdrant_path=qdrant_path, collection_name=collection_name)

def create_vector_store_from_documents(documents: list, config: dict = None):
    """
    Creates a vector store from a provided list of document dictionaries.
    
    Each document dict should include at least:
        - "filename": Name of the document.
        - "content": Text content of the document.
    
    Config keys may include:
       - 'qdrant_path': Destination path.
       - 'collection_name': Collection name.
       - 'embedding_model': Embedding model name.
    """
    config = config or {}
    qdrant_path = config.get("qdrant_path", CACHE_VECTOR_STORE_DIR / "temp_vector_store")
    collection_name = config.get("collection_name", "temp_collection")
    embedding_model = config.get("embedding_model", EMBEDDING_MODEL_NAME)
    
    # Convert each dict into a Document object.
    from langchain.docstore.document import Document
    doc_objects = []
    for doc in documents:
        doc_objects.append(Document(page_content=doc["content"], metadata={"source": doc.get("filename", "unknown")}))
    
    return initialize_qdrant(doc_objects, model_name=embedding_model, qdrant_path=qdrant_path, collection_name=collection_name)

if __name__ == "__main__":
    # Quick test to verify vector store initialization.
    vector_store = get_vector_store()
    print(f"Number of vectors stored: {vector_store.client.count(collection_name=QDRANT_COLLECTION)}")
