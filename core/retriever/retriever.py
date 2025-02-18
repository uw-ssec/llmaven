from langchain_qdrant import Qdrant
from qdrant_client import QdrantClient
from core.embeddings.embedding_model import get_embedding_model

def initialize_qdrant():
    base_path = Path(__file__).parent.parent.resolve()
    qdrant_path = base_path / "data" / "vector_stores" / "rubin_qdrant_exp"
    qdrant_collection = "rubin_telescope_exp"

    embedding = get_embedding_model()
    client = QdrantClient(path=str(qdrant_path))
    return Qdrant(client=client, collection_name=qdrant_collection, embeddings=embedding)
