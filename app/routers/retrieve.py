from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from app.services.retrieval_service import perform_retrieval

router = APIRouter()

class RetrieveRequest(BaseModel):
    documents: Optional[List[Dict[str, Any]]] = []
    query: str
    existing_collection: str
    existing_qdrant_path: str
    embedding_model: str


@router.post("/retrieve/")
async def retrieve(request: RetrieveRequest):
    try:
        result = perform_retrieval(
            request.documents,
            request.query,
            request.existing_collection,
            request.existing_qdrant_path,
            request.embedding_model
        )
        return result
    except Exception as e:
        # You might choose to log the error as well
        raise HTTPException(status_code=500, detail=str(e))