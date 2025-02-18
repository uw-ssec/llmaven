from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from app.services.rag_service import RAGService

router = APIRouter()
rag_service = RAGService()

class QueryPayload(BaseModel):
    question: str
    documents: Optional[List[dict]] = None  # Adjust type if you want a specific structure

@router.post("/query")
def query_rag(payload: QueryPayload):
    try:
        documents, response = rag_service.process_query(payload.question, payload.documents)
        return {"documents": documents, "response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
