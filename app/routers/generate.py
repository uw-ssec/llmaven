from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from app.services.generation_service import generate_answer

router = APIRouter()

class GenerationRequest(BaseModel):
    prompt: str
    generation_model: str

@router.post("/generate/")
async def retrieve(request: GenerationRequest):
    try:
        result = generate_answer(request.prompt, request.generation_model)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))