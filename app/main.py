from fastapi import FastAPI
from app.routers import query

app = FastAPI(title="RAG Application")

# Include the query endpoints under an /api prefix
app.include_router(query.router, prefix="/api")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
