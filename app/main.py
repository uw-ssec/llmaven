from fastapi import FastAPI
from app.routers import retrieve
from app.routers import generate

app = FastAPI(title="RAG Application")

app.include_router(retrieve.router, prefix="/api")
app.include_router(generate.router, prefix="/api")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
