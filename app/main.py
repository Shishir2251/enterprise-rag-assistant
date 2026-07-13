from fastapi import FastAPI

from app.presentation.controllers.auth_controller import router as auth_router
from app.presentation.controllers.document_controller import (
    router as document_router,
)
from app.presentation.controllers.user_controller import router as user_router


app = FastAPI(
    title="Enterprise RAG Assistant",
    version="1.0.0",
)

app.include_router(auth_router)
app.include_router(user_router)
app.include_router(document_router)


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "service": "Enterprise RAG Assistant",
    }