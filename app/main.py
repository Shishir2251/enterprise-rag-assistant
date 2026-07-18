from fastapi import FastAPI

from app.core.config import settings
from app.presentation.controllers.auth_controller import router as auth_router
from app.presentation.controllers.chat_controller import router as chat_router
from app.presentation.controllers.context_controller import (
    router as context_router,
)
from app.presentation.controllers.document_controller import (
    router as document_router,
)
from app.presentation.controllers.retrieval_controller import (
    router as retrieval_router,
)
from app.presentation.controllers.user_controller import router as user_router
from app.presentation.exception_handlers import register_exception_handlers


app = FastAPI(
    title=settings.APP_NAME,
    version="1.0.0",
    debug=settings.APP_DEBUG,
)

register_exception_handlers(app)

app.include_router(auth_router)
app.include_router(user_router)
app.include_router(document_router)
app.include_router(retrieval_router)
app.include_router(context_router)
app.include_router(chat_router)


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "service": settings.APP_NAME,
    }
