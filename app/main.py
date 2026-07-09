from fastapi import FastAPI

from app.infrastructure.database.base import Base
from app.infrastructure.database.session import engine
from app.presentation.controllers.auth_controller import router as auth_router

# Important: import models before create_all
from app.data_access.models.user_model import UserModel


Base.metadata.create_all(bind=engine)

app = FastAPI(title="Enterprise RAG Assistant")

app.include_router(auth_router)


@app.get("/")
def health_check():
    return {"message": "Enterprise RAG Assistant is running"}