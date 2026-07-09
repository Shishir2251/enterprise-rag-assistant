from fastapi import APIRouter, Depends

from sqlalchemy.orm import Session

from app.infrastructure.database.session import get_db
from app.presentation.dependencies.service_dependency import get_auth_service
from app.presentation.schemas.auth_schema import RegisterRequest, LoginRequest, AuthResponse, UserResponse

router = APIRouter(prefix="/api/v1/auth", tags=["Auth"])


@router.post("/register", response_model=UserResponse)
def register(payload: RegisterRequest, db: Session = Depends(get_db)):
    auth_service = get_auth_service(db)
    return auth_service.register(payload)


@router.post("/login", response_model=AuthResponse)
def login(payload: LoginRequest, db: Session = Depends(get_db)):
    auth_service = get_auth_service(db)
    return auth_service.login(payload)