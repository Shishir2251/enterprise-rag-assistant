from fastapi import APIRouter, Depends, status

from app.business.interfaces.auth_service_interface import IAuthService
from app.presentation.dependencies.service_dependency import get_auth_service
from app.presentation.schemas.auth_schema import (
    AuthResponse,
    LoginRequest,
    RegisterRequest,
    UserResponse,
)

router = APIRouter(prefix="/api/v1/auth", tags=["Auth"])


@router.post(
    "/register",
    response_model=UserResponse,
    status_code=status.HTTP_201_CREATED,
)
def register(
    payload: RegisterRequest,
    auth_service: IAuthService = Depends(get_auth_service),
):
    return auth_service.register(
        full_name=payload.full_name,
        email=str(payload.email),
        password=payload.password,
    )


@router.post("/login", response_model=AuthResponse)
def login(
    payload: LoginRequest,
    auth_service: IAuthService = Depends(get_auth_service),
):
    access_token = auth_service.login(
        email=str(payload.email),
        password=payload.password,
    )
    return AuthResponse(access_token=access_token)
