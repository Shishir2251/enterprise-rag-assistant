from fastapi import APIRouter, Depends

from app.data_access.models.user_model import UserModel
from app.presentation.dependencies.auth_dependency import get_current_user
from app.presentation.schemas.auth_schema import UserResponse


router = APIRouter(prefix="/api/v1/users", tags=["Users"])


@router.get("/me", response_model=UserResponse)
def get_me(
    current_user: UserModel = Depends(get_current_user),
) -> UserModel:
    return current_user
