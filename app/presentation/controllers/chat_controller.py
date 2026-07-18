from fastapi import APIRouter, Depends, status

from app.business.interfaces.chat_service_interface import IChatService
from app.data_access.models.user_model import UserModel
from app.presentation.dependencies.auth_dependency import get_current_user
from app.presentation.dependencies.service_dependency import get_chat_service
from app.presentation.schemas.chat_schema import (
    ChatMessageCreateRequest,
    ChatSessionCreateRequest,
    ChatSessionResponse,
    ChatTurnResponse,
    ConversationHistoryResponse,
)


router = APIRouter(
    prefix="/api/v1/chat/sessions",
    tags=["Chat"],
)


@router.post(
    "",
    response_model=ChatSessionResponse,
    status_code=status.HTTP_201_CREATED,
)
def create_chat_session(
    payload: ChatSessionCreateRequest,
    current_user: UserModel = Depends(get_current_user),
    chat_service: IChatService = Depends(get_chat_service),
):
    return chat_service.create_session(
        owner_id=current_user.id,
        title=payload.title,
    )


@router.get("", response_model=list[ChatSessionResponse])
def list_chat_sessions(
    current_user: UserModel = Depends(get_current_user),
    chat_service: IChatService = Depends(get_chat_service),
):
    return chat_service.list_sessions(owner_id=current_user.id)


@router.get(
    "/{session_id}/messages",
    response_model=ConversationHistoryResponse,
)
def get_conversation_history(
    session_id: str,
    current_user: UserModel = Depends(get_current_user),
    chat_service: IChatService = Depends(get_chat_service),
) -> ConversationHistoryResponse:
    messages = chat_service.get_history(
        session_id=session_id,
        owner_id=current_user.id,
    )
    return ConversationHistoryResponse(
        session_id=session_id,
        messages=messages,
    )


@router.post(
    "/{session_id}/messages",
    response_model=ChatTurnResponse,
)
def send_chat_message(
    session_id: str,
    payload: ChatMessageCreateRequest,
    current_user: UserModel = Depends(get_current_user),
    chat_service: IChatService = Depends(get_chat_service),
):
    return chat_service.send_message(
        session_id=session_id,
        owner_id=current_user.id,
        message=payload.message,
        top_k=payload.top_k,
        document_ids=(
            [str(document_id) for document_id in payload.document_ids]
            if payload.document_ids
            else None
        ),
    )

