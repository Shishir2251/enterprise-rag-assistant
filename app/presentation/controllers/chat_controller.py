import inspect

from fastapi import APIRouter, Depends, status

from app.business.dtos.chat_turn_dto import ChatTurnDTO
from app.business.interfaces.chat_service_interface import IChatService
from app.core.exceptions import LLMConfigurationError
from app.data_access.models.user_model import UserModel
from app.presentation.dependencies.auth_dependency import get_current_user
from app.presentation.dependencies.service_dependency import get_chat_service
from app.presentation.schemas.chat_schema import (
    ChatMessageCreateRequest,
    ChatSessionCreateRequest,
    ChatSessionResponse,
    ChatTurnResponse,
    ConversationDetailResponse,
    ConversationHistoryResponse,
    GroundedChatRequest,
    GroundedChatResponse,
)


router = APIRouter(tags=["Chat"])


@router.post(
    "/api/v1/chat/sessions",
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


@router.get(
    "/api/v1/chat/sessions",
    response_model=list[ChatSessionResponse],
)
def list_chat_sessions(
    current_user: UserModel = Depends(get_current_user),
    chat_service: IChatService = Depends(get_chat_service),
):
    return chat_service.list_sessions(owner_id=current_user.id)


@router.get(
    "/api/v1/chat/sessions/{session_id}/messages",
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
    "/api/v1/chat/sessions/{session_id}/messages",
    response_model=ChatTurnResponse,
)
async def send_chat_message(
    session_id: str,
    payload: ChatMessageCreateRequest,
    current_user: UserModel = Depends(get_current_user),
    chat_service: IChatService = Depends(get_chat_service),
):
    result = chat_service.send_message(
        session_id=session_id,
        owner_id=current_user.id,
        message=payload.message,
        top_k=payload.top_k,
        document_ids=_document_ids(payload.document_ids),
    )
    return await result if inspect.isawaitable(result) else result


@router.post(
    "/api/v1/chat",
    response_model=GroundedChatResponse,
)
async def grounded_chat(
    payload: GroundedChatRequest,
    current_user: UserModel = Depends(get_current_user),
    chat_service: IChatService = Depends(get_chat_service),
) -> GroundedChatResponse:
    pending_result = chat_service.send_message(
        session_id=(
            str(payload.conversation_id)
            if payload.conversation_id is not None
            else None
        ),
        owner_id=current_user.id,
        message=payload.message,
        top_k=payload.top_k,
        document_ids=_document_ids(payload.document_ids),
    )
    result: ChatTurnDTO = (
        await pending_result
        if inspect.isawaitable(pending_result)
        else pending_result
    )
    if result.status == "llm_not_configured":
        raise LLMConfigurationError("LLM provider is not configured.")
    return GroundedChatResponse(
        conversation_id=result.session_id,
        message_id=result.assistant_message_id,
        answer=result.answer,
        status=result.status,
        llm_provider=result.llm_provider,
        llm_model=result.llm_model,
        citations=list(result.citations),
    )


@router.get(
    "/api/v1/conversations",
    response_model=list[ChatSessionResponse],
)
def list_conversations(
    current_user: UserModel = Depends(get_current_user),
    chat_service: IChatService = Depends(get_chat_service),
):
    return chat_service.list_sessions(owner_id=current_user.id)


@router.get(
    "/api/v1/conversations/{conversation_id}",
    response_model=ConversationDetailResponse,
)
def get_conversation(
    conversation_id: str,
    current_user: UserModel = Depends(get_current_user),
    chat_service: IChatService = Depends(get_chat_service),
) -> ConversationDetailResponse:
    session = chat_service.get_session(
        session_id=conversation_id,
        owner_id=current_user.id,
    )
    messages = chat_service.get_history(
        session_id=conversation_id,
        owner_id=current_user.id,
    )
    return ConversationDetailResponse(
        id=session.id,
        title=session.title,
        created_at=session.created_at,
        updated_at=session.updated_at,
        messages=messages,
    )


@router.get(
    "/api/v1/conversations/{conversation_id}/messages",
    response_model=ConversationHistoryResponse,
)
def get_conversation_messages(
    conversation_id: str,
    current_user: UserModel = Depends(get_current_user),
    chat_service: IChatService = Depends(get_chat_service),
) -> ConversationHistoryResponse:
    messages = chat_service.get_history(
        session_id=conversation_id,
        owner_id=current_user.id,
    )
    return ConversationHistoryResponse(
        session_id=conversation_id,
        messages=messages,
    )


def _document_ids(document_ids: list) -> list[str] | None:
    return [str(document_id) for document_id in document_ids] or None
