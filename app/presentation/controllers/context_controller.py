from fastapi import APIRouter, Depends

from app.business.interfaces.context_builder_interface import IContextBuilder
from app.business.interfaces.retrieval_service_interface import (
    IRetrievalService,
)
from app.data_access.models.user_model import UserModel
from app.presentation.dependencies.auth_dependency import get_current_user
from app.presentation.dependencies.service_dependency import (
    get_context_builder_service,
    get_retrieval_service,
)
from app.presentation.schemas.context_schema import (
    ContextBuildRequest,
    ContextBuildResponse,
)


router = APIRouter(
    prefix="/api/v1/context",
    tags=["Context"],
)


@router.post("/build", response_model=ContextBuildResponse)
def build_context(
    payload: ContextBuildRequest,
    current_user: UserModel = Depends(get_current_user),
    retrieval_service: IRetrievalService = Depends(get_retrieval_service),
    context_builder: IContextBuilder = Depends(get_context_builder_service),
) -> ContextBuildResponse:
    document_ids = (
        [str(document_id) for document_id in payload.document_ids]
        if payload.document_ids
        else None
    )
    retrieval_results = retrieval_service.search(
        query=payload.query,
        owner_id=current_user.id,
        top_k=payload.top_k,
        document_ids=document_ids,
    )
    context, sources = context_builder.build_context(retrieval_results)

    return ContextBuildResponse(
        query=payload.query,
        context=context,
        sources=sources,
        llm_status="not_configured",
    )
