from fastapi import APIRouter, Depends

from app.business.interfaces.retrieval_service_interface import (
    IRetrievalService,
)
from app.data_access.models.user_model import UserModel
from app.presentation.dependencies.auth_dependency import get_current_user
from app.presentation.dependencies.service_dependency import (
    get_retrieval_service,
)
from app.presentation.schemas.retrieval_schema import (
    RetrievalSearchRequest,
    RetrievalSearchResponse,
)


router = APIRouter(
    prefix="/api/v1/retrieval",
    tags=["Retrieval"],
)


@router.post("/search", response_model=RetrievalSearchResponse)
def search_documents(
    payload: RetrievalSearchRequest,
    current_user: UserModel = Depends(get_current_user),
    retrieval_service: IRetrievalService = Depends(get_retrieval_service),
) -> RetrievalSearchResponse:
    document_ids = (
        [str(document_id) for document_id in payload.document_ids]
        if payload.document_ids is not None
        else None
    )
    results = retrieval_service.search(
        query=payload.query,
        owner_id=current_user.id,
        top_k=payload.top_k,
        document_ids=document_ids,
    )
    return RetrievalSearchResponse(
        query=payload.query,
        total_results=len(results),
        results=results,
    )
