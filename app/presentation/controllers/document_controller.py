from fastapi import APIRouter, Depends, File, Response, UploadFile, status

from app.business.interfaces.document_service_interface import IDocumentService
from app.business.interfaces.ingestion_service_interface import (
    IIngestionService,
)
from app.data_access.models.user_model import UserModel
from app.presentation.dependencies.auth_dependency import get_current_user
from app.presentation.dependencies.service_dependency import (
    get_document_service,
    get_ingestion_service,
)
from app.presentation.schemas.document_chunk_schema import (
    DocumentChunkResponse,
)
from app.presentation.schemas.document_schema import (
    DocumentProcessResponse,
    DocumentResponse,
)


router = APIRouter(
    prefix="/api/v1/documents",
    tags=["Documents"],
)


@router.post(
    "/upload",
    response_model=DocumentResponse,
    status_code=status.HTTP_201_CREATED,
)
def upload_document(
    file: UploadFile = File(...),
    current_user: UserModel = Depends(get_current_user),
    service: IDocumentService = Depends(get_document_service),
):
    return service.upload(
        file=file,
        owner_id=current_user.id,
    )


@router.get(
    "",
    response_model=list[DocumentResponse],
)
def list_documents(
    current_user: UserModel = Depends(get_current_user),
    service: IDocumentService = Depends(get_document_service),
):
    return service.list_documents(current_user.id)


@router.get(
    "/{document_id}",
    response_model=DocumentResponse,
)
def get_document(
    document_id: str,
    current_user: UserModel = Depends(get_current_user),
    service: IDocumentService = Depends(get_document_service),
):
    return service.get_document(
        document_id=document_id,
        owner_id=current_user.id,
    )


@router.delete(
    "/{document_id}",
    status_code=status.HTTP_204_NO_CONTENT,
)
def delete_document(
    document_id: str,
    current_user: UserModel = Depends(get_current_user),
    service: IDocumentService = Depends(get_document_service),
):
    service.delete_document(
        document_id=document_id,
        owner_id=current_user.id,
    )

    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.post(
    "/{document_id}/process",
    response_model=DocumentProcessResponse,
)
def process_document(
    document_id: str,
    current_user: UserModel = Depends(get_current_user),
    ingestion_service: IIngestionService = Depends(get_ingestion_service),
):
    return ingestion_service.process_document(
        document_id=document_id,
        owner_id=current_user.id,
    )


@router.get(
    "/{document_id}/chunks",
    response_model=list[DocumentChunkResponse],
)
def list_document_chunks(
    document_id: str,
    current_user: UserModel = Depends(get_current_user),
    ingestion_service: IIngestionService = Depends(get_ingestion_service),
):
    return ingestion_service.list_document_chunks(
        document_id=document_id,
        owner_id=current_user.id,
    )
