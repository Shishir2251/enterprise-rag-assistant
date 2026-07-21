import logging

from app.core.config import settings
from app.core.exceptions import (
    ConfigurationError,
    DocumentProcessingError,
    EmbeddingError,
    ValidationError,
)
from app.data_access.models.document_model import DocumentStatus
from app.infrastructure.queue.celery_app import celery_app
from app.infrastructure.queue.task_dependencies import (
    get_document_task_dependencies,
)


logger = logging.getLogger(__name__)

PERMANENT_PROCESSING_ERRORS = (
    ConfigurationError,
    DocumentProcessingError,
    ValidationError,
)


@celery_app.task(
    bind=True,
    name="documents.process_document",
    autoretry_for=(),
    max_retries=settings.DOCUMENT_PROCESSING_MAX_RETRIES,
)
def process_document_task(
    self,
    document_id: str,
    reindex_embeddings: bool = False,
) -> dict:
    task_id = str(self.request.id or "")
    logger.info(
        "Document processing task received",
        extra={
            "document_id": document_id,
            "task_id": task_id,
            "status": "received",
            "retry_count": self.request.retries,
        },
    )

    with get_document_task_dependencies() as dependencies:
        repository = dependencies.document_repository
        try:
            document = repository.get_by_id_internal(document_id)
            if document is None:
                logger.error(
                    "Document processing task could not find document",
                    extra={
                        "document_id": document_id,
                        "task_id": task_id,
                        "status": "missing",
                        "retry_count": self.request.retries,
                    },
                )
                return {
                    "document_id": document_id,
                    "status": "failed",
                    "chunks_processed": 0,
                    "chunks_embedded": 0,
                }

            if (
                document.status in DocumentStatus.process_complete_values()
                and (
                    not reindex_embeddings
                    or (task_id and document.task_id == task_id)
                )
            ):
                chunks = dependencies.chunk_repository.list_by_document(
                    document_id
                )
                return {
                    "document_id": document_id,
                    "status": DocumentStatus.READY.value,
                    "chunks_processed": len(chunks),
                    "chunks_embedded": sum(
                        chunk.embedding is not None for chunk in chunks
                    ),
                }

            if (
                reindex_embeddings
                and task_id
                and document.task_id
                and document.task_id != task_id
            ):
                chunks = dependencies.chunk_repository.list_by_document(
                    document_id
                )
                logger.info(
                    "Stale document reindex task ignored",
                    extra=_log_context(
                        document_id,
                        task_id,
                        "superseded",
                        self.request.retries,
                    ),
                )
                return {
                    "document_id": document_id,
                    "status": document.status,
                    "chunks_processed": len(chunks),
                    "chunks_embedded": sum(
                        chunk.embedding is not None for chunk in chunks
                    ),
                }

            if task_id:
                repository.set_task_id(document_id, task_id)

            repository.mark_processing(document_id)
            repository.update_progress(
                document_id,
                20,
                (
                    "reindexing_embeddings"
                    if reindex_embeddings
                    else "extracting_text"
                ),
            )
            logger.info(
                "Document processing started",
                extra=_log_context(
                    document_id,
                    task_id,
                    "processing",
                    self.request.retries,
                ),
            )

            if reindex_embeddings:
                existing_chunks = (
                    dependencies.chunk_repository.list_by_document(document_id)
                )
                if not existing_chunks:
                    raise ValidationError(
                        "No chunks were generated from the document"
                    )
                chunks_processed = len(existing_chunks)
            else:
                def report_progress(progress: int, step: str) -> None:
                    repository.update_progress(document_id, progress, step)
                    if step == "chunking":
                        logger.info(
                            "Document text extraction completed",
                            extra=_log_context(
                                document_id,
                                task_id,
                                step,
                                self.request.retries,
                            ),
                        )

                chunks_processed = (
                    dependencies.ingestion_service.ingest_document(
                        document_id=document_id,
                        owner_id=document.owner_id,
                        progress_callback=report_progress,
                    )
                )
                logger.info(
                    "Document chunks generated",
                    extra=_log_context(
                        document_id,
                        task_id,
                        "saving_chunks",
                        self.request.retries,
                    ),
                )

            repository.update_progress(
                document_id,
                75,
                "generating_embeddings",
            )
            embedding_method = (
                dependencies.embedding_service.reindex_document_chunks
                if reindex_embeddings
                else dependencies.embedding_service.embed_document_chunks
            )
            chunks_embedded = embedding_method(
                document_id=document_id,
                owner_id=document.owner_id,
            )
            chunks = dependencies.chunk_repository.list_by_document(
                document_id
            )
            if not chunks:
                raise ValidationError(
                    "No chunks were generated from the document"
                )
            if any(chunk.embedding is None for chunk in chunks):
                raise EmbeddingError(
                    "Not all document chunks received embeddings"
                )
            active_provider = getattr(
                dependencies.embedding_service,
                "__dict__",
                {},
            ).get("embedding_provider")
            if active_provider is not None and any(
                chunk.embedding_model != active_provider.model_name
                or chunk.embedding_provider != active_provider.provider_name
                for chunk in chunks
            ):
                raise EmbeddingError(
                    "Document chunks contain stale embedding metadata"
                )

            logger.info(
                "Document embeddings generated",
                extra=_log_context(
                    document_id,
                    task_id,
                    "generating_embeddings",
                    self.request.retries,
                ),
            )
            repository.update_progress(
                document_id,
                90,
                "saving_embeddings",
            )
            repository.mark_ready(document_id)
            logger.info(
                "Document processing completed",
                extra=_log_context(
                    document_id,
                    task_id,
                    DocumentStatus.READY.value,
                    self.request.retries,
                ),
            )
            return {
                "document_id": document_id,
                "status": DocumentStatus.READY.value,
                "chunks_processed": chunks_processed,
                "chunks_embedded": chunks_embedded,
            }

        except PERMANENT_PROCESSING_ERRORS as exc:
            dependencies.session.rollback()
            safe_error = _safe_error_message(exc)
            retry_count = self.request.retries + 1
            try:
                failed_document = repository.increment_retry_count(
                    document_id
                )
                retry_count = failed_document.retry_count
                repository.mark_failed(document_id, safe_error)
            except Exception:
                dependencies.session.rollback()
                logger.exception(
                    "Unable to persist permanent processing failure",
                    extra=_log_context(
                        document_id,
                        task_id,
                        DocumentStatus.FAILED.value,
                        retry_count,
                    ),
                )
            logger.exception(
                "Document processing failed permanently",
                extra=_log_context(
                    document_id,
                    task_id,
                    DocumentStatus.FAILED.value,
                    retry_count,
                ),
            )
            return {
                "document_id": document_id,
                "status": DocumentStatus.FAILED.value,
                "chunks_processed": 0,
                "chunks_embedded": 0,
            }

        except Exception as exc:
            dependencies.session.rollback()
            safe_error = _safe_error_message(exc)
            retries_exhausted = (
                self.request.retries
                >= settings.DOCUMENT_PROCESSING_MAX_RETRIES
            )
            retry_count = self.request.retries + 1
            try:
                retried_document = repository.increment_retry_count(
                    document_id
                )
                retry_count = retried_document.retry_count
                if retries_exhausted:
                    repository.mark_failed(document_id, safe_error)
                else:
                    repository.mark_retry_scheduled(
                        document_id=document_id,
                        error_message=safe_error,
                        task_id=task_id or None,
                    )
            except Exception:
                dependencies.session.rollback()
                logger.exception(
                    "Unable to persist processing retry state",
                    extra=_log_context(
                        document_id,
                        task_id,
                        "retry_state_unavailable",
                        retry_count,
                    ),
                )

            if retries_exhausted:
                logger.exception(
                    "Document processing failed after retries",
                    extra=_log_context(
                        document_id,
                        task_id,
                        DocumentStatus.FAILED.value,
                        retry_count,
                    ),
                )
                return {
                    "document_id": document_id,
                    "status": DocumentStatus.FAILED.value,
                    "chunks_processed": 0,
                    "chunks_embedded": 0,
                }

            logger.exception(
                "Document processing retry scheduled",
                extra=_log_context(
                    document_id,
                    task_id,
                    "retry_scheduled",
                    retry_count,
                ),
            )
            raise self.retry(
                exc=exc,
                countdown=settings.DOCUMENT_PROCESSING_RETRY_DELAY_SECONDS,
            )


def _safe_error_message(exc: Exception) -> str:
    if isinstance(exc, PERMANENT_PROCESSING_ERRORS):
        return exc.detail[:1000]
    return "Document processing failed"


def _log_context(
    document_id: str,
    task_id: str,
    status: str,
    retry_count: int,
) -> dict[str, str | int]:
    return {
        "document_id": document_id,
        "task_id": task_id,
        "status": status,
        "retry_count": retry_count,
    }
