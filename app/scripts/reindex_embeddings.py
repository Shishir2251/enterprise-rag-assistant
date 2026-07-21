import argparse
import json
import logging
import sys
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass

from app.business.interfaces.embedding_provider_interface import (
    IEmbeddingProvider,
)
from app.business.interfaces.embedding_service_interface import (
    IEmbeddingService,
)
from app.core.exceptions import EmbeddingError, ValidationError
from app.data_access.interfaces.document_chunk_repository_interface import (
    IDocumentChunkRepository,
)
from app.data_access.interfaces.document_repository_interface import (
    IDocumentRepository,
)
from app.data_access.models.document_model import DocumentStatus


logger = logging.getLogger(__name__)
SAFE_FAILURE_MESSAGE = "Embedding reindex failed"


@dataclass(frozen=True, slots=True)
class BulkReindexDependencies:
    document_repository: IDocumentRepository
    chunk_repository: IDocumentChunkRepository
    embedding_service: IEmbeddingService
    embedding_provider: IEmbeddingProvider
    rollback: Callable[[], None] = lambda: None


@dataclass(frozen=True, slots=True)
class BulkReindexSummary:
    stale_documents_found: int
    documents_attempted: int
    documents_succeeded: int
    documents_failed: int
    documents_skipped: int
    chunks_embedded: int


def _positive_integer(value: str) -> int:
    try:
        parsed_value = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a positive integer") from exc
    if parsed_value <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed_value


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Rebuild embeddings for ready documents whose active embedding "
            "metadata is stale."
        )
    )
    parser.add_argument(
        "--batch-size",
        type=_positive_integer,
        default=None,
        help=(
            "Chunk batch size for this run (default: EMBEDDING_BATCH_SIZE)."
        ),
    )
    parser.add_argument(
        "--limit",
        type=_positive_integer,
        default=None,
        help="Maximum number of eligible stale documents to process.",
    )
    return parser


def reindex_stale_documents(
    dependencies: BulkReindexDependencies,
    *,
    limit: int | None = None,
) -> BulkReindexSummary:
    if limit is not None and limit <= 0:
        raise ValidationError("Reindex limit must be greater than zero")

    provider = dependencies.embedding_provider
    stale_document_ids = list(
        dict.fromkeys(
            dependencies.chunk_repository.list_stale_document_ids(
                model_name=provider.model_name,
                provider_name=provider.provider_name,
            )
        )
    )

    attempted = 0
    succeeded = 0
    failed = 0
    skipped = 0
    chunks_embedded = 0

    for document_id in stale_document_ids:
        if limit is not None and attempted >= limit:
            break

        document = dependencies.document_repository.get_by_id_internal(
            document_id
        )
        if (
            document is None
            or document.status not in DocumentStatus.process_complete_values()
        ):
            skipped += 1
            continue

        attempted += 1
        try:
            logger.info(
                "Embedding reindex started for a document",
                extra={"document_id": document_id, "status": "processing"},
            )
            dependencies.document_repository.mark_processing(document_id)
            dependencies.document_repository.update_progress(
                document_id,
                20,
                "reindexing_embeddings",
            )
            embedded_count = (
                dependencies.embedding_service.reindex_document_chunks(
                    document_id=document_id,
                    owner_id=document.owner_id,
                )
            )
            remaining_stale_chunks = (
                dependencies.chunk_repository.list_stale_embeddings(
                    document_id=document_id,
                    model_name=provider.model_name,
                    provider_name=provider.provider_name,
                )
            )
            if remaining_stale_chunks:
                raise EmbeddingError(
                    "Document chunks contain stale embedding metadata"
                )

            dependencies.document_repository.update_progress(
                document_id,
                90,
                "saving_embeddings",
            )
            dependencies.document_repository.mark_ready(document_id)
            chunks_embedded += embedded_count
            succeeded += 1
            logger.info(
                "Embedding reindex completed for a document",
                extra={
                    "document_id": document_id,
                    "status": DocumentStatus.READY.value,
                    "chunks_embedded": embedded_count,
                },
            )
        except Exception:
            _rollback_safely(dependencies.rollback)
            _mark_failed_safely(
                dependencies.document_repository,
                document_id,
            )
            failed += 1
            logger.error(
                "Embedding reindex failed for a document",
                extra={"document_id": document_id, "status": "failed"},
            )

    return BulkReindexSummary(
        stale_documents_found=len(stale_document_ids),
        documents_attempted=attempted,
        documents_succeeded=succeeded,
        documents_failed=failed,
        documents_skipped=skipped,
        chunks_embedded=chunks_embedded,
    )


def _rollback_safely(rollback: Callable[[], None]) -> None:
    try:
        rollback()
    except Exception:
        logger.error("Unable to roll back a failed embedding reindex")


def _mark_failed_safely(
    document_repository: IDocumentRepository,
    document_id: str,
) -> None:
    try:
        document_repository.mark_failed(
            document_id,
            SAFE_FAILURE_MESSAGE,
        )
    except Exception:
        logger.error(
            "Unable to persist an embedding reindex failure",
            extra={"document_id": document_id, "status": "failed"},
        )


@contextmanager
def _configured_dependencies(
    batch_size: int | None,
) -> Iterator[BulkReindexDependencies]:
    # Infrastructure remains lazy so importing this module never opens a
    # database connection or loads the local sentence-transformer model.
    from app.business.services.embedding_service import EmbeddingService
    from app.core.config import settings
    from app.data_access.repositories.document_chunk_repository import (
        DocumentChunkRepository,
    )
    from app.data_access.repositories.document_repository import (
        DocumentRepository,
    )
    from app.infrastructure.database.session import SessionLocal
    from app.infrastructure.embeddings.embedding_provider_factory import (
        create_embedding_provider,
    )

    db = SessionLocal()
    try:
        document_repository = DocumentRepository(db)
        chunk_repository = DocumentChunkRepository(db)
        # Construct exactly once so a local model is shared for the full run.
        embedding_provider = create_embedding_provider(settings)
        embedding_service = EmbeddingService(
            document_repository=document_repository,
            chunk_repository=chunk_repository,
            embedding_provider=embedding_provider,
            batch_size=batch_size or settings.EMBEDDING_BATCH_SIZE,
        )
        yield BulkReindexDependencies(
            document_repository=document_repository,
            chunk_repository=chunk_repository,
            embedding_service=embedding_service,
            embedding_provider=embedding_provider,
            rollback=db.rollback,
        )
    finally:
        db.close()


def main(
    argv: Sequence[str] | None = None,
    *,
    dependencies: BulkReindexDependencies | None = None,
) -> int:
    args = _build_parser().parse_args(argv)
    try:
        if dependencies is not None:
            summary = reindex_stale_documents(
                dependencies,
                limit=args.limit,
            )
        else:
            with _configured_dependencies(args.batch_size) as configured:
                summary = reindex_stale_documents(
                    configured,
                    limit=args.limit,
                )
    except Exception:
        # Startup failures are deliberately opaque: provider and database
        # exceptions can contain credentials, connection strings, or paths.
        print("Bulk embedding reindex could not start.", file=sys.stderr)
        return 1

    print(json.dumps(asdict(summary), indent=2))
    return 0 if summary.documents_failed == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
