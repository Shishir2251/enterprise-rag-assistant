from app.business.interfaces.document_processing_queue_interface import (
    IDocumentProcessingQueue,
)
from app.infrastructure.queue.tasks.document_tasks import (
    process_document_task,
)


class CeleryDocumentProcessingQueue(IDocumentProcessingQueue):

    def enqueue(self, document_id: str) -> str:
        result = process_document_task.apply_async(args=[document_id])
        return str(result.id)

    def enqueue_reindex(self, document_id: str) -> str:
        result = process_document_task.apply_async(
            args=[document_id],
            kwargs={"reindex_embeddings": True},
        )
        return str(result.id)
