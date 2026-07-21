import importlib.util
import unittest
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import Mock, patch

from sqlalchemy.dialects import postgresql

from app.data_access.models.document_chunk_model import DocumentChunkModel
from app.data_access.repositories.document_chunk_repository import (
    DocumentChunkRepository,
)


class EmbeddingStorageTests(unittest.TestCase):
    def test_chunk_vector_uses_local_model_dimension(self) -> None:
        embedding_type = DocumentChunkModel.__table__.c.embedding.type

        self.assertEqual(embedding_type.dim, 384)
        self.assertIn(
            "embedding_provider",
            DocumentChunkModel.__table__.columns,
        )

    def test_stale_chunk_query_checks_vector_model_and_provider(self) -> None:
        session = Mock()
        session.scalars.return_value.all.return_value = []
        repository = DocumentChunkRepository(session)

        repository.list_stale_embeddings(
            "document-id",
            "active-model",
            "local",
        )

        statement = session.scalars.call_args.args[0]
        compiled = statement.compile(dialect=postgresql.dialect())
        sql = str(compiled)
        self.assertIn("document_chunks.embedding IS NULL", sql)
        self.assertIn("document_chunks.embedding_model IS NULL", sql)
        self.assertIn("document_chunks.embedding_model !=", sql)
        self.assertIn("document_chunks.embedding_provider IS NULL", sql)
        self.assertIn("document_chunks.embedding_provider !=", sql)
        self.assertEqual(compiled.params["document_id_1"], "document-id")
        self.assertEqual(compiled.params["embedding_model_1"], "active-model")
        self.assertEqual(compiled.params["embedding_provider_1"], "local")

    def test_stale_document_query_returns_distinct_ids(self) -> None:
        session = Mock()
        session.scalars.return_value.all.return_value = [
            "document-a",
            "document-b",
        ]
        repository = DocumentChunkRepository(session)

        result = repository.list_stale_document_ids(
            "active-model",
            "local",
        )

        statement = session.scalars.call_args.args[0]
        sql = str(statement.compile(dialect=postgresql.dialect()))
        self.assertIn("SELECT DISTINCT document_chunks.document_id", sql)
        self.assertIn("ORDER BY document_chunks.document_id ASC", sql)
        self.assertEqual(result, ["document-a", "document-b"])

    def test_saving_embeddings_persists_active_provider_metadata(self) -> None:
        session = Mock()
        repository = DocumentChunkRepository(session)
        chunk = DocumentChunkModel(
            id="chunk-id",
            document_id="document-id",
            chunk_index=0,
            content="content",
            character_count=7,
            embedding=[0.0] * 384,
        )
        embedded_at = datetime.now(UTC)

        repository.save_embeddings(
            [chunk],
            model_name="active-model",
            provider_name="local",
            embedded_at=embedded_at,
        )

        self.assertEqual(chunk.embedding_model, "active-model")
        self.assertEqual(chunk.embedding_provider, "local")
        self.assertEqual(chunk.embedded_at, embedded_at)
        session.add_all.assert_called_once_with([chunk])
        session.commit.assert_called_once_with()


class LocalEmbeddingMigrationTests(unittest.TestCase):
    def setUp(self) -> None:
        migration_path = (
            Path(__file__).parents[1]
            / "alembic"
            / "versions"
            / "e8f3a1b6c2d4_use_local_embedding_dimension.py"
        )
        spec = importlib.util.spec_from_file_location(
            "local_embedding_dimension_migration",
            migration_path,
        )
        assert spec is not None and spec.loader is not None
        self.migration = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.migration)

    def test_upgrade_clears_metadata_and_changes_vector_dimension(self) -> None:
        with (
            patch.object(self.migration.op, "add_column") as add_column,
            patch.object(self.migration.op, "execute") as execute,
        ):
            self.migration.upgrade()

        added_column = add_column.call_args.args[1]
        sql = " ".join(call.args[0] for call in execute.call_args_list)
        self.assertEqual(added_column.name, "embedding_provider")
        self.assertIn("embedding = NULL", sql)
        self.assertIn("embedding_model = NULL", sql)
        self.assertIn("embedding_provider = NULL", sql)
        self.assertIn("TYPE extensions.vector(384)", sql)

    def test_downgrade_resets_vectors_before_restoring_dimension(self) -> None:
        with (
            patch.object(self.migration.op, "execute") as execute,
            patch.object(self.migration.op, "drop_column") as drop_column,
        ):
            self.migration.downgrade()

        sql = " ".join(call.args[0] for call in execute.call_args_list)
        self.assertIn("embedding = NULL", sql)
        self.assertIn("TYPE extensions.vector(1536)", sql)
        drop_column.assert_called_once_with(
            "document_chunks",
            "embedding_provider",
        )


if __name__ == "__main__":
    unittest.main()
