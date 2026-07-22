import importlib.util
import unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock, call, patch

from sqlalchemy.dialects import postgresql

from app.data_access.models.chat_message_model import (
    ChatMessageModel,
    ChatMessageStatus,
)
from app.data_access.repositories.chat_message_repository import (
    ChatMessageRepository,
)
from app.data_access.repositories.chat_session_repository import (
    ChatSessionRepository,
)


NOW = datetime(2026, 7, 22, 12, 0, 0)


class ChatMessageRepositoryTests(unittest.TestCase):
    def test_new_message_has_completed_status_before_database_flush(self) -> None:
        message = ChatMessageModel(
            session_id="session-id",
            role="user",
            content="Question",
        )

        self.assertEqual(message.status, ChatMessageStatus.COMPLETED.value)

    def test_owner_scoped_message_list_joins_chat_session(self) -> None:
        db = Mock()
        db.scalars.return_value.all.return_value = []
        repository = ChatMessageRepository(db)

        repository.list_by_session("session-id", "owner-id")

        statement = db.scalars.call_args.args[0]
        compiled = statement.compile(dialect=postgresql.dialect())
        sql = str(compiled)
        self.assertIn("JOIN chat_sessions", sql)
        self.assertIn("chat_messages.session_id =", sql)
        self.assertIn("chat_sessions.owner_id =", sql)
        self.assertEqual(compiled.params["session_id_1"], "session-id")
        self.assertEqual(compiled.params["owner_id_1"], "owner-id")

    def test_recent_history_is_owner_scoped_filtered_and_chronological(
        self,
    ) -> None:
        newer = Mock(id="newer")
        older = Mock(id="older")
        db = Mock()
        db.scalars.return_value.all.return_value = [newer, older]
        repository = ChatMessageRepository(db)

        messages = repository.list_recent(
            "session-id",
            "owner-id",
            max_messages=2,
        )

        self.assertEqual(messages, [older, newer])
        statement = db.scalars.call_args.args[0]
        compiled = statement.compile(dialect=postgresql.dialect())
        sql = str(compiled)
        self.assertIn("JOIN chat_sessions", sql)
        self.assertIn("chat_sessions.owner_id =", sql)
        self.assertIn("chat_messages.status IN", sql)
        self.assertIn("chat_messages.created_at DESC", sql)
        self.assertEqual(compiled.params["param_1"], 2)
        self.assertIn(
            ChatMessageStatus.COMPLETED.value,
            compiled.params["status_1"],
        )
        self.assertIn(
            ChatMessageStatus.FALLBACK.value,
            compiled.params["status_1"],
        )

    def test_zero_recent_history_does_not_query_database(self) -> None:
        db = Mock()
        repository = ChatMessageRepository(db)

        self.assertEqual(
            repository.list_recent(
                "session-id",
                "owner-id",
                max_messages=0,
            ),
            [],
        )
        db.scalars.assert_not_called()

    def test_negative_recent_history_limit_is_rejected(self) -> None:
        repository = ChatMessageRepository(Mock())

        with self.assertRaisesRegex(ValueError, "max_messages"):
            repository.list_recent(
                "session-id",
                "owner-id",
                max_messages=-1,
            )

    def test_finalize_persists_safe_provider_metadata_and_citations(
        self,
    ) -> None:
        db = Mock()
        repository = ChatMessageRepository(db)
        message = ChatMessageModel(
            id="message-id",
            session_id="session-id",
            role="assistant",
            content="",
            status=ChatMessageStatus.PENDING.value,
            created_at=NOW,
        )
        citations = [{"source_number": 1, "chunk_id": "chunk-id"}]

        result = repository.finalize(
            message,
            content="Grounded answer. [SOURCE 1]",
            status=ChatMessageStatus.COMPLETED,
            citations=citations,
            llm_provider="fake",
            llm_model="fake-grounded-llm-v1",
        )

        self.assertIs(result, message)
        self.assertEqual(message.status, "completed")
        self.assertEqual(message.citations, citations)
        self.assertEqual(message.llm_provider, "fake")
        self.assertEqual(message.llm_model, "fake-grounded-llm-v1")
        db.add.assert_called_once_with(message)
        db.commit.assert_called_once_with()
        db.refresh.assert_called_once_with(message)

    def test_finalize_rejects_failed_status_without_database_write(self) -> None:
        db = Mock()
        repository = ChatMessageRepository(db)
        message = ChatMessageModel(
            session_id="session-id",
            role="assistant",
            content="",
            status=ChatMessageStatus.PENDING.value,
        )

        with self.assertRaisesRegex(ValueError, "completed or fallback"):
            repository.finalize(
                message,
                content="Should not persist",
                status=ChatMessageStatus.FAILED,
            )

        db.add.assert_not_called()
        db.commit.assert_not_called()

    def test_mark_failed_clears_generated_metadata(self) -> None:
        db = Mock()
        repository = ChatMessageRepository(db)
        message = ChatMessageModel(
            session_id="session-id",
            role="assistant",
            content="",
            status=ChatMessageStatus.PENDING.value,
            citations=[{"source_number": 1}],
            llm_provider="fake",
            llm_model="fake-model",
        )

        repository.mark_failed(message)

        self.assertEqual(message.status, ChatMessageStatus.FAILED.value)
        self.assertIsNone(message.citations)
        self.assertIsNone(message.llm_provider)
        self.assertIsNone(message.llm_model)


class ChatSessionRepositoryLifecycleTests(unittest.TestCase):
    def test_title_update_uses_owner_scoped_lookup(self) -> None:
        repository = ChatSessionRepository(Mock())
        session = Mock(title="Old title")
        repository.get_by_id = Mock(return_value=session)
        repository.touch = Mock(return_value=session)

        result = repository.update_title(
            "session-id",
            "owner-id",
            "New safe title",
        )

        self.assertIs(result, session)
        self.assertEqual(session.title, "New safe title")
        repository.get_by_id.assert_called_once_with(
            "session-id",
            "owner-id",
        )
        repository.touch.assert_called_once_with(session)

    def test_title_update_hides_unowned_session(self) -> None:
        repository = ChatSessionRepository(Mock())
        repository.get_by_id = Mock(return_value=None)

        self.assertIsNone(
            repository.update_title(
                "session-id",
                "other-owner",
                "Must not update",
            )
        )


class PhaseNineChatMigrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        migration_path = (
            Path(__file__).resolve().parents[1]
            / "alembic"
            / "versions"
            / "f9a2c4e7d1b3_extend_chat_message_lifecycle.py"
        )
        spec = importlib.util.spec_from_file_location(
            "phase_nine_chat_migration",
            migration_path,
        )
        if spec is None or spec.loader is None:
            raise RuntimeError("Unable to load Phase 9 chat migration")
        cls.migration = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.migration)

    def test_upgrade_adds_only_message_lifecycle_columns(self) -> None:
        with (
            patch.object(self.migration.op, "add_column") as add_column,
            patch.object(self.migration.op, "execute") as execute,
            patch.object(self.migration.op, "alter_column") as alter_column,
            patch.object(
                self.migration.op,
                "create_check_constraint",
            ) as create_check_constraint,
        ):
            self.migration.upgrade()

        self.assertEqual(
            [item.args[1].name for item in add_column.call_args_list],
            ["status", "llm_provider", "llm_model"],
        )
        self.assertTrue(
            all(item.args[0] == "chat_messages" for item in add_column.call_args_list)
        )
        self.assertIn("status = 'completed'", str(execute.call_args.args[0]))
        alter_column.assert_called_once()
        alter_args, alter_kwargs = alter_column.call_args
        self.assertEqual(alter_args, ("chat_messages", "status"))
        self.assertIsInstance(alter_kwargs["existing_type"], self.migration.sa.String)
        self.assertEqual(alter_kwargs["existing_type"].length, 20)
        self.assertFalse(alter_kwargs["nullable"])
        create_check_constraint.assert_called_once_with(
            "ck_chat_messages_status",
            "chat_messages",
            "status IN ('pending', 'completed', 'failed', 'fallback')",
        )

    def test_downgrade_removes_only_message_lifecycle_columns(self) -> None:
        with (
            patch.object(self.migration.op, "drop_constraint") as drop_constraint,
            patch.object(self.migration.op, "drop_column") as drop_column,
        ):
            self.migration.downgrade()

        drop_constraint.assert_called_once_with(
            "ck_chat_messages_status",
            "chat_messages",
            type_="check",
        )
        self.assertEqual(
            drop_column.call_args_list,
            [
                call("chat_messages", "llm_model"),
                call("chat_messages", "llm_provider"),
                call("chat_messages", "status"),
            ],
        )


if __name__ == "__main__":
    unittest.main()
