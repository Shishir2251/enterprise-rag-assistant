import os
import subprocess
import sys
import textwrap
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_http_mode_imports_app_and_celery_when_sentence_transformers_is_blocked():
    """Exercise startup in a clean interpreter with the native import blocked."""
    startup_probe = textwrap.dedent(
        """
        import builtins
        import sys

        original_import = builtins.__import__

        def block_sentence_transformers(name, *args, **kwargs):
            if name == "sentence_transformers" or name.startswith(
                "sentence_transformers."
            ):
                raise ImportError(
                    "simulated Windows Application Control block"
                )
            return original_import(name, *args, **kwargs)

        builtins.__import__ = block_sentence_transformers

        from app.main import app
        from app.core.config import settings
        from app.infrastructure.embeddings.embedding_provider_factory import (
            create_embedding_provider,
        )
        from app.infrastructure.queue.tasks.document_tasks import (
            process_document_task,
        )

        provider = create_embedding_provider(settings)

        assert app.title == settings.APP_NAME
        assert provider.provider_name == "http"
        assert provider.model_name == (
            "sentence-transformers/all-MiniLM-L6-v2"
        )
        assert provider.dimensions == 384
        assert process_document_task.name == "documents.process_document"
        assert "sentence_transformers" not in sys.modules
        print("http-startup-ok")
        """
    )
    environment = os.environ.copy()
    environment.update(
        {
            "APP_ENV": "development",
            "APP_DEBUG": "false",
            "DATABASE_URL": (
                "postgresql://test-user:test-password@127.0.0.1/test-db"
            ),
            "JWT_SECRET_KEY": "startup-test-secret",
            "EMBEDDING_PROVIDER": "http",
            "EMBEDDING_MODEL": (
                "sentence-transformers/all-MiniLM-L6-v2"
            ),
            "EMBEDDING_DIMENSION": "384",
            "HTTP_EMBEDDING_BASE_URL": "http://127.0.0.1:8090",
            "HTTP_EMBEDDING_TIMEOUT_SECONDS": "30",
            "LLM_PROVIDER": "disabled",
            "OPENAI_API_KEY": "",
        }
    )

    completed = subprocess.run(
        [sys.executable, "-c", startup_probe],
        cwd=PROJECT_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert completed.returncode == 0, (
        "HTTP-mode startup probe failed.\n"
        f"stdout:\n{completed.stdout}\n"
        f"stderr:\n{completed.stderr}"
    )
    assert completed.stdout.strip() == "http-startup-ok"
