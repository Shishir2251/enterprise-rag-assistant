import argparse
import json
import sys
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from app.business.dtos.retrieval_evaluation_dto import (
    RetrievalEvaluationCaseDTO,
    RetrievalEvaluationReportDTO,
)
from app.business.interfaces.retrieval_service_interface import (
    IRetrievalService,
)
from app.business.services.retrieval_evaluation_service import (
    RetrievalEvaluationService,
)
from app.core.exceptions import ApplicationError, ValidationError


DEFAULT_DATASET_PATH = Path("tests/fixtures/retrieval_eval_cases.json")


def load_evaluation_cases(
    dataset_path: Path,
) -> tuple[RetrievalEvaluationCaseDTO, ...]:
    dataset_path = Path(dataset_path)
    try:
        raw_dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValidationError(
            "Retrieval evaluation dataset is not valid JSON"
        ) from exc

    raw_cases: Any
    if isinstance(raw_dataset, list):
        raw_cases = raw_dataset
    elif isinstance(raw_dataset, dict):
        raw_cases = raw_dataset.get("cases")
    else:
        raw_cases = None
    if not isinstance(raw_cases, list) or not raw_cases:
        raise ValidationError(
            "Retrieval evaluation dataset must contain a non-empty cases list"
        )

    cases: list[RetrievalEvaluationCaseDTO] = []
    for index, raw_case in enumerate(raw_cases, start=1):
        if not isinstance(raw_case, dict):
            raise ValidationError(
                f"Retrieval evaluation case {index} must be an object"
            )

        query = raw_case.get("query")
        expected_chunk_index = raw_case.get("expected_chunk_index")
        expected_document_id = raw_case.get("expected_document_id")
        if not isinstance(query, str):
            raise ValidationError(
                f"Retrieval evaluation case {index} has an invalid query"
            )
        if (
            isinstance(expected_chunk_index, bool)
            or not isinstance(expected_chunk_index, int)
        ):
            raise ValidationError(
                f"Retrieval evaluation case {index} has an invalid "
                "expected_chunk_index"
            )
        if (
            expected_document_id is not None
            and not isinstance(expected_document_id, str)
        ):
            raise ValidationError(
                f"Retrieval evaluation case {index} has an invalid "
                "expected_document_id"
            )
        cases.append(
            RetrievalEvaluationCaseDTO(
                query=query,
                expected_chunk_index=expected_chunk_index,
                expected_document_id=expected_document_id,
            )
        )

    return tuple(cases)


def report_payload(
    report: RetrievalEvaluationReportDTO,
) -> dict[str, int | float | str | None]:
    return {
        "Total cases": report.total_cases,
        "Hit@1": report.hit_at_1,
        "Hit@3": report.hit_at_3,
        "Hit@5": report.hit_at_5,
        "MRR": report.mrr,
        "Average relevant score": report.average_relevant_score,
        "Recommendation": _recommendation(report),
    }


def _recommendation(report: RetrievalEvaluationReportDTO) -> str:
    unchanged_notice = " No configuration files were modified."
    if report.average_relevant_score is None:
        return (
            "No relevant chunks were retrieved. Review case labels, "
            "embeddings, and retrieval settings before tuning thresholds."
            + unchanged_notice
        )
    if report.hit_at_5 < 1.0:
        return (
            "Some relevant chunks were missed in the top five. Validate on "
            "more cases before considering a lower retrieval threshold."
            + unchanged_notice
        )
    return (
        "All labelled chunks were retrieved in the top five. Use the "
        "reported relevant scores as evidence for manual threshold review."
        + unchanged_notice
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate owner-scoped retrieval with labelled query/chunk cases."
        )
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET_PATH,
        help=f"Evaluation JSON path (default: {DEFAULT_DATASET_PATH})",
    )
    parser.add_argument(
        "--owner-id",
        required=True,
        help="Owner whose indexed documents are evaluated.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path for the JSON summary.",
    )
    return parser


@contextmanager
def _configured_retrieval_service() -> Iterator[IRetrievalService]:
    # Keep infrastructure construction lazy so importing the evaluator remains
    # side-effect free and unit tests never need a database or network client.
    from app.business.services.retrieval_service import RetrievalService
    from app.core.config import settings
    from app.data_access.repositories.pgvector_repository import (
        PgVectorRepository,
    )
    from app.infrastructure.database.session import SessionLocal
    from app.infrastructure.embeddings.embedding_provider_factory import (
        create_embedding_provider,
    )

    db = SessionLocal()
    try:
        yield RetrievalService(
            vector_repository=PgVectorRepository(db),
            embedding_provider=create_embedding_provider(settings),
            default_top_k=settings.RETRIEVAL_TOP_K_DEFAULT,
            minimum_score=settings.RETRIEVAL_MIN_SCORE,
            maximum_top_k=settings.RETRIEVAL_TOP_K_MAX,
        )
    finally:
        db.close()


def _run(
    args: argparse.Namespace,
    retrieval_service: IRetrievalService,
) -> int:
    cases = load_evaluation_cases(args.dataset)
    report = RetrievalEvaluationService(retrieval_service).evaluate(
        owner_id=args.owner_id,
        cases=cases,
    )
    rendered_report = json.dumps(
        report_payload(report),
        indent=2,
        sort_keys=False,
    )
    if args.output is not None:
        args.output.write_text(rendered_report + "\n", encoding="utf-8")
    print(rendered_report)
    return 0


def main(
    argv: Sequence[str] | None = None,
    *,
    retrieval_service: IRetrievalService | None = None,
) -> int:
    args = _build_parser().parse_args(argv)
    try:
        if retrieval_service is not None:
            return _run(args, retrieval_service)
        with _configured_retrieval_service() as configured_service:
            return _run(args, configured_service)
    except ApplicationError as exc:
        print(f"Retrieval evaluation failed: {exc.detail}", file=sys.stderr)
        return 1
    except OSError as exc:
        print(
            f"Retrieval evaluation failed: {type(exc).__name__}",
            file=sys.stderr,
        )
        return 1
    except Exception as exc:
        # Database/provider exceptions may contain URLs, credentials, or paths.
        print(
            f"Retrieval evaluation failed: {type(exc).__name__}",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
