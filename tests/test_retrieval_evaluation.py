import io
import json
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import Mock

from app.business.dtos.retrieval_evaluation_dto import (
    RetrievalEvaluationCaseDTO,
)
from app.business.dtos.retrieval_result_dto import RetrievalResult
from app.business.services.retrieval_evaluation_service import (
    RetrievalEvaluationService,
)
from app.core.exceptions import ValidationError
from app.scripts.evaluate_retrieval import (
    DEFAULT_DATASET_PATH,
    load_evaluation_cases,
    main,
)


def make_result(
    *,
    chunk_id: str,
    document_id: str,
    chunk_index: int,
    score: float,
) -> RetrievalResult:
    return RetrievalResult(
        chunk_id=chunk_id,
        document_id=document_id,
        document_name=f"{document_id}.pdf",
        chunk_index=chunk_index,
        content=f"Content for {chunk_id}",
        page_number=chunk_index + 1,
        similarity_score=score,
    )


class FakeRetrievalService:
    def __init__(
        self,
        results_by_query: dict[str, list[RetrievalResult]] | None = None,
    ) -> None:
        self.results_by_query = results_by_query or {}
        self.calls: list[dict] = []

    def search(self, **kwargs) -> list[RetrievalResult]:
        self.calls.append(kwargs)
        return self.results_by_query.get(kwargs["query"], [])


class RetrievalEvaluationServiceTests(unittest.TestCase):
    def test_exact_hit_mrr_and_relevant_score_metrics(self) -> None:
        cases = (
            RetrievalEvaluationCaseDTO("Rank one", 0, "document-a"),
            RetrievalEvaluationCaseDTO("Rank three", 2, "document-b"),
            RetrievalEvaluationCaseDTO("Rank five", 4),
            RetrievalEvaluationCaseDTO("Missing", 7, "document-d"),
        )
        retrieval_service = FakeRetrievalService(
            {
                "Rank one": [
                    make_result(
                        chunk_id="a",
                        document_id="document-a",
                        chunk_index=0,
                        score=0.91,
                    )
                ],
                "Rank three": [
                    # Same chunk index but wrong document must not match.
                    make_result(
                        chunk_id="wrong-document",
                        document_id="document-x",
                        chunk_index=2,
                        score=0.94,
                    ),
                    make_result(
                        chunk_id="noise",
                        document_id="document-b",
                        chunk_index=1,
                        score=0.80,
                    ),
                    make_result(
                        chunk_id="b",
                        document_id="document-b",
                        chunk_index=2,
                        score=0.72,
                    ),
                ],
                "Rank five": [
                    make_result(
                        chunk_id=f"noise-{index}",
                        document_id="any-document",
                        chunk_index=index,
                        score=0.90 - index * 0.05,
                    )
                    for index in range(4)
                ]
                + [
                    make_result(
                        chunk_id="target-five",
                        document_id="document-c",
                        chunk_index=4,
                        score=0.64,
                    )
                ],
            }
        )

        report = RetrievalEvaluationService(retrieval_service).evaluate(
            owner_id="owner-id",
            cases=cases,
        )

        self.assertEqual(report.total_cases, 4)
        self.assertEqual(report.hit_at_1, 0.25)
        self.assertEqual(report.hit_at_3, 0.50)
        self.assertEqual(report.hit_at_5, 0.75)
        self.assertAlmostEqual(
            report.mrr,
            (1.0 + 1.0 / 3.0 + 1.0 / 5.0) / 4.0,
        )
        self.assertAlmostEqual(
            report.average_relevant_score,
            (0.91 + 0.72 + 0.64) / 3.0,
        )
        self.assertEqual(
            [result.relevant_rank for result in report.cases],
            [1, 3, 5, None],
        )
        self.assertTrue(
            all(
                call["owner_id"] == "owner-id"
                and call["top_k"] == 5
                and call["document_ids"] is None
                for call in retrieval_service.calls
            )
        )

    def test_results_beyond_rank_five_do_not_count(self) -> None:
        results = [
            make_result(
                chunk_id=f"noise-{index}",
                document_id="document-a",
                chunk_index=index,
                score=0.90 - index * 0.05,
            )
            for index in range(5)
        ]
        results.append(
            make_result(
                chunk_id="target-six",
                document_id="document-a",
                chunk_index=9,
                score=0.50,
            )
        )
        service = RetrievalEvaluationService(
            FakeRetrievalService({"Question": results})
        )

        report = service.evaluate(
            owner_id="owner-id",
            cases=[
                RetrievalEvaluationCaseDTO(
                    query="Question",
                    expected_chunk_index=9,
                    expected_document_id="document-a",
                )
            ],
        )

        self.assertEqual(report.hit_at_5, 0.0)
        self.assertEqual(report.mrr, 0.0)
        self.assertIsNone(report.average_relevant_score)

    def test_invalid_owner_case_set_and_chunk_index_are_rejected(self) -> None:
        service = RetrievalEvaluationService(FakeRetrievalService())

        with self.assertRaises(ValidationError):
            service.evaluate(owner_id=" ", cases=[])
        with self.assertRaises(ValidationError):
            service.evaluate(owner_id="owner-id", cases=[])
        with self.assertRaises(ValidationError):
            service.evaluate(
                owner_id="owner-id",
                cases=[RetrievalEvaluationCaseDTO("Question", -1)],
            )


class RetrievalEvaluationCLITests(unittest.TestCase):
    def test_synthetic_default_fixture_has_required_case_keys(self) -> None:
        cases = load_evaluation_cases(DEFAULT_DATASET_PATH)

        self.assertEqual(len(cases), 3)
        self.assertTrue(all(case.query for case in cases))
        self.assertTrue(
            all(case.expected_chunk_index >= 0 for case in cases)
        )

    def test_cli_prints_exact_metrics_and_does_not_modify_configuration(
        self,
    ) -> None:
        cases = load_evaluation_cases(DEFAULT_DATASET_PATH)
        retrieval_service = FakeRetrievalService(
            {
                case.query: [
                    make_result(
                        chunk_id=f"target-{index}",
                        document_id=case.expected_document_id or "any-document",
                        chunk_index=case.expected_chunk_index,
                        score=0.80 + index * 0.05,
                    )
                ]
                for index, case in enumerate(cases)
            }
        )
        stdout = io.StringIO()
        stderr = io.StringIO()

        with redirect_stdout(stdout), redirect_stderr(stderr):
            exit_code = main(
                ["--owner-id", "owner-id"],
                retrieval_service=retrieval_service,
            )

        self.assertEqual(exit_code, 0)
        self.assertEqual(stderr.getvalue(), "")
        payload = json.loads(stdout.getvalue())
        self.assertEqual(
            list(payload),
            [
                "Total cases",
                "Hit@1",
                "Hit@3",
                "Hit@5",
                "MRR",
                "Average relevant score",
                "Recommendation",
            ],
        )
        self.assertEqual(payload["Total cases"], 3)
        self.assertEqual(payload["Hit@1"], 1.0)
        self.assertEqual(payload["Hit@3"], 1.0)
        self.assertEqual(payload["Hit@5"], 1.0)
        self.assertEqual(payload["MRR"], 1.0)
        self.assertAlmostEqual(payload["Average relevant score"], 0.85)
        self.assertIn(
            "No configuration files were modified.",
            payload["Recommendation"],
        )

    def test_cli_returns_safe_failure_for_missing_dataset(self) -> None:
        stdout = io.StringIO()
        stderr = io.StringIO()

        with redirect_stdout(stdout), redirect_stderr(stderr):
            exit_code = main(
                [
                    "--owner-id",
                    "owner-id",
                    "--dataset",
                    str(Path("tests/fixtures/does-not-exist.json")),
                ],
                retrieval_service=FakeRetrievalService(),
            )

        self.assertEqual(exit_code, 1)
        self.assertEqual(stdout.getvalue(), "")
        self.assertIn("Retrieval evaluation failed", stderr.getvalue())
        self.assertNotIn("does-not-exist.json", stderr.getvalue())

    def test_cli_sanitizes_unexpected_provider_or_database_errors(self) -> None:
        retrieval_service = Mock()
        retrieval_service.search.side_effect = RuntimeError(
            "postgresql://user:secret@private-host/database"
        )
        stdout = io.StringIO()
        stderr = io.StringIO()

        with redirect_stdout(stdout), redirect_stderr(stderr):
            exit_code = main(
                ["--owner-id", "owner-id"],
                retrieval_service=retrieval_service,
            )

        self.assertEqual(exit_code, 1)
        self.assertEqual(stdout.getvalue(), "")
        self.assertIn("RuntimeError", stderr.getvalue())
        self.assertNotIn("secret", stderr.getvalue())
        self.assertNotIn("private-host", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
