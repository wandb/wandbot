from wandbot.evaluation.stale_eval_audit import (
    AuditRecord,
    ReplacementPair,
    StalenessAssessment,
    build_summary_markdown,
    extract_search_terms,
    normalize_dataset_rows,
)


def test_normalize_dataset_rows_supports_answer_and_ground_truth():
    rows = [
        {"id": "a1", "question": "How to init wandb?", "answer": "Use wandb.init()", "notes": "old"},
        {"id": "a2", "question": "How to log metrics?", "ground_truth": "Use wandb.log()"},
        {"id": "skip", "answer": "missing question"},
    ]

    normalized = normalize_dataset_rows(rows)
    assert len(normalized) == 2
    assert normalized[0]["item_id"] == "a1"
    assert normalized[0]["reference_answer"] == "Use wandb.init()"
    assert normalized[1]["item_id"] == "a2"
    assert normalized[1]["reference_answer"] == "Use wandb.log()"


def test_extract_search_terms_contains_question_and_keywords():
    question = "How do I use wandb sweeps with Bayesian optimization?"
    reference = "Use wandb.sweep with method='bayes' and run agents with wandb.agent."

    terms = extract_search_terms(question, reference, max_terms=6)

    assert terms[0].startswith("How do I use wandb sweeps")
    assert any("sweep" in term for term in terms)
    assert any("bayes" in term for term in terms)


def test_build_summary_markdown_counts_statuses_and_recommendations():
    fresh_assessment = StalenessAssessment(
        overall_status="fresh",
        question_status="fresh",
        answer_status="correct",
        confidence=0.95,
        reason_summary="Still accurate.",
        stale_signals=[],
        recommended_edits=[],
        needs_new_question_answer_pair=False,
        replacement_pair=None,
        source_links=["https://docs.wandb.ai/"],
        ground_truth_links=["https://docs.wandb.ai/"],
    )
    stale_assessment = StalenessAssessment(
        overall_status="stale",
        question_status="stale",
        answer_status="stale",
        confidence=0.88,
        reason_summary="Deprecated API call.",
        stale_signals=["References removed API."],
        recommended_edits=["Update to new SDK call."],
        needs_new_question_answer_pair=True,
        replacement_pair=ReplacementPair(
            question="How do I track experiments with the current SDK?",
            reference_answer="Use wandb.init and wandb.log with current SDK patterns.",
            rationale="Original pair depended on deprecated behavior.",
            source_links=["https://docs.wandb.ai/guides/track"],
        ),
        source_links=["https://docs.wandb.ai/guides/track"],
        ground_truth_links=["https://docs.wandb.ai/guides/track"],
    )

    records = [
        AuditRecord(
            item_id="1",
            question="q1",
            reference_answer="a1",
            assessment=fresh_assessment,
            created_at_utc="2026-01-01T00:00:00+00:00",
        ),
        AuditRecord(
            item_id="2",
            question="q2",
            reference_answer="a2",
            assessment=stale_assessment,
            created_at_utc="2026-01-01T00:00:00+00:00",
        ),
    ]

    summary = build_summary_markdown(records, dataset="weave:///test", model="gpt-5.2", concurrency=20)

    assert "Fresh: `1`" in summary
    assert "Stale questions: `1`" in summary
    assert "Stale answers: `1`" in summary
    assert "Recommended New Question/Reference Pairs" in summary
    assert "How do I track experiments with the current SDK?" in summary

