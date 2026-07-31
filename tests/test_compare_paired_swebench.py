import pytest

from scripts.analysis.compare_paired_swebench import compare, exact_mcnemar_p


def test_paired_comparison_counts_transitions() -> None:
    report = compare(
        {"a": 0.0, "b": 1.0, "c": 1.0, "d": 0.0},
        {"a": 1.0, "b": 0.0, "c": 1.0, "d": 0.0},
        expected_tasks=4,
        bootstrap_samples=1000,
        seed=7,
    )
    assert report["baseline_rate"] == 0.5
    assert report["post_grpo_rate"] == 0.5
    assert report["post_only_wins"] == 1
    assert report["baseline_only_wins"] == 1
    assert report["both_resolved"] == 1
    assert report["neither_resolved"] == 1


def test_mcnemar_exact_one_sided_outcome_is_doubled() -> None:
    assert exact_mcnemar_p(5, 0) == pytest.approx(0.0625)


def test_task_mismatch_fails_closed() -> None:
    with pytest.raises(ValueError, match="task sets differ"):
        compare(
            {"a": 0.0},
            {"b": 1.0},
            expected_tasks=1,
            bootstrap_samples=10,
            seed=1,
        )
