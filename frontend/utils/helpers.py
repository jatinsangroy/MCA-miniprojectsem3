from collections.abc import Sequence


def calculate_score(answers: Sequence[str], correct_answers: Sequence[str]) -> int:
    """Return the number of answers that match the answer key."""
    return sum(answer == correct for answer, correct in zip(answers, correct_answers))
