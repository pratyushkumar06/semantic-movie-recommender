"""Ground truth definitions for evaluation.

Each query maps to a list of relevant movie IDs (uuid5 of name|year).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List
from uuid import NAMESPACE_DNS, uuid5

from data.movies import movies


@dataclass(frozen=True)
class GroundTruthQuery:
    """Ground truth entry for a single query."""

    use_case: str
    query: str
    relevant_ids: List[str]


def _movie_id(name: str, year: int) -> str:
    return str(uuid5(NAMESPACE_DNS, f"{name}|{year}"))


def _build_title_to_id() -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    for movie in movies:
        name = movie["name"]
        year = movie["year"]
        mapping[name] = _movie_id(name, year)
    return mapping


_TITLE_TO_ID = _build_title_to_id()


def _ids_for_titles(titles: List[str]) -> List[str]:
    ids: List[str] = []
    for title in titles:
        if title not in _TITLE_TO_ID:
            raise ValueError(f"Unknown movie title in ground truth: {title}")
        ids.append(_TITLE_TO_ID[title])
    return ids


GROUND_TRUTH: List[GroundTruthQuery] = [
    GroundTruthQuery(
        use_case="semantic_discovery",
        query="movies about memory and identity",
        relevant_ids=_ids_for_titles(["Blade Runner 2049", "Moon", "Solaris (1972)", "Total Recall"]),
    ),
    GroundTruthQuery(
        use_case="keyword_precision",
        query="movies starring Sigourney Weaver",
        relevant_ids=_ids_for_titles(["Alien", "Aliens", "Avatar"]),
    ),
    GroundTruthQuery(
        use_case="multi_constraint",
        query="post-apocalyptic movies with class conflict released after 2000",
        relevant_ids=_ids_for_titles(["Snowpiercer", "Elysium", "The Platform"]),
    ),
    GroundTruthQuery(
        use_case="semantic_discovery",
        query="philosophical sci-fi about consciousness",
        relevant_ids=_ids_for_titles(["Solaris (1972)", "Ghost in the Shell", "Ex Machina", "2001: A Space Odyssey"]),
    ),
    GroundTruthQuery(
        use_case="keyword_precision",
        query="Denis Villeneuve sci-fi",
        relevant_ids=_ids_for_titles(["Blade Runner 2049", "Arrival"]),
    ),
]


def get_ground_truth() -> List[GroundTruthQuery]:
    """Return all ground truth queries."""

    return list(GROUND_TRUTH)
