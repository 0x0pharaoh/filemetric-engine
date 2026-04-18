"""
filemetric_engine/types.py
--------------------
Shared dataclasses / return types for the public API.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List


@dataclass
class PairResult:
    """Result of a single file-vs-file comparison."""
    file_1: str
    file_2: str
    common_in_percentage: float

    def to_dict(self) -> dict:
        return {
            "file_1": self.file_1,
            "file_2": self.file_2,
            "common_in_percentage": self.common_in_percentage,
        }


@dataclass
class FileMatch:
    """One entry inside a MultiResult comparison list."""
    file: str
    percentage: float

    def to_dict(self) -> dict:
        return {"file": self.file, "percentage": self.percentage}


@dataclass
class MultiResult:
    """Result of comparing one main file against many base files."""
    file: str
    compare: List[FileMatch] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "file": self.file,
            "compare": [m.to_dict() for m in self.compare],
        }

    def top(self, n: int) -> "MultiResult":
        """Return a new MultiResult with only the top-n matches."""
        return MultiResult(
            file=self.file,
            compare=sorted(self.compare, key=lambda x: x.percentage, reverse=True)[:n],
        )
