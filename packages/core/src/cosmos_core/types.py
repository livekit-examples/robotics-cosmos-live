from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

# H x W x C image array (uint8)
Frame = NDArray[np.uint8]

# Raw audio bytes
AudioSegment = bytes


@dataclass
class AnalysisResult:
    """Output from a vision model analyzing a set of frames."""

    answer: str
    metadata: dict[str, Any] = field(default_factory=dict)
    timestamp: float = 0.0


@dataclass
class Document:
    """A storable unit for the vector database."""

    content: str
    track_id: str = ""
    kind: str = ""  # e.g. "transcript", "analysis"
    timestamp: float = 0.0
    embedding: list[float] | None = None


@dataclass
class Overlay:
    """Describes an overlay to render on the live output."""

    kind: str  # e.g. "text", "image", "lower_third"
    data: dict[str, Any] = field(default_factory=dict)
