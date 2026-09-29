from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple


class ObjectType(Enum):
    UNKNOWN = "unknown"
    EMPTY = "empty"
    WALL = "wall"
    PLAYER = "player"
    GOAL = "goal"
    TRANSFORMER_ROTATION = "transformer_rotation"
    TRANSFORMER_COLOR = "transformer_color"
    TRANSFORMER_SHAPE = "transformer_shape"
    REFILL = "refill"
    PUSHER = "pusher"
    MOVING_PLATFORM = "moving_platform"


@dataclass
class GameObject:
    """Structured representation of an observable object in ARC-AGI-3."""

    object_type: ObjectType
    x: int  # Screen pixel x
    y: int  # Screen pixel y
    col: int  # Discrete grid col (0..10)
    row: int  # Discrete grid row (0..10)
    width: int = 5
    height: int = 5
    color: Optional[int] = None
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def grid_pos(self) -> Tuple[int, int]:
        return (self.col, self.row)

    @property
    def pixel_pos(self) -> Tuple[int, int]:
        return (self.x, self.y)
