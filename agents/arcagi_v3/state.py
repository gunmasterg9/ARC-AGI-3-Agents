from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from .objects import GameObject


class TransitionPhase(str, Enum):
    """Generic transition state tracking to synchronize level handoffs."""

    NORMAL = "NORMAL"
    GOAL_COMPLETED = "GOAL_COMPLETED"
    WAITING_FOR_NEW_LEVEL_FRAME = "WAITING_FOR_NEW_LEVEL_FRAME"
    VERIFYING_NEW_FRAME = "VERIFYING_NEW_FRAME"
    INITIALIZING_WORLD_MODEL = "INITIALIZING_WORLD_MODEL"


@dataclass
class PlayerState:
    """Internal semantic state of the agent's character."""

    x: int  # Screen pixel X
    y: int  # Screen pixel Y
    col: int  # Discrete grid col (0..10)
    row: int  # Discrete grid row (0..10)
    top_color: int = 12
    bottom_color: int = 9
    shape_idx: int = 0  # 0..5
    color_idx: int = 0  # 0..3
    rot_idx: int = 0  # 0..3 (0: 0 deg, 1: 90 deg, 2: 180 deg, 3: 270 deg)

    @property
    def grid_pos(self) -> Tuple[int, int]:
        return (self.col, self.row)

    @property
    def pixel_pos(self) -> Tuple[int, int]:
        return (self.x, self.y)

    def matches_properties(
        self,
        req_shape: Optional[int] = None,
        req_color: Optional[int] = None,
        req_rot: Optional[int] = None,
    ) -> bool:
        """Check whether current player properties satisfy requirements."""
        if req_shape is not None and self.shape_idx != req_shape:
            return False
        if req_color is not None and self.color_idx != req_color:
            return False
        if req_rot is not None and self.rot_idx != req_rot:
            return False
        return True


@dataclass
class HUDState:
    """Parsed indicators from HUD display."""

    remaining_steps: int = 42
    remaining_lives: int = 3
    has_fog: bool = False
    levels_completed: int = 0
    target_visible: bool = False
    target_shape: Optional[int] = None
    target_color: Optional[int] = None
    target_rotation: Optional[int] = None


@dataclass
class WorldState:
    """Consolidated state representation at a single observation step."""

    level_idx: int
    player: Optional[PlayerState]
    hud: HUDState
    detected_objects: List[GameObject] = field(default_factory=list)
    screen: Optional[np.ndarray] = None
    completed_goals: List[int] = field(default_factory=list)
