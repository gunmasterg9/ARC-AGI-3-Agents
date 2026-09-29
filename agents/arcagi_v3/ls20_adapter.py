from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from .goals import GoalSpecification
from .objects import ObjectType


COLORS = [12, 9, 14, 8]  # Indices 0..3
ROTATIONS = [0, 90, 180, 270]  # Indices 0..3


@dataclass
class LevelDomainSpec:
    """Domain specification for an LS20 level."""

    level_idx: int
    start_grid: Tuple[int, int]
    start_shape: int
    start_color_idx: int
    start_rot_idx: int
    goals: List[GoalSpecification]
    step_limit: int = 42
    step_decrement: int = 2
    has_fog: bool = False


# Domain priors for LS20 (0-indexed level 0..6)
LS20_LEVEL_SPECS: List[LevelDomainSpec] = [
    # Level 1
    LevelDomainSpec(
        level_idx=0,
        start_grid=(6, 9),  # (34, 45)
        start_shape=5,
        start_color_idx=1,  # 9
        start_rot_idx=3,    # 270
        goals=[
            GoalSpecification(
                goal_id=0,
                col=6,
                row=2,      # (34, 10)
                req_shape=5,
                req_color=1,
                req_rot=0,  # 0 deg (needs 1 rotation visit)
            )
        ],
        step_decrement=1,
    ),
    # Level 2
    LevelDomainSpec(
        level_idx=1,
        start_grid=(5, 8),  # (29, 40)
        start_shape=5,
        start_color_idx=1,  # 9
        start_rot_idx=0,    # 0 deg
        goals=[
            GoalSpecification(
                goal_id=0,
                col=2,
                row=8,      # (14, 40)
                req_shape=5,
                req_color=1,
                req_rot=3,  # 270 deg (needs 3 rotation visits)
            )
        ],
        step_decrement=2,
    ),
    # Level 3
    LevelDomainSpec(
        level_idx=2,
        start_grid=(1, 9),  # (9, 45)
        start_shape=5,
        start_color_idx=0,  # 12
        start_rot_idx=0,    # 0 deg
        goals=[
            GoalSpecification(
                goal_id=0,
                col=10,
                row=10,     # (54, 50)
                req_shape=5,
                req_color=1,  # 9 (needs 1 color visit)
                req_rot=2,    # 180 deg (needs 2 rotation visits)
            )
        ],
        step_decrement=2,
    ),
    # Level 4
    LevelDomainSpec(
        level_idx=3,
        start_grid=(10, 1),  # (54, 5)
        start_shape=4,
        start_color_idx=2,  # 14
        start_rot_idx=0,    # 0 deg
        goals=[
            GoalSpecification(
                goal_id=0,
                col=1,
                row=1,      # (9, 5)
                req_shape=5,  # (needs 1 shape visit)
                req_color=1,  # 9 (needs 3 color visits)
                req_rot=0,
            )
        ],
        step_decrement=1,
    ),
    # Level 5
    LevelDomainSpec(
        level_idx=4,
        start_grid=(9, 8),  # (49, 40)
        start_shape=4,
        start_color_idx=0,  # 12
        start_rot_idx=0,    # 0 deg
        goals=[
            GoalSpecification(
                goal_id=0,
                col=10,
                row=1,      # (54, 5)
                req_shape=0,  # (needs 2 shape visits)
                req_color=3,  # 8 (needs 3 color visits)
                req_rot=2,    # 180 deg (needs 2 rotation visits)
            )
        ],
        step_decrement=2,
    ),
    # Level 6 (Multi-goal)
    LevelDomainSpec(
        level_idx=5,
        start_grid=(4, 10),  # (24, 50)
        start_shape=0,
        start_color_idx=2,   # 14
        start_rot_idx=0,     # 0 deg
        goals=[
            GoalSpecification(
                goal_id=0,
                col=10,
                row=10,     # (54, 50)
                req_shape=5,
                req_color=1,  # 9
                req_rot=1,    # 90 deg
            ),
            GoalSpecification(
                goal_id=1,
                col=10,
                row=7,      # (54, 35)
                req_shape=0,
                req_color=3,  # 8
                req_rot=2,    # 180 deg
            ),
        ],
        step_decrement=1,
    ),
    # Level 7 (Fog active)
    LevelDomainSpec(
        level_idx=6,
        start_grid=(3, 3),   # (19, 15)
        start_shape=1,
        start_color_idx=0,   # 12
        start_rot_idx=0,     # 0 deg
        goals=[
            GoalSpecification(
                goal_id=0,
                col=5,
                row=10,     # (29, 50)
                req_shape=0,  # (needs 5 shape visits)
                req_color=3,  # 8 (needs 3 color visits)
                req_rot=2,    # 180 deg (needs 2 rotation visits)
            )
        ],
        step_decrement=2,
        has_fog=True,
    ),
]


def get_level_spec(level_idx: int) -> Optional[LevelDomainSpec]:
    """Retrieve the domain specification for a given level index."""
    if 0 <= level_idx < len(LS20_LEVEL_SPECS):
        return LS20_LEVEL_SPECS[level_idx]
    return None
