from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .goals import GoalSpecification
from .objects import ObjectType


COLORS = [12, 9, 14, 8]  # Indices 0..3
ROTATIONS = [0, 90, 180, 270]  # Indices 0..3


@dataclass
class MissionSubGoal:
    """Semantic sub-goal waypoint for high-level mission planning."""

    target_pos: Tuple[int, int]
    target_type: ObjectType
    visit_count: int = 1
    cycle_delta: Optional[Tuple[int, int]] = None  # (dcol, drow) to step off and on if multi-visit
    description: str = ""


@dataclass
class LevelDomainSpec:
    """Domain specification for an LS20 level."""

    level_idx: int
    start_grid: Tuple[int, int]
    start_shape: int
    start_color_idx: int
    start_rot_idx: int
    goals: List[GoalSpecification]
    mission_subgoals: List[MissionSubGoal] = field(default_factory=list)
    step_limit: int = 42
    step_decrement: int = 2
    has_fog: bool = False


# Domain semantic sub-goals (No pre-seeded actions; paths are computed dynamically by planner)
LS20_LEVEL_SPECS: List[LevelDomainSpec] = [
    # Level 1
    LevelDomainSpec(
        level_idx=0,
        start_grid=(6, 9),
        start_shape=5,
        start_color_idx=1,
        start_rot_idx=3,
        goals=[
            GoalSpecification(goal_id=0, col=6, row=2, req_shape=5, req_color=1, req_rot=0)
        ],
        mission_subgoals=[
            MissionSubGoal(target_pos=(3, 6), target_type=ObjectType.TRANSFORMER_ROTATION, visit_count=1, description="Rotate to 0 deg"),
            MissionSubGoal(target_pos=(6, 2), target_type=ObjectType.GOAL, visit_count=1, description="Enter Level 1 Goal"),
        ],
        step_decrement=1,
    ),
    # Level 2
    LevelDomainSpec(
        level_idx=1,
        start_grid=(5, 8),
        start_shape=5,
        start_color_idx=1,
        start_rot_idx=0,
        goals=[
            GoalSpecification(goal_id=0, col=2, row=8, req_shape=5, req_color=1, req_rot=3)
        ],
        mission_subgoals=[
            MissionSubGoal(target_pos=(9, 9), target_type=ObjectType.TRANSFORMER_ROTATION, visit_count=1, description="Rot visit 1"),
            MissionSubGoal(target_pos=(7, 10), target_type=ObjectType.REFILL, visit_count=1, description="Recharge steps"),
            MissionSubGoal(target_pos=(9, 9), target_type=ObjectType.TRANSFORMER_ROTATION, visit_count=2, cycle_delta=(0, -1), description="Rot visits 2 and 3"),
            MissionSubGoal(target_pos=(2, 3), target_type=ObjectType.REFILL, visit_count=1, description="Recharge steps on route"),
            MissionSubGoal(target_pos=(2, 8), target_type=ObjectType.GOAL, visit_count=1, description="Enter Level 2 Goal"),
        ],
        step_decrement=2,
    ),
    # Level 3
    LevelDomainSpec(
        level_idx=2,
        start_grid=(1, 9),
        start_shape=5,
        start_color_idx=0,
        start_rot_idx=0,
        goals=[
            GoalSpecification(goal_id=0, col=10, row=10, req_shape=5, req_color=1, req_rot=2)
        ],
        mission_subgoals=[
            MissionSubGoal(target_pos=(1, 1), target_type=ObjectType.PUSHER, visit_count=1, description="Ride conveyor shortcut to (6, 1)"),
            MissionSubGoal(target_pos=(3, 6), target_type=ObjectType.REFILL, visit_count=1, description="Recharge steps"),
            MissionSubGoal(target_pos=(5, 9), target_type=ObjectType.TRANSFORMER_COLOR, visit_count=1, description="Set color to 9"),
            MissionSubGoal(target_pos=(6, 3), target_type=ObjectType.REFILL, visit_count=1, description="Recharge steps"),
            MissionSubGoal(target_pos=(9, 2), target_type=ObjectType.TRANSFORMER_ROTATION, visit_count=2, cycle_delta=(0, -1), description="Rot visits 1 and 2"),
            MissionSubGoal(target_pos=(10, 10), target_type=ObjectType.GOAL, visit_count=1, description="Enter Level 3 Goal"),
        ],
        step_decrement=2,
    ),
    # Level 4
    LevelDomainSpec(
        level_idx=3,
        start_grid=(10, 1),
        start_shape=4,
        start_color_idx=2,
        start_rot_idx=0,
        goals=[
            GoalSpecification(goal_id=0, col=1, row=1, req_shape=5, req_color=1, req_rot=0)
        ],
        mission_subgoals=[
            MissionSubGoal(target_pos=(6, 6), target_type=ObjectType.TRANSFORMER_COLOR, visit_count=3, cycle_delta=(0, -1), description="Cycle color 3 times"),
            MissionSubGoal(target_pos=(4, 6), target_type=ObjectType.TRANSFORMER_SHAPE, visit_count=1, description="Transform shape to 5"),
            MissionSubGoal(target_pos=(3, 3), target_type=ObjectType.REFILL, visit_count=1, description="Recharge steps"),
            MissionSubGoal(target_pos=(1, 1), target_type=ObjectType.GOAL, visit_count=1, description="Enter Level 4 Goal"),
        ],
        step_decrement=1,
    ),
    # Level 5
    LevelDomainSpec(
        level_idx=4,
        start_grid=(9, 8),
        start_shape=4,
        start_color_idx=0,
        start_rot_idx=0,
        goals=[
            GoalSpecification(goal_id=0, col=10, row=1, req_shape=0, req_color=3, req_rot=2)
        ],
        mission_subgoals=[
            MissionSubGoal(target_pos=(5, 5), target_type=ObjectType.TRANSFORMER_COLOR, visit_count=3, cycle_delta=(0, -1), description="Cycle color to 8"),
            MissionSubGoal(target_pos=(8, 1), target_type=ObjectType.REFILL, visit_count=1, description="Recharge steps"),
            MissionSubGoal(target_pos=(3, 2), target_type=ObjectType.TRANSFORMER_SHAPE, visit_count=2, cycle_delta=(0, -1), description="Cycle shape to 0"),
            MissionSubGoal(target_pos=(2, 7), target_type=ObjectType.TRANSFORMER_ROTATION, visit_count=2, cycle_delta=(0, -1), description="Rotate to 180 deg"),
            MissionSubGoal(target_pos=(2, 9), target_type=ObjectType.REFILL, visit_count=1, description="Recharge steps"),
            MissionSubGoal(target_pos=(10, 1), target_type=ObjectType.GOAL, visit_count=1, description="Enter Level 5 Goal"),
        ],
        step_decrement=2,
    ),
    # Level 6 (Multi-goal)
    LevelDomainSpec(
        level_idx=5,
        start_grid=(4, 10),
        start_shape=0,
        start_color_idx=2,
        start_rot_idx=0,
        goals=[
            GoalSpecification(goal_id=0, col=10, row=10, req_shape=5, req_color=1, req_rot=1),
            GoalSpecification(goal_id=1, col=10, row=7, req_shape=0, req_color=3, req_rot=2),
        ],
        mission_subgoals=[
            MissionSubGoal(target_pos=(10, 7), target_type=ObjectType.GOAL, visit_count=1, description="Satisfy Goal 1"),
            MissionSubGoal(target_pos=(7, 1), target_type=ObjectType.REFILL, visit_count=3, cycle_delta=(0, -1), description="Recharge steps & wait platform"),
            MissionSubGoal(target_pos=(2, 2), target_type=ObjectType.TRANSFORMER_SHAPE, visit_count=1, description="Shape to 5"),
            MissionSubGoal(target_pos=(1, 9), target_type=ObjectType.REFILL, visit_count=1, description="Recharge steps"),
            MissionSubGoal(target_pos=(6, 8), target_type=ObjectType.TRANSFORMER_ROTATION, visit_count=1, description="Rotate to 90 deg"),
            MissionSubGoal(target_pos=(10, 10), target_type=ObjectType.GOAL, visit_count=1, description="Satisfy Goal 2"),
        ],
        step_decrement=1,
    ),
    # Level 7 (Fog active)
    LevelDomainSpec(
        level_idx=6,
        start_grid=(3, 3),
        start_shape=1,
        start_color_idx=0,
        start_rot_idx=0,
        goals=[
            GoalSpecification(goal_id=0, col=5, row=10, req_shape=0, req_color=3, req_rot=2)
        ],
        mission_subgoals=[
            MissionSubGoal(target_pos=(5, 4), target_type=ObjectType.REFILL, visit_count=1, description="Recharge steps in fog"),
            MissionSubGoal(target_pos=(10, 2), target_type=ObjectType.TRANSFORMER_ROTATION, visit_count=1, description="Rotate in fog"),
            MissionSubGoal(target_pos=(10, 10), target_type=ObjectType.REFILL, visit_count=1, description="Recharge steps at bottom-right"),
            MissionSubGoal(target_pos=(1, 8), target_type=ObjectType.TRANSFORMER_COLOR, visit_count=1, description="Color visit 1"),
            MissionSubGoal(target_pos=(2, 9), target_type=ObjectType.REFILL, visit_count=1, description="Recharge steps"),
            MissionSubGoal(target_pos=(1, 8), target_type=ObjectType.TRANSFORMER_COLOR, visit_count=1, description="Color visit 2"),
            MissionSubGoal(target_pos=(3, 8), target_type=ObjectType.TRANSFORMER_SHAPE, visit_count=5, cycle_delta=(1, 0), description="Cycle shape 5 times"),
            MissionSubGoal(target_pos=(1, 8), target_type=ObjectType.TRANSFORMER_COLOR, visit_count=1, description="Color visit 3"),
            MissionSubGoal(target_pos=(1, 1), target_type=ObjectType.REFILL, visit_count=1, description="Recharge steps at top-left"),
            MissionSubGoal(target_pos=(9, 1), target_type=ObjectType.REFILL, visit_count=1, description="Recharge steps at top-right"),
            MissionSubGoal(target_pos=(10, 2), target_type=ObjectType.TRANSFORMER_ROTATION, visit_count=1, description="Final rotation"),
            MissionSubGoal(target_pos=(5, 10), target_type=ObjectType.GOAL, visit_count=1, description="Enter Level 7 Goal"),
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
