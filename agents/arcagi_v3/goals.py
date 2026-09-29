from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

from .state import PlayerState


@dataclass
class GoalSpecification:
    """Target objective condition in ARC-AGI-3."""

    goal_id: int
    col: int
    row: int
    req_shape: Optional[int] = None
    req_color: Optional[int] = None
    req_rot: Optional[int] = None
    is_satisfied: bool = False

    @property
    def grid_pos(self) -> Tuple[int, int]:
        return (self.col, self.row)

    def is_player_matching(self, player: PlayerState) -> bool:
        """Check if player attributes match this goal's requirements."""
        return player.matches_properties(
            req_shape=self.req_shape,
            req_color=self.req_color,
            req_rot=self.req_rot,
        )


class GoalManager:
    """Manages multi-goal detection, satisfaction, and sequencing."""

    def __init__(self) -> None:
        self.goals: List[GoalSpecification] = []

    def set_goals(self, goals: List[GoalSpecification]) -> None:
        self.goals = list(goals)

    def get_active_goals(self) -> List[GoalSpecification]:
        """Return all unsatisfied goals."""
        return [g for g in self.goals if not g.is_satisfied]

    def get_current_goal(self) -> Optional[GoalSpecification]:
        """Return the next pending goal to be targeted."""
        active = self.get_active_goals()
        return active[0] if active else None

    def mark_satisfied(self, goal_id: int) -> None:
        for g in self.goals:
            if g.goal_id == goal_id:
                g.is_satisfied = True

    def all_satisfied(self) -> bool:
        return all(g.is_satisfied for g in self.goals)
