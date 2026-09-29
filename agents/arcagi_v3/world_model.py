from __future__ import annotations

import logging
from typing import Dict, List, Optional, Set, Tuple
import numpy as np

from .actions import is_valid_grid_pos
from .goals import GoalSpecification
from .objects import GameObject, ObjectType
from .state import PlayerState

logger = logging.getLogger()


class WorldModel:
    """Internal spatial and semantic model of the environment."""

    def __init__(self, cols: int = 11, rows: int = 11) -> None:
        self.cols = cols
        self.rows = rows
        # Grid stores known ObjectType per cell
        self.grid: np.ndarray = np.full((rows, cols), ObjectType.UNKNOWN.value, dtype=object)
        # Sets of known object coordinates
        self.walls: Set[Tuple[int, int]] = set()
        self.goals: Dict[Tuple[int, int], GoalSpecification] = {}
        self.refills: Set[Tuple[int, int]] = set()
        self.transformers: Dict[ObjectType, List[Tuple[int, int]]] = {
            ObjectType.TRANSFORMER_ROTATION: [],
            ObjectType.TRANSFORMER_COLOR: [],
            ObjectType.TRANSFORMER_SHAPE: [],
        }
        self.pushers: Dict[Tuple[int, int], Tuple[int, int]] = {}
        self.visited_cells: Set[Tuple[int, int]] = set()
        self.hazard_cells: Set[Tuple[int, int]] = set()

    def reset_level(self) -> None:
        """Reset internal map for a new level."""
        self.grid.fill(ObjectType.UNKNOWN.value)
        self.walls.clear()
        self.goals.clear()
        self.refills.clear()
        for k in self.transformers:
            self.transformers[k].clear()
        self.pushers.clear()
        self.visited_cells.clear()
        self.hazard_cells.clear()

    def update_from_perception(
        self, objects: List[GameObject], player: Optional[PlayerState]
    ) -> None:
        """Integrate newly perceived objects into world model."""
        if player:
            pos = player.grid_pos
            self.visited_cells.add(pos)
            if self.grid[pos[1], pos[0]] == ObjectType.UNKNOWN.value:
                self.grid[pos[1], pos[0]] = ObjectType.EMPTY.value

        for obj in objects:
            pos = (obj.col, obj.row)
            if not is_valid_grid_pos(obj.col, obj.row):
                continue

            self.grid[obj.row, obj.col] = obj.object_type.value

            if obj.object_type == ObjectType.WALL:
                self.walls.add(pos)
            elif obj.object_type == ObjectType.REFILL:
                self.refills.add(pos)
            elif obj.object_type in self.transformers:
                if pos not in self.transformers[obj.object_type]:
                    self.transformers[obj.object_type].append(pos)

    def mark_obstacle(self, col: int, row: int) -> None:
        """Dynamically mark a cell as an impassable obstacle."""
        if is_valid_grid_pos(col, row):
            self.walls.add((col, row))
            self.grid[row, col] = ObjectType.WALL.value

    def is_passable(
        self,
        col: int,
        row: int,
        player: Optional[PlayerState] = None,
        target_goal: Optional[GoalSpecification] = None,
    ) -> bool:
        """Determine whether a discrete grid cell is passable for the player."""
        if not is_valid_grid_pos(col, row):
            return False

        pos = (col, row)
        if pos in self.walls:
            return False

        if pos in self.hazard_cells:
            return False

        # If cell is a goal, it is only passable if player matches goal condition or targeting it
        if pos in self.goals:
            goal = self.goals[pos]
            if player and target_goal and goal == target_goal:
                return goal.is_player_matching(player)
            # Cannot safely walk through mismatched goal as an intermediate path
            return False

        return True
