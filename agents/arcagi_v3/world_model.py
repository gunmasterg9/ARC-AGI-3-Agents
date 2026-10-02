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

    def __init__(self, cols: int = 11, rows: int = 11, stride: int = 5, offset_x: int = 4, offset_y: int = 0) -> None:
        self.cols = cols
        self.rows = rows
        self.stride = stride
        self.offset_x = offset_x
        self.offset_y = offset_y
        # Grid stores known ObjectType per cell
        self.grid: np.ndarray = np.full((rows, cols), ObjectType.UNKNOWN.value, dtype=object)
        # Sets of known object coordinates
        self.walls: Set[Tuple[int, int]] = set()
        self.dynamic_obstacles: Set[Tuple[int, int]] = set()
        self.goals: Dict[Tuple[int, int], GoalSpecification] = {}
        self.refills: Set[Tuple[int, int]] = set()
        self.items: Set[Tuple[int, int]] = set()
        self.receptacles: Set[Tuple[int, int]] = set()
        self.switches: Set[Tuple[int, int]] = set()
        self.carried_object_id: Optional[str] = None
        self.transformers: Dict[ObjectType, List[Tuple[int, int]]] = {
            ObjectType.TRANSFORMER_ROTATION: [],
            ObjectType.TRANSFORMER_COLOR: [],
            ObjectType.TRANSFORMER_SHAPE: [],
        }
        self.pushers: Dict[Tuple[int, int], Tuple[int, int]] = {}
        self.visited_cells: Set[Tuple[int, int]] = set()
        self.hazard_cells: Set[Tuple[int, int]] = set()

    def configure_grid(self, cols: int, rows: int, stride: int, offset_x: int = 0, offset_y: int = 0) -> None:
        """Dynamically reconfigure grid bounds and stride when new grid dimensions are inferred."""
        if cols != self.cols or rows != self.rows or stride != self.stride:
            self.cols = cols
            self.rows = rows
            self.stride = stride
            self.offset_x = offset_x
            self.offset_y = offset_y
            self.grid = np.full((rows, cols), ObjectType.UNKNOWN.value, dtype=object)
            logger.debug(f"[WORLD_MODEL] Reconfigured grid to {cols}x{rows}, stride={stride}px")

    def reset_level(self) -> None:
        """Reset internal map completely for a new level (100% isolated)."""
        self.grid.fill(ObjectType.UNKNOWN.value)
        self.walls.clear()
        self.dynamic_obstacles.clear()
        self.goals.clear()
        self.refills.clear()
        self.items.clear()
        self.receptacles.clear()
        self.switches.clear()
        self.carried_object_id = None
        for k in self.transformers:
            self.transformers[k].clear()
        self.pushers.clear()
        self.visited_cells.clear()
        self.hazard_cells.clear()

    def clear_dynamic_obstacles(self) -> None:
        """Remove learned temporary collision obstacles while keeping static walls."""
        for pos in self.dynamic_obstacles:
            self.walls.discard(pos)
            if self.grid[pos[1], pos[0]] == ObjectType.WALL.value:
                self.grid[pos[1], pos[0]] = ObjectType.EMPTY.value
        self.dynamic_obstacles.clear()

    def rebuild_from_perception(
        self, objects: List[GameObject], player: Optional[PlayerState]
    ) -> None:
        """Completely rebuild level-local model from a fresh perception scan."""
        self.reset_level()
        self.update_from_perception(objects, player)

    def update_from_perception(
        self, objects: List[GameObject], player: Optional[PlayerState]
    ) -> None:
        """Integrate newly perceived objects into world model."""
        if player:
            pos = player.grid_pos
            self.visited_cells.add(pos)
            self.walls.discard(pos)
            if 0 <= pos[0] < self.cols and 0 <= pos[1] < self.rows:
                if self.grid[pos[1], pos[0]] == ObjectType.UNKNOWN.value:
                    self.grid[pos[1], pos[0]] = ObjectType.EMPTY.value

        for obj in objects:
            pos = (obj.col, obj.row)
            if not is_valid_grid_pos(obj.col, obj.row, cols=self.cols, rows=self.rows):
                continue

            self.grid[obj.row, obj.col] = obj.object_type.value

            if obj.object_type == ObjectType.WALL:
                self.walls.add(pos)
            elif obj.object_type == ObjectType.REFILL:
                self.refills.add(pos)
            elif obj.object_type == ObjectType.ITEM:
                self.items.add(pos)
            elif obj.object_type == ObjectType.RECEPTACLE:
                self.receptacles.add(pos)
            elif obj.object_type == ObjectType.SWITCH:
                self.switches.add(pos)
            elif obj.object_type in self.transformers:
                if pos not in self.transformers[obj.object_type]:
                    self.transformers[obj.object_type].append(pos)

        if player:
            self.walls.discard(player.grid_pos)

    def mark_obstacle(self, col: int, row: int) -> None:
        """Dynamically mark a cell as an impassable obstacle."""
        if is_valid_grid_pos(col, row, cols=self.cols, rows=self.rows):
            self.walls.add((col, row))
            self.dynamic_obstacles.add((col, row))
            self.grid[row, col] = ObjectType.WALL.value

    def is_passable(
        self,
        col: int,
        row: int,
        player: Optional[PlayerState] = None,
        target_goal: Optional[GoalSpecification] = None,
    ) -> bool:
        """Determine whether a discrete grid cell is passable for the player."""
        if not is_valid_grid_pos(col, row, cols=self.cols, rows=self.rows):
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

