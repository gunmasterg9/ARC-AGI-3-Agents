from __future__ import annotations

from collections import deque
import logging
from typing import Deque, Dict, List, Optional, Set, Tuple
from arcengine import GameAction

from .actions import grid_delta_to_action
from .goals import GoalSpecification
from .objects import ObjectType
from .state import HUDState, PlayerState
from .world_model import WorldModel

logger = logging.getLogger()


class Planner:
    """Hierarchical mission planner and A*/BFS pathfinder."""

    def __init__(self, world_model: WorldModel) -> None:
        self.world_model = world_model
        self.current_plan: Deque[GameAction] = deque()

    def find_grid_path(
        self,
        start: Tuple[int, int],
        target: Tuple[int, int],
        blocked_cells: Optional[Set[Tuple[int, int]]] = None,
    ) -> Optional[List[Tuple[int, int]]]:
        """Find the shortest passable grid path using Breadth-First Search."""
        if start == target:
            return [start]

        obstacles = set(self.world_model.walls)
        if blocked_cells:
            obstacles.update(blocked_cells)

        # Target itself should be accessible even if marked
        obstacles.discard(target)

        queue = deque([(start, [start])])
        visited: Set[Tuple[int, int]] = {start}

        while queue:
            curr, path = queue.popleft()
            if curr == target:
                return path

            for dc, dr in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
                nxt = (curr[0] + dc, curr[1] + dr)
                if (
                    0 <= nxt[0] < self.world_model.cols
                    and 0 <= nxt[1] < self.world_model.rows
                    and nxt not in obstacles
                    and nxt not in visited
                ):
                    visited.add(nxt)
                    queue.append((nxt, path + [nxt]))

        return None

    def path_to_actions(self, path: List[Tuple[int, int]]) -> List[GameAction]:
        """Convert a sequence of adjacent grid waypoints into discrete GameActions."""
        actions: List[GameAction] = []
        for i in range(1, len(path)):
            prev_pos = path[i - 1]
            curr_pos = path[i]
            dc = curr_pos[0] - prev_pos[0]
            dr = curr_pos[1] - prev_pos[1]
            act = grid_delta_to_action(dc, dr)
            if act is not None:
                actions.append(act)
        return actions

    def plan_sequence_to_target(
        self,
        start: Tuple[int, int],
        target: Tuple[int, int],
        count: int = 1,
        blocked_cells: Optional[Set[Tuple[int, int]]] = None,
    ) -> List[GameAction]:
        """Plan path to target, stepping on it `count` times to trigger transformations."""
        actions: List[GameAction] = []
        path = self.find_grid_path(start, target, blocked_cells)
        if not path:
            return actions

        actions.extend(self.path_to_actions(path))

        # If multiple triggers needed, step to free adjacent neighbor and back
        if count > 1:
            # Find free neighbor
            free_neighbor = None
            for dc, dr in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
                nb = (target[0] + dc, target[1] + dr)
                if (
                    0 <= nb[0] < self.world_model.cols
                    and 0 <= nb[1] < self.world_model.rows
                    and nb not in self.world_model.walls
                ):
                    free_neighbor = nb
                    break

            if free_neighbor:
                act_out = grid_delta_to_action(free_neighbor[0] - target[0], free_neighbor[1] - target[1])
                act_back = grid_delta_to_action(target[0] - free_neighbor[0], target[1] - free_neighbor[1])
                if act_out and act_back:
                    for _ in range(count - 1):
                        actions.append(act_out)
                        actions.append(act_back)

        return actions
