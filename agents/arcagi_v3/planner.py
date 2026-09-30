from __future__ import annotations

from collections import deque
import heapq
import logging
from typing import Deque, Dict, List, Optional, Set, Tuple
from arcengine import GameAction

from .actions import grid_delta_to_action
from .goals import GoalSpecification
from .objects import ObjectType
from .state import HUDState, PlayerState
from .temporal import TemporalWorldModel
from .world_model import WorldModel

logger = logging.getLogger()


class Planner:
    """Hierarchical mission planner and Space-Time A* / BFS pathfinder."""

    def __init__(self, world_model: WorldModel) -> None:
        self.world_model = world_model
        self.current_plan: Deque[GameAction] = deque()

    def reset_level(self) -> None:
        """Reset planner state for a new level."""
        self.current_plan.clear()

    def get_safe_recovery_action(
        self,
        current_pos: Tuple[int, int],
        recent_positions: Optional[List[Tuple[int, int]]] = None,
    ) -> GameAction:
        """
        Controlled recovery when no global path exists.
        Selects a passable neighboring cell that avoids immediate collisions and recent loops.
        """
        recent = set(recent_positions or [])
        passable_options: List[Tuple[GameAction, bool]] = []
        for act, (dc, dr) in [
            (GameAction.ACTION1, (0, -1)),
            (GameAction.ACTION2, (0, 1)),
            (GameAction.ACTION3, (-1, 0)),
            (GameAction.ACTION4, (1, 0)),
        ]:
            nxt = (current_pos[0] + dc, current_pos[1] + dr)
            if (
                0 <= nxt[0] < self.world_model.cols
                and 0 <= nxt[1] < self.world_model.rows
                and nxt not in self.world_model.walls
                and nxt not in self.world_model.hazard_cells
            ):
                is_fresh = nxt not in recent
                passable_options.append((act, is_fresh))

        # Prefer fresh unvisited neighbor over recently visited
        for act, is_fresh in passable_options:
            if is_fresh:
                return act
        if passable_options:
            return passable_options[0][0]
        return GameAction.ACTION1

    def find_grid_path(
        self,
        start: Tuple[int, int],
        target: Tuple[int, int],
        blocked_cells: Optional[Set[Tuple[int, int]]] = None,
    ) -> Optional[List[Tuple[int, int]]]:
        """Find the shortest passable grid path using Breadth-First Search (static 2D)."""
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

    def find_space_time_path(
        self,
        start: Tuple[int, int],
        target: Tuple[int, int],
        start_t: int = 0,
        temporal_model: Optional[TemporalWorldModel] = None,
        blocked_cells: Optional[Set[Tuple[int, int]]] = None,
        max_horizon: int = 60,
        legal_actions: Optional[List[GameAction]] = None,
    ) -> Optional[Tuple[List[GameAction], List[Tuple[int, int]]]]:
        """
        Find collision-free trajectory using Space-Time A* search over (x, y, t).
        Evaluates dynamic obstacle collisions and edge-swaps at each discrete time step.
        Uses periodic state compression (x, y, t % P) when cyclic period P is known.
        """
        if start == target:
            return ([], [start])

        obstacles = set(self.world_model.walls)
        if blocked_cells:
            obstacles.update(blocked_cells)
        obstacles.discard(target)

        # Available action primitives: strictly legal movement actions unless wait is explicitly provided
        action_deltas: List[Tuple[GameAction, int, int]] = []
        candidate_actions = legal_actions or [
            GameAction.ACTION1,
            GameAction.ACTION2,
            GameAction.ACTION3,
            GameAction.ACTION4,
        ]

        for act in candidate_actions:
            if act == GameAction.ACTION1:
                action_deltas.append((act, 0, -1))
            elif act == GameAction.ACTION2:
                action_deltas.append((act, 0, 1))
            elif act == GameAction.ACTION3:
                action_deltas.append((act, -1, 0))
            elif act == GameAction.ACTION4:
                action_deltas.append((act, 1, 0))
            elif act.name == "ACTION5" or act.name == "ACTION0":
                # Only if wait action exists in environment API
                action_deltas.append((act, 0, 0))

        # Check effective period P for periodic state compression
        effective_p = temporal_model.compute_effective_period() if temporal_model else None

        def heuristic(pos: Tuple[int, int]) -> int:
            return abs(pos[0] - target[0]) + abs(pos[1] - target[1])

        # Priority queue entry: (f_score, g_score, (col, row, t), path_actions, path_coords)
        pq: List[Tuple[int, int, Tuple[int, int, int], List[GameAction], List[Tuple[int, int]]]] = []
        start_f = heuristic(start)
        heapq.heappush(pq, (start_f, 0, (start[0], start[1], start_t), [], [start]))

        # Visited map: state_key -> min g_score
        visited: Dict[Tuple[int, int, int], int] = {}
        initial_key = (start[0], start[1], (start_t % effective_p) if effective_p else start_t)
        visited[initial_key] = 0

        while pq:
            f, g, (c, r, t), actions, coords = heapq.heappop(pq)

            if (c, r) == target:
                return (actions, coords)

            if g >= max_horizon:
                continue

            curr_pos = (c, r)
            nt = t + 1
            ng = g + 1

            for act, dc, dr in action_deltas:
                nc = c + dc
                nr = r + dr
                nxt_pos = (nc, nr)

                # Bounds check
                if not (0 <= nc < self.world_model.cols and 0 <= nr < self.world_model.rows):
                    continue

                # Static obstacle check
                if nxt_pos in obstacles:
                    continue

                # Temporal moving-obstacle collision check
                if temporal_model and temporal_model.is_temporally_blocked(
                    nc, nr, nt, prev_pos=curr_pos, target_cell=target
                ):
                    continue

                state_key = (nc, nr, (nt % effective_p) if effective_p else nt)
                if state_key in visited and visited[state_key] <= ng:
                    continue

                visited[state_key] = ng
                h = heuristic(nxt_pos)
                heapq.heappush(
                    pq,
                    (
                        ng + h,
                        ng,
                        (nc, nr, nt),
                        actions + [act],
                        coords + [nxt_pos],
                    ),
                )

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
        cycle_delta: Optional[Tuple[int, int]] = None,
        blocked_cells: Optional[Set[Tuple[int, int]]] = None,
        temporal_model: Optional[TemporalWorldModel] = None,
        start_t: int = 0,
    ) -> List[GameAction]:
        """
        Plan path to target, stepping on it `count` times to trigger transformations.
        Uses Space-Time A* if moving obstacles are active, otherwise static 2D BFS.
        """
        actions: List[GameAction] = []

        if temporal_model and temporal_model.has_active_moving_objects():
            st_result = self.find_space_time_path(
                start=start,
                target=target,
                start_t=start_t,
                temporal_model=temporal_model,
                blocked_cells=blocked_cells,
            )
            if st_result:
                st_actions, st_coords = st_result
                actions.extend(st_actions)
            else:
                # Fallback to static path if no temporal path found
                path = self.find_grid_path(start, target, blocked_cells)
                if path:
                    actions.extend(self.path_to_actions(path))
        else:
            path = self.find_grid_path(start, target, blocked_cells)
            if not path:
                return actions
            actions.extend(self.path_to_actions(path))

        if not actions and start != target:
            return []

        # If multiple triggers needed, step to neighbor and back
        if count > 1:
            act_out = None
            act_back = None
            if cycle_delta:
                act_out = grid_delta_to_action(cycle_delta[0], cycle_delta[1])
                act_back = grid_delta_to_action(-cycle_delta[0], -cycle_delta[1])
            else:
                # Find free neighbor
                for dc, dr in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
                    nb = (target[0] + dc, target[1] + dr)
                    if (
                        0 <= nb[0] < self.world_model.cols
                        and 0 <= nb[1] < self.world_model.rows
                        and nb not in self.world_model.walls
                    ):
                        act_out = grid_delta_to_action(dc, dr)
                        act_back = grid_delta_to_action(-dc, -dr)
                        break

            if act_out and act_back:
                for _ in range(count - 1):
                    actions.append(act_out)
                    actions.append(act_back)

        return actions
