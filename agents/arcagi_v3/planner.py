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
        self.last_planned_coords: List[Tuple[int, int]] = []
        self.last_is_wait_flags: List[bool] = []

    def reset_level(self) -> None:
        """Reset planner state for a new level."""
        self.current_plan.clear()
        self.last_planned_coords.clear()
        self.last_is_wait_flags.clear()

    def resolve_safe_wait_action(
        self,
        col: int,
        row: int,
        carried_offset: Optional[Tuple[int, int]] = None,
        temporal_model: Optional[TemporalWorldModel] = None,
        current_t: int = 0,
    ) -> Optional[GameAction]:
        """
        Determine the safest physical environment action to wait in-place for 1 step:
        - If not carrying: GameAction.ACTION5 safely yields in place without moving.
        - If carrying: ACTION5 would drop the carried item! Instead, execute a zero-displacement
          intentional bump into an adjacent confirmed static wall.
        """
        if carried_offset is None:
            return GameAction.ACTION5

        # Carrying an item: find adjacent static wall
        for dc, dr in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
            wp = (col + dc, row + dr)
            if wp in self.world_model.walls:
                # Must be a static wall, not a transient dynamic obstacle
                if temporal_model and wp in temporal_model.transient_obstacles:
                    continue
                if temporal_model and any(m.current_pos == wp for m in temporal_model.moving_objects.values()):
                    continue
                return grid_delta_to_action(dc, dr)

        # If no static wall adjacent, check if out-of-bounds boundary bump is possible
        for dc, dr in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
            wp = (col + dc, row + dr)
            if not (0 <= wp[0] < self.world_model.cols and 0 <= wp[1] < self.world_model.rows):
                return grid_delta_to_action(dc, dr)

        # If no static wall or boundary adjacent, cannot safely wait while carrying without dropping
        return None

    def get_safe_recovery_action(
        self,
        current_pos: Tuple[int, int],
        recent_positions: Optional[List[Tuple[int, int]]] = None,
        temporal_model: Optional[TemporalWorldModel] = None,
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
                and (temporal_model is None or nxt not in temporal_model.transient_obstacles)
                and (self.world_model.carried_object_id is not None or nxt not in self.world_model.receptacles)
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
        carried_offset: Optional[Tuple[int, int]] = None,
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
                    if carried_offset is not None:
                        it_c = nxt[0] + carried_offset[0]
                        it_r = nxt[1] + carried_offset[1]
                        if not (
                            0 <= it_c < self.world_model.cols
                            and 0 <= it_r < self.world_model.rows
                        ):
                            continue
                        if (it_c, it_r) in obstacles and nxt != target:
                            continue

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
        carried_offset: Optional[Tuple[int, int]] = None,
        allow_wait: bool = True,
        max_wait_horizon: int = 4,
    ) -> Optional[Tuple[List[GameAction], List[Tuple[int, int]]]]:
        """
        Find collision-free trajectory using Space-Time A* search over (x, y, t).
        Evaluates dynamic obstacle collisions and edge-swaps at each discrete time step.
        Supports composite footprint collision avoidance when an item is carried.
        Supports adaptive in-place temporal waiting when routes are temporarily blocked.
        Uses periodic state compression (x, y, t % P) when cyclic period P is known.
        """
        if start == target:
            self.last_planned_coords = [start]
            self.last_is_wait_flags = []
            return ([], [start])

        obstacles = set(self.world_model.walls)
        if blocked_cells:
            obstacles.update(blocked_cells)
        obstacles.discard(target)

        # Available action primitives: movement actions
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

        # Check effective period P for periodic state compression
        effective_p = temporal_model.compute_effective_period() if temporal_model else None

        def heuristic(pos: Tuple[int, int]) -> int:
            return abs(pos[0] - target[0]) + abs(pos[1] - target[1])

        # Priority queue entry: (f_score, g_score, (col, row, t, consecutive_waits), path_actions, path_coords)
        pq: List[Tuple[float, float, Tuple[int, int, int, int], List[GameAction], List[Tuple[int, int]]]] = []
        start_f = float(heuristic(start))
        heapq.heappush(pq, (start_f, 0.0, (start[0], start[1], start_t, 0), [], [start]))

        # Visited map: state_key -> min g_score
        visited: Dict[Tuple[int, int, int], float] = {}
        initial_key = (start[0], start[1], (start_t % effective_p) if effective_p else start_t)
        visited[initial_key] = 0.0

        while pq:
            f, g, (c, r, t, num_waits), actions, coords = heapq.heappop(pq)

            if (c, r) == target:
                self.last_planned_coords = coords
                self.last_is_wait_flags = [coords[i + 1] == coords[i] for i in range(len(actions))]
                return (actions, coords)

            if g >= max_horizon:
                continue

            curr_pos = (c, r)

            # 1. Evaluate directional movement transitions
            for act, dc, dr in action_deltas:
                nc = c + dc
                nr = r + dr
                nxt_pos = (nc, nr)
                nt = t + 1
                ng = g + 1.0

                # Bounds check
                if not (0 <= nc < self.world_model.cols and 0 <= nr < self.world_model.rows):
                    continue

                # Static obstacle check
                if nxt_pos in obstacles:
                    continue

                # Composite footprint check if carrying an item
                if carried_offset is not None:
                    it_c = nc + carried_offset[0]
                    it_r = nr + carried_offset[1]
                    if not (0 <= it_c < self.world_model.cols and 0 <= it_r < self.world_model.rows):
                        continue
                    if (it_c, it_r) in obstacles and nxt_pos != target:
                        continue
                    if temporal_model and temporal_model.is_temporally_blocked(
                        it_c, it_r, nt, target_cell=target
                    ):
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
                        (nc, nr, nt, 0),
                        actions + [act],
                        coords + [nxt_pos],
                    ),
                )

            # 2. Adaptive in-place temporal waiting transition
            if (
                allow_wait
                and temporal_model
                and (temporal_model.has_active_moving_objects() or temporal_model.transient_obstacles)
                and num_waits < max_wait_horizon
            ):
                nt = t + 1
                ng = g + 1.0001  # Wait cost slightly higher than 1 to break ties in favor of immediate movement
                # Check current cell safety at nt
                if not temporal_model.is_temporally_blocked(c, r, nt, prev_pos=curr_pos, target_cell=target):
                    carrier_safe = True
                    if carried_offset is not None:
                        it_c = c + carried_offset[0]
                        it_r = r + carried_offset[1]
                        if (
                            not (0 <= it_c < self.world_model.cols and 0 <= it_r < self.world_model.rows)
                            or (it_c, it_r) in obstacles
                            or temporal_model.is_temporally_blocked(it_c, it_r, nt, target_cell=target)
                        ):
                            carrier_safe = False

                    if carrier_safe:
                        wait_act = self.resolve_safe_wait_action(c, r, carried_offset, temporal_model, current_t=t)
                        if wait_act is not None:
                            state_key = (c, r, (nt % effective_p) if effective_p else nt)
                            if state_key not in visited or visited[state_key] > ng:
                                visited[state_key] = ng
                                h = heuristic(curr_pos)
                                heapq.heappush(
                                    pq,
                                    (
                                        ng + h,
                                        ng,
                                        (c, r, nt, num_waits + 1),
                                        actions + [wait_act],
                                        coords + [curr_pos],
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
        carried_offset: Optional[Tuple[int, int]] = None,
    ) -> List[GameAction]:
        """
        Plan path to target, stepping on it `count` times to trigger transformations.
        Uses Space-Time A* if moving obstacles are active, otherwise static 2D BFS.
        """
        actions: List[GameAction] = []

        if temporal_model and (temporal_model.has_active_moving_objects() or temporal_model.transient_obstacles):
            st_result = self.find_space_time_path(
                start=start,
                target=target,
                start_t=start_t,
                temporal_model=temporal_model,
                blocked_cells=blocked_cells,
                carried_offset=carried_offset,
            )
            if st_result:
                st_actions, st_coords = st_result
                actions.extend(st_actions)
                self.last_planned_coords = st_coords
                self.last_is_wait_flags = [st_coords[i + 1] == st_coords[i] for i in range(len(st_actions))]
            else:
                # Fallback to static path if no temporal path found
                path = self.find_grid_path(start, target, blocked_cells, carried_offset=carried_offset)
                if path:
                    new_acts = self.path_to_actions(path)
                    actions.extend(new_acts)
                    self.last_planned_coords = path
                    self.last_is_wait_flags = [False] * len(new_acts)
        else:
            path = self.find_grid_path(start, target, blocked_cells, carried_offset=carried_offset)
            if not path:
                self.last_is_wait_flags = []
                return actions
            new_acts = self.path_to_actions(path)
            actions.extend(new_acts)
            self.last_planned_coords = path
            self.last_is_wait_flags = [False] * len(new_acts)

        if not actions and start != target:
            self.last_is_wait_flags = []
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

    def plan_composite_interaction(
        self,
        start: Tuple[int, int],
        target: Tuple[int, int],
        interaction_action: GameAction = GameAction.ACTION5,
        required_distance: int = 1,
        temporal_model: Optional[TemporalWorldModel] = None,
        start_t: int = 0,
        blocked_cells: Optional[Set[Tuple[int, int]]] = None,
        carried_offset: Optional[Tuple[int, int]] = None,
        destination_targets: Optional[List[Tuple[int, int]]] = None,
    ) -> List[GameAction]:
        """
        Plan navigation to reach an affordance target at required_distance,
        evaluating approach orientations and carried footprint clearances,
        then execute the interaction action (e.g. ACTION5 for pickup/drop/switch).
        """
        actions: List[GameAction] = []
        if required_distance == 0:
            if start == target:
                return [interaction_action]
            nav_actions = self.plan_sequence_to_target(
                start=start,
                target=target,
                count=1,
                temporal_model=temporal_model,
                start_t=start_t,
                blocked_cells=blocked_cells,
                carried_offset=carried_offset,
            )
            if not nav_actions:
                return []
            actions.extend(nav_actions)
            actions.append(interaction_action)
            self.last_is_wait_flags.append(False)
            return actions

        # required_distance == 1: evaluate candidate approach orientations around target
        candidates = [
            (target[0] + 0, target[1] - 1),
            (target[0] + 0, target[1] + 1),
            (target[0] - 1, target[1] + 0),
            (target[0] + 1, target[1] + 0),
        ]

        # Ensure target cell itself is treated as an obstacle during approach navigation
        approach_blocked = set(blocked_cells or set()) | {target}

        evaluated_candidates = []
        for nb in candidates:
            # 1. Bounds check
            if not (0 <= nb[0] < self.world_model.cols and 0 <= nb[1] < self.world_model.rows):
                continue
            # 2. Obstacle check
            if nb in self.world_model.walls:
                continue
            if approach_blocked and nb in approach_blocked:
                continue

            # 3. Approach route planning
            if start == nb:
                nav = []
            else:
                nav = self.plan_sequence_to_target(
                    start=start,
                    target=nb,
                    count=1,
                    temporal_model=temporal_model,
                    start_t=start_t,
                    blocked_cells=approach_blocked,
                    carried_offset=carried_offset,
                )
                if not nav and start != nb:
                    continue

            # 4. Resulting carried footprint if picked up from nb
            resulting_offset = (target[0] - nb[0], target[1] - nb[1])

            # 5. Route simulation & corridor clearance check
            if destination_targets:
                feasible_route_len = None
                for dest in destination_targets:
                    dest_player_pos = (dest[0] - resulting_offset[0], dest[1] - resulting_offset[1])
                    if not (0 <= dest_player_pos[0] < self.world_model.cols and 0 <= dest_player_pos[1] < self.world_model.rows):
                        continue
                    if dest_player_pos in self.world_model.walls:
                        continue
                    if blocked_cells and dest_player_pos in blocked_cells:
                        continue
                    sim_route = self.find_grid_path(
                        start=nb,
                        target=dest_player_pos,
                        blocked_cells=blocked_cells,
                        carried_offset=resulting_offset,
                    )
                    if sim_route is not None:
                        if feasible_route_len is None or len(sim_route) < feasible_route_len:
                            feasible_route_len = len(sim_route)

                if feasible_route_len is not None:
                    priority = 1  # Verified feasible downstream route
                    downstream_cost = feasible_route_len
                else:
                    # Check local corridor clearance
                    clear_steps = 0
                    for dc, dr in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
                        step_p = (nb[0] + dc, nb[1] + dr)
                        step_it = (step_p[0] + resulting_offset[0], step_p[1] + resulting_offset[1])
                        if (
                            0 <= step_p[0] < self.world_model.cols and 0 <= step_p[1] < self.world_model.rows
                            and step_p not in self.world_model.walls
                            and 0 <= step_it[0] < self.world_model.cols and 0 <= step_it[1] < self.world_model.rows
                            and step_it not in self.world_model.walls
                        ):
                            clear_steps += 1
                    if clear_steps > 0:
                        priority = 2  # Local corridor clearance
                        downstream_cost = 100 - clear_steps
                    else:
                        priority = 3  # Impossible orientation: trapped in corridor
                        downstream_cost = 9999
            else:
                clear_steps = 0
                for dc, dr in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
                    step_p = (nb[0] + dc, nb[1] + dr)
                    step_it = (step_p[0] + resulting_offset[0], step_p[1] + resulting_offset[1])
                    if (
                        0 <= step_p[0] < self.world_model.cols and 0 <= step_p[1] < self.world_model.rows
                        and step_p not in self.world_model.walls
                        and 0 <= step_it[0] < self.world_model.cols and 0 <= step_it[1] < self.world_model.rows
                        and step_it not in self.world_model.walls
                    ):
                        clear_steps += 1
                if clear_steps > 0:
                    priority = 1
                    downstream_cost = -clear_steps
                else:
                    priority = 3
                    downstream_cost = 9999

            evaluated_candidates.append((priority, len(nav), downstream_cost, nb, nav))

        if not evaluated_candidates:
            return []

        # 6. Reject impossible orientations if viable alternatives exist
        viable = [c for c in evaluated_candidates if c[0] < 3]
        pool = viable if viable else evaluated_candidates
        pool.sort(key=lambda c: (c[0], c[1], c[2]))

        _, _, _, best_nb, best_nav = pool[0]

        if best_nav:
            actions.extend(best_nav)
            wait_flags = list(self.last_is_wait_flags)
        else:
            wait_flags = []

        dc = target[0] - best_nb[0]
        dr = target[1] - best_nb[1]
        face_action = grid_delta_to_action(dc, dr)
        if face_action and (not best_nav or best_nav[-1] != face_action):
            actions.append(face_action)
            wait_flags.append(False)
        actions.append(interaction_action)
        wait_flags.append(False)
        self.last_is_wait_flags = wait_flags
        return actions

