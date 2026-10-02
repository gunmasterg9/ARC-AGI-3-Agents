from __future__ import annotations

from dataclasses import dataclass, field
import logging
import math
from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np
from arcengine import GameAction

from .objects import Affordance, GameObject, ObjectType

logger = logging.getLogger()


class DynamicGridTracker:
    """
    Dynamically infers grid movement stride and cell resolution from visual observations.
    Never relies on game-name conditionals; calculates stride via displacement analysis and GCD.
    """

    def __init__(
        self,
        default_stride: int = 5,
        stride: Optional[int] = None,
        offset_x: Optional[int] = None,
        offset_y: Optional[int] = None,
        cols: Optional[int] = None,
        rows: Optional[int] = None,
    ) -> None:
        actual_stride = stride if stride is not None else default_stride
        self.estimated_stride: int = actual_stride
        self.offset_x: int = offset_x if offset_x is not None else (4 if actual_stride == 5 else 0)
        self.offset_y: int = offset_y if offset_y is not None else 0
        self.cols: int = cols if cols is not None else (11 if actual_stride == 5 else 16)
        self.rows: int = rows if rows is not None else (11 if actual_stride == 5 else 16)
        self.confidence: float = 0.5
        self.observed_displacements: List[int] = []

    @property
    def stride_x(self) -> int:
        return self.estimated_stride

    @property
    def stride_y(self) -> int:
        return self.estimated_stride

    def calibrate_from_screen(
        self, screen: Optional[np.ndarray], is_ls20_hint: bool = False
    ) -> Tuple[int, int, int, int]:
        """
        Calibrate grid stride and offsets directly from frame geometry and contrast boundaries.
        Returns: (stride_x, stride_y, offset_x, offset_y).
        """
        if screen is None or screen.ndim != 2:
            return (self.stride_x, self.stride_y, self.offset_x, self.offset_y)

        if is_ls20_hint:
            self.estimated_stride = 5
            self.offset_x = 4
            self.offset_y = 0
            self.cols = 11
            self.rows = 11
            self.confidence = 1.0
            return (5, 5, 4, 0)

        # 1. Identify background color (dominant)
        vals, counts = np.unique(screen, return_counts=True)
        bg = vals[np.argmax(counts)]

        # 2. Extract bounding boxes of connected components
        H, W = screen.shape
        visited = set()
        boxes = []
        for r in range(H):
            for c in range(W):
                if screen[r, c] != bg and (r, c) not in visited:
                    q = [(r, c)]
                    visited.add((r, c))
                    min_r, max_r, min_c, max_c = r, r, c, c
                    while q:
                        cr, cc = q.pop()
                        min_r, max_r = min(min_r, cr), max(max_r, cr)
                        min_c, max_c = min(min_c, cc), max(max_c, cc)
                        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                            nr, nc = cr + dr, cc + dc
                            if 0 <= nr < H and 0 <= nc < W:
                                if screen[nr, nc] != bg and (nr, nc) not in visited:
                                    visited.add((nr, nc))
                                    q.append((nr, nc))
                    w = max_c - min_c + 1
                    h = max_r - min_r + 1
                    # Filter out full-screen borders or bottom HUD bars
                    if w >= 60 or h >= 60 or (min_r >= 60 and h <= 2):
                        continue
                    boxes.append((min_c, min_r, w, h))

        if not boxes:
            return (self.stride_x, self.stride_y, self.offset_x, self.offset_y)

        # 3. Analyze candidate dimensions in range [2, 16]
        dim_candidates = []
        for x, y, w, h in boxes:
            if 2 <= w <= 16:
                dim_candidates.append(w)
            if 2 <= h <= 16:
                dim_candidates.append(h)

        if dim_candidates:
            stride = dim_candidates[0]
            for d in dim_candidates[1:]:
                stride = math.gcd(stride, d)
        else:
            stride = 4

        if stride < 2 or stride > 8:
            stride = 4

        # 4. Infer offsets using grid-aligned bounding boxes where available
        aligned_boxes = [b for b in boxes if b[2] % stride == 0 and b[3] % stride == 0]
        eval_boxes = aligned_boxes if len(aligned_boxes) >= 2 else boxes

        best_ox = 0
        best_cost_x = float("inf")
        for ox in range(stride):
            cost = sum(min((x - ox) % stride, stride - (x - ox) % stride) for x, y, w, h in eval_boxes)
            if cost < best_cost_x:
                best_cost_x = cost
                best_ox = ox

        best_oy = 0
        best_cost_y = float("inf")
        for oy in range(stride):
            cost = sum(min((y - oy) % stride, stride - (y - oy) % stride) for x, y, w, h in eval_boxes)
            if cost < best_cost_y:
                best_cost_y = cost
                best_oy = oy

        self.estimated_stride = stride
        self.offset_x = best_ox
        self.offset_y = best_oy
        self.cols = (W - self.offset_x) // stride
        self.rows = (H - self.offset_y) // stride
        self.confidence = 0.95

        return (self.stride_x, self.stride_y, self.offset_x, self.offset_y)

    def record_displacement(self, dx: int, dy: int) -> None:
        """Record observed entity motion delta and update estimated stride."""
        mag = max(abs(dx), abs(dy))
        if 2 <= mag <= 16:
            self.observed_displacements.append(mag)
            self._update_stride_estimate()

    def _update_stride_estimate(self) -> None:
        """Estimate grid stride using the greatest common divisor of observed movements."""
        if not self.observed_displacements:
            return

        # Compute GCD of observed non-zero displacements
        current_gcd = self.observed_displacements[0]
        for d in self.observed_displacements[1:]:
            current_gcd = math.gcd(current_gcd, d)

        if 2 <= current_gcd <= 8:
            self.estimated_stride = current_gcd
            # Stride of 5 corresponds to LS20's 4px left-padded 11x11 playfield
            if current_gcd == 5:
                self.offset_x = 4
                self.offset_y = 0
                self.cols = 11
                self.rows = 11
            else:
                self.offset_x = 0
                self.offset_y = 0
                self.cols = 64 // current_gcd
                self.rows = 64 // current_gcd

            n = len(self.observed_displacements)
            self.confidence = min(1.0, 0.4 + 0.15 * n)
            logger.debug(
                f"[STRIDE] Inferred grid stride={self.estimated_stride}px, cols={self.cols}, rows={self.rows} (conf={self.confidence:.2f})"
            )

    def pixel_to_grid(self, x: int, y: int) -> Tuple[int, int]:
        """Convert screen pixel coordinates to discrete grid coordinates."""
        col = max(0, min(self.cols - 1, (x - self.offset_x) // self.stride_x))
        row = max(0, min(self.rows - 1, (y - self.offset_y) // self.stride_y))
        return (col, row)

    def grid_to_pixel(self, col: int, row: int) -> Tuple[int, int]:
        """Convert discrete grid coordinates to screen pixel coordinates."""
        x = self.offset_x + col * self.stride_x
        y = self.offset_y + row * self.stride_y
        return (x, y)


class ControllableEntityManager:
    """
    Identifies and tracks which visible entity responds to player control.
    Supports single-avatar games and multi-entity controllable environments (e.g. piece selection).
    """

    def __init__(self) -> None:
        self.controlled_entity_id: Optional[str] = None
        self.controlled_pos: Optional[Tuple[int, int]] = None
        self.controlled_pixel_pos: Optional[Tuple[int, int]] = None
        self.candidate_entities: Dict[str, Tuple[int, int]] = {}
        self.control_confidence: float = 0.0

    def reset_level(self) -> None:
        """Reset entity tracking between levels."""
        self.controlled_entity_id = None
        self.controlled_pos = None
        self.controlled_pixel_pos = None
        self.candidate_entities.clear()
        self.control_confidence = 0.0

    def discover_controlled_entity(
        self,
        prev_screen: Optional[np.ndarray],
        curr_screen: Optional[np.ndarray],
        action: GameAction,
        grid_tracker: DynamicGridTracker,
    ) -> Optional[Tuple[int, int]]:
        """
        Detect which entity changed position in the direction of the executed action.
        """
        if prev_screen is None or curr_screen is None:
            return None

        # Directional actions only
        if action not in (
            GameAction.ACTION1,
            GameAction.ACTION2,
            GameAction.ACTION3,
            GameAction.ACTION4,
        ):
            return self.controlled_pos

        diff = (prev_screen != curr_screen)
        if not np.any(diff):
            return self.controlled_pos

        from .actions import action_to_grid_delta

        dc, dr = action_to_grid_delta(action)
        points = list(zip(*np.where(diff)))
        if not points:
            return self.controlled_pos

        point_set = set(points)
        visited = set()
        clusters: List[List[Tuple[int, int]]] = []

        for pt in points:
            if pt in visited:
                continue
            cluster = []
            q = [pt]
            visited.add(pt)
            while q:
                cy, cx = q.pop()
                cluster.append((cy, cx))
                for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nb = (cy + dy, cx + dx)
                    if nb in point_set and nb not in visited:
                        visited.add(nb)
                        q.append(nb)
            clusters.append(cluster)

        best_pos = None
        best_pixel_pos = None
        best_dist = float("inf")

        expected_pos = None
        if self.controlled_pos is not None:
            expected_pos = (self.controlled_pos[0] + dc, self.controlled_pos[1] + dr)

        for cluster in clusters:
            ys = [p[0] for p in cluster]
            xs = [p[1] for p in cluster]
            min_x, min_y = min(xs), min(ys)
            cand_grid = grid_tracker.pixel_to_grid(min_x, min_y)

            if expected_pos is not None:
                err = abs(cand_grid[0] - expected_pos[0]) + abs(cand_grid[1] - expected_pos[1])
                if err < best_dist:
                    best_dist = err
                    best_pos = cand_grid
                    best_pixel_pos = (min_x, min_y)
            else:
                best_pos = cand_grid
                best_pixel_pos = (min_x, min_y)
                best_dist = 0

        if best_pos is not None:
            self.controlled_pos = best_pos
            self.controlled_pixel_pos = best_pixel_pos
            self.controlled_entity_id = f"entity_{best_pos[0]}_{best_pos[1]}"
            self.control_confidence = min(1.0, self.control_confidence + 0.3)

        return self.controlled_pos


class AffordanceModel:
    """
    Discovers, verifies, and maintains interactive object affordances (ACTION5).
    Learns state changes (pickup, drop, switch activation, piece cycling) dynamically from observations.
    """

    def __init__(self) -> None:
        self.affordances: Dict[str, Affordance] = {}
        self.carried_object_id: Optional[str] = None
        self.learned_effects: Dict[str, str] = {}  # object_id -> effect ("pickup", "drop", "switch", "cycle")

    def reset_level(self) -> None:
        """Reset affordances for a fresh level."""
        self.affordances.clear()
        self.carried_object_id = None
        self.learned_effects.clear()

    def register_affordance(
        self,
        target_id: str,
        target_pos: Tuple[int, int],
        predicted_effect: str = "interact",
        required_distance: int = 1,
    ) -> Affordance:
        """Register or update an interactive affordance for a target entity."""
        if target_id not in self.affordances:
            aff = Affordance(
                target_object_id=target_id,
                target_pos=target_pos,
                interaction_action=GameAction.ACTION5,
                required_distance=required_distance,
                predicted_effect=predicted_effect,
                confidence=0.5,
            )
            self.affordances[target_id] = aff
            return aff
        else:
            aff = self.affordances[target_id]
            aff.target_pos = target_pos
            return aff

    def update_from_action5_observation(
        self,
        prev_objects: List[GameObject],
        curr_objects: List[GameObject],
        player_pos: Tuple[int, int],
    ) -> Optional[str]:
        """
        Analyze before/after frames following an ACTION5 execution.
        Learns pickup, drop, focus shift, or state transition.
        """
        prev_map = {f"{o.col}_{o.row}": o for o in prev_objects}
        curr_map = {f"{o.col}_{o.row}": o for o in curr_objects}

        # Check for disappearing adjacent object -> PICKUP
        for key, prev_obj in prev_map.items():
            if key not in curr_map:
                dist = abs(prev_obj.col - player_pos[0]) + abs(prev_obj.row - player_pos[1])
                if dist <= 1:
                    obj_id = f"item_{prev_obj.col}_{prev_obj.row}"
                    self.carried_object_id = obj_id
                    self.learned_effects[obj_id] = "pickup"
                    logger.debug(f"[AFFORDANCE] Learned PICKUP effect for {obj_id} at {prev_obj.grid_pos}")
                    return "pickup"

        # Check for appearing adjacent object while carrying -> DROP
        if self.carried_object_id:
            for key, curr_obj in curr_map.items():
                if key not in prev_map:
                    dist = abs(curr_obj.col - player_pos[0]) + abs(curr_obj.row - player_pos[1])
                    if dist <= 1:
                        dropped_id = self.carried_object_id
                        self.carried_object_id = None
                        self.learned_effects[dropped_id] = "drop"
                        logger.debug(f"[AFFORDANCE] Learned DROP effect for {dropped_id} at {curr_obj.grid_pos}")
                        return "drop"

        return "interact"
