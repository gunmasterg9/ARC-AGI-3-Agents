from __future__ import annotations

from dataclasses import dataclass, field
import logging
import math
from typing import Dict, List, Optional, Set, Tuple
from arcengine import GameAction

from .objects import GameObject, ObjectType

logger = logging.getLogger()


@dataclass
class TransientObstacle:
    """Represents a temporary dynamic blocked cell caused by a moving entity or dynamic collision."""

    position: Tuple[int, int]
    source_id: Optional[str] = None
    observed_step: int = 0
    confidence: float = 1.0
    ttl: int = 3
    expiration_step: int = 3

    @property
    def col(self) -> int:
        return self.position[0]

    @property
    def row(self) -> int:
        return self.position[1]


@dataclass
class MovingObjectModel:
    """
    Representation of a moving obstacle or dynamic entity learned online from observations.
    Maintains observed positions, estimated velocity, trajectory cycle, and period.
    """

    object_id: str
    current_pos: Tuple[int, int]
    previous_pos: Optional[Tuple[int, int]] = None
    velocity: Tuple[int, int] = (0, 0)
    history: List[Tuple[int, int]] = field(default_factory=list)
    time_history: List[int] = field(default_factory=list)
    estimated_period: Optional[int] = None
    estimated_phase: Optional[int] = None
    trajectory_cycle: List[Tuple[int, int]] = field(default_factory=list)
    confidence: float = 0.0
    is_periodic: bool = False
    blocks_player: bool = True
    object_type: ObjectType = ObjectType.MOVING_PLATFORM

    def update_observation(self, pos: Tuple[int, int], t: int) -> None:
        """Integrate a newly observed grid position at time step t."""
        self.previous_pos = self.current_pos
        self.current_pos = pos
        self.history.append(pos)
        self.time_history.append(t)

        if self.previous_pos is not None:
            self.velocity = (pos[0] - self.previous_pos[0], pos[1] - self.previous_pos[1])

        # Attempt to estimate periodicity from observation history
        self._estimate_periodicity()

    def _estimate_periodicity(self) -> None:
        """Estimate the cyclic period P and phase from history of observations."""
        n = len(self.history)
        if n < 4:
            self.confidence = 0.2
            return

        # Check candidate periods from P=2 up to min(16, n // 2)
        best_p: Optional[int] = None
        best_matches = 0

        for candidate_p in range(2, min(17, n)):
            matches = 0
            checks = 0
            for i in range(candidate_p, n):
                checks += 1
                if self.history[i] == self.history[i - candidate_p]:
                    matches += 1

            if checks > 0 and matches == checks:
                best_p = candidate_p
                best_matches = matches
                break

        if best_p is not None:
            self.estimated_period = best_p
            self.is_periodic = True
            # Extract cycle: last P positions
            self.trajectory_cycle = list(self.history[-best_p:])
            # The current position is the last element of the extracted cycle
            self.estimated_phase = len(self.trajectory_cycle) - 1

            # Confidence scales with number of consistent cycles observed
            cycles_observed = best_matches / best_p
            self.confidence = min(1.0, 0.4 + 0.2 * cycles_observed)
        else:
            self.is_periodic = False
            self.estimated_period = None
            self.trajectory_cycle = []
            self.confidence = 0.3 if (self.velocity[0] != 0 or self.velocity[1] != 0) else 0.1

    def predict_position_at(
        self,
        target_t: int,
        current_t: Optional[int] = None,
        grid_cols: int = 16,
        grid_rows: int = 16,
    ) -> Tuple[int, int]:
        """
        Predict the object's grid position at future time step target_t.
        Uses periodic cycle if known and confident, otherwise linear extrapolation or last position.
        """
        if current_t is None:
            current_t = self.time_history[-1] if self.time_history else 0

        dt = target_t - current_t
        if dt == 0:
            return self.current_pos

        # Periodic prediction
        if self.is_periodic and self.estimated_period and len(self.trajectory_cycle) == self.estimated_period:
            idx = (self.estimated_phase + dt) % self.estimated_period
            return self.trajectory_cycle[idx]

        # Linear extrapolation with bounds damping if velocity is non-zero
        if self.velocity != (0, 0) and (self.confidence >= 0.25 or len(self.history) >= 2):
            pred_col = self.current_pos[0] + self.velocity[0] * dt
            pred_row = self.current_pos[1] + self.velocity[1] * dt
            # Clamp to grid bounds (at least 15 for 16x16 grid, e.g. WA30)
            max_c = max(10, grid_cols - 1)
            max_r = max(10, grid_rows - 1)
            return (max(0, min(max_c, pred_col)), max(0, min(max_r, pred_row)))

        # Default fallback
        return self.current_pos

    def predict_trajectory(self, start_t: int, horizon: int) -> List[Tuple[int, int]]:
        """Generate predicted trajectory for horizon steps starting from start_t."""
        return [self.predict_position_at(start_t + step, current_t=start_t) for step in range(horizon)]


class TemporalWorldModel:
    """
    Temporal extension of the environment model.
    Maintains discrete time, tracks dynamic moving entities, and evaluates space-time safety.
    """

    def __init__(self) -> None:
        self.current_time: int = 0
        self.moving_objects: Dict[str, MovingObjectModel] = {}
        self.prev_object_positions: Dict[str, Tuple[int, int]] = {}
        self.transient_obstacles: Dict[Tuple[int, int], TransientObstacle] = {}

    def estimate_cell_vacancy(
        self,
        col: int,
        row: int,
        start_t: int,
        max_k: int = 4,
    ) -> Optional[int]:
        """
        Estimate when (in how many time steps k in 1..max_k) the given cell will become vacant.
        Returns the earliest k where is_temporally_blocked is False, or None if it remains blocked.
        """
        for k in range(1, max_k + 1):
            t = start_t + k
            if not self.is_temporally_blocked(col, row, target_t=t):
                return k
        return None

    def reset_level(self) -> None:
        """Reset temporal state for clean level isolation."""
        self.current_time = 0
        self.moving_objects.clear()
        self.prev_object_positions.clear()
        self.transient_obstacles.clear()

    def add_transient_obstacle(
        self,
        position: Tuple[int, int],
        source_id: Optional[str] = None,
        t: int = 0,
        ttl: int = 3,
        confidence: float = 1.0,
    ) -> TransientObstacle:
        """Register a temporary blocked cell with expiration TTL."""
        obs = TransientObstacle(
            position=position,
            source_id=source_id,
            observed_step=t,
            confidence=confidence,
            ttl=ttl,
            expiration_step=t + ttl,
        )
        self.transient_obstacles[position] = obs
        logger.debug(
            f"[TEMPORAL] Added transient obstacle at {position} (source={source_id}, TTL={ttl}, expires={t+ttl})"
        )
        return obs

    def update_transient_obstacles(
        self,
        current_t: int,
        moving_positions: Optional[Set[Tuple[int, int]]] = None,
    ) -> None:
        """Prune expired transient obstacles or those whose source entity has moved away."""
        self.current_time = current_t
        to_remove = []
        for pos, obs in self.transient_obstacles.items():
            # 1. Expire based on TTL
            if current_t >= obs.expiration_step:
                to_remove.append(pos)
                continue
            # 2. If source entity is known and has moved to a different cell
            if obs.source_id and obs.source_id in self.moving_objects:
                mover = self.moving_objects[obs.source_id]
                if mover.current_pos != pos:
                    to_remove.append(pos)
                    continue
            # 3. If moving positions set is provided and this cell is now vacated
            if moving_positions is not None and pos not in moving_positions and obs.confidence < 0.95:
                if current_t > obs.observed_step:
                    to_remove.append(pos)
        for pos in to_remove:
            logger.debug(f"[TEMPORAL] Pruned transient obstacle at {pos}")
            del self.transient_obstacles[pos]

    def update_time(self, t: int) -> None:
        """Advance internal discrete clock."""
        self.current_time = t

    def update_from_detected_objects(
        self,
        detected_objects: List[GameObject],
        t: int,
    ) -> None:
        """
        Discover and update moving objects online from frame observations.
        Matches detected objects to track trajectories, velocities, and periodicity.
        """
        self.current_time = t
        current_seen: Dict[str, Tuple[int, int]] = {}

        # Filter candidate moving entities (moving platforms, unknown objects)
        for obj in detected_objects:
            if obj.object_type in (
                ObjectType.MOVING_PLATFORM,
                ObjectType.UNKNOWN,
            ):
                pos = (obj.col, obj.row)
                obj_id = f"{obj.object_type.value}_{pos[0]}_{pos[1]}"
                current_seen[obj_id] = pos

        occupied_current = set(current_seen.values())

        # Check motion against previous frame positions
        for obj_id, pos in current_seen.items():
            base_type = obj_id.rsplit("_", 2)[0]
            # Try to match with an existing moving object of the same type within 1 cell Manhattan distance
            matched_key: Optional[str] = None
            for existing_id, mover in self.moving_objects.items():
                if mover.object_type.value == base_type:
                    dist = abs(mover.current_pos[0] - pos[0]) + abs(mover.current_pos[1] - pos[1])
                    if dist <= 1:
                        matched_key = existing_id
                        break

            if matched_key is not None:
                self.moving_objects[matched_key].update_observation(pos, t)
            else:
                # Check if this object moved from a previous position
                for prev_id, prev_pos in self.prev_object_positions.items():
                    # If previous position is still occupied in the current frame, it did not vacate/move
                    if prev_pos in occupied_current:
                        continue
                    prev_type = prev_id.rsplit("_", 2)[0]
                    if prev_type == base_type:
                        dist = abs(prev_pos[0] - pos[0]) + abs(prev_pos[1] - pos[1])
                        if dist == 1 and prev_pos != pos:
                            # Confirmed movement! Register as a moving object
                            mover_id = f"mover_{base_type}_{len(self.moving_objects)}"
                            obj_type = ObjectType(base_type) if base_type in [e.value for e in ObjectType] else ObjectType.MOVING_PLATFORM
                            is_transformer = obj_type in (
                                ObjectType.TRANSFORMER_ROTATION,
                                ObjectType.TRANSFORMER_COLOR,
                                ObjectType.TRANSFORMER_SHAPE,
                            )
                            mover = MovingObjectModel(
                                object_id=mover_id,
                                current_pos=pos,
                                previous_pos=prev_pos,
                                object_type=obj_type,
                                blocks_player=(not is_transformer),
                            )
                            mover.history = [prev_pos, pos]
                            mover.time_history = [t - 1, t]
                            mover.velocity = (pos[0] - prev_pos[0], pos[1] - prev_pos[1])
                            self.moving_objects[mover_id] = mover
                            logger.debug(f"[TEMPORAL] Discovered moving object {mover_id}: {prev_pos} -> {pos}")
                            break

        self.prev_object_positions = current_seen

    def has_active_moving_objects(self) -> bool:
        """Return True if at least one moving obstacle that blocks player is actively tracked."""
        return any(mover.blocks_player for mover in self.moving_objects.values())

    def compute_effective_period(self) -> Optional[int]:
        """
        Compute the least common multiple (LCM) of all confirmed periodic moving objects.
        Returns None if no periodic objects exist.
        """
        periods = [
            m.estimated_period
            for m in self.moving_objects.values()
            if m.is_periodic and m.estimated_period and m.confidence >= 0.5
        ]
        if not periods:
            return None

        # Compute LCM
        lcm = periods[0]
        for p in periods[1:]:
            lcm = (lcm * p) // math.gcd(lcm, p)
        return min(lcm, 64)  # Bound state space explosion

    def is_temporally_blocked(
        self,
        col: int,
        row: int,
        target_t: int,
        prev_pos: Optional[Tuple[int, int]] = None,
        target_cell: Optional[Tuple[int, int]] = None,
    ) -> bool:
        """
        Check whether the target grid cell at target_t is blocked by any predicted moving obstacle
        or active transient obstacle.
        Also detects vertex swap (edge collision) with an oncoming obstacle.
        """
        pos = (col, row)
        # Check active transient obstacles
        if pos in self.transient_obstacles:
            obs = self.transient_obstacles[pos]
            if target_t < obs.expiration_step:
                if target_cell is None or pos != target_cell:
                    return True

        curr_t = self.current_time
        for mover in self.moving_objects.values():
            if not mover.blocks_player:
                continue

            pred_pos = mover.predict_position_at(target_t, current_t=curr_t)

            # If this cell is the intended destination target, do not treat as obstacle
            if target_cell is not None and (col, row) == target_cell:
                continue

            # Direct collision at target_t
            if (col, row) == pred_pos:
                return True

            # Vertex swap collision: player moves (prev_pos -> target) while mover moves (target -> prev_pos)
            if prev_pos is not None:
                mover_prev = mover.predict_position_at(target_t - 1, current_t=curr_t)
                if prev_pos == pred_pos and (col, row) == mover_prev:
                    return True

        return False

    def predict_hazard_cells(self, target_t: int) -> Set[Tuple[int, int]]:
        """Return set of all grid cells occupied by moving obstacles at target_t."""
        hazards: Set[Tuple[int, int]] = set()
        curr_t = self.current_time
        for mover in self.moving_objects.values():
            if mover.blocks_player:
                hazards.add(mover.predict_position_at(target_t, current_t=curr_t))
        return hazards

    def verify_prediction(self, observed_objects: List[GameObject], t: int) -> bool:
        """
        Compare predicted positions against newly observed positions.
        Returns False if a prediction mismatch is detected (signaling need to replan).
        """
        curr_seen = {(o.col, o.row) for o in observed_objects}
        for mover in self.moving_objects.values():
            if mover.confidence >= 0.6:
                pred = mover.predict_position_at(t, current_t=self.current_time)
                # If predicted position has no detected object of that type, mismatch
                if pred not in curr_seen and mover.current_pos in curr_seen:
                    mover.confidence = max(0.1, mover.confidence - 0.3)
                    logger.debug(f"[TEMPORAL] Prediction mismatch for {mover.object_id}: pred={pred}")
                    return False
        return True
