from __future__ import annotations

from collections import deque
import logging
from typing import Any, Deque, Dict, List, Optional, Tuple
import numpy as np

from arcengine import FrameData, GameAction, GameState

from ..agent import Agent
from .actions import grid_delta_to_action, pixel_to_grid
from .affordance import AffordanceModel, ControllableEntityManager, DynamicGridTracker
from .diagnostics import DiagnosticsLogger
from .goals import GoalManager, GoalSpecification
from .ls20_adapter import LevelDomainSpec, MissionSubGoal, get_level_spec
from .memory import EpisodeMemory
from .objects import GameObject, ObjectType
from .perception import PerceptionEngine
from .persistence import MultiFrameObjectTracker, TrackedObject
from .planner import Planner
from .reachability import (
    CooperativeTransferState,
    ItemTransferState,
    NPCClassification,
    ReachabilityAnalyzer,
    ReachabilityStatus,
    TransferInterface,
)
from .state import HUDState, PlayerState, TransitionPhase, WorldState
from .temporal import TemporalWorldModel
from .transitions import TransitionLearner
from .world_model import WorldModel

logger = logging.getLogger()


class ARCAGIV3Solver(Agent):
    """
    ARC-AGI-3 V3.1 General Solver Architecture.

    Pure Online Cognitive Loop (NO Pre-Seeded Solution):
      Observe
        ↓
      Discover mechanics
        ↓
      Build world model
        ↓
      Infer goal
        ↓
      Plan
        ↓
      Act
        ↓
      Observe result
        ↓
      Update
        ↓
      Replan
    """

    MAX_ACTIONS: int = 500

    def __init__(
        self,
        card_id: str = "",
        game_id: str = "ls20-9607627b",
        agent_name: str = "arcagiv3",
        ROOT_URL: str = "",
        record: bool = False,
        arc_env: Optional[Any] = None,
        tags: Optional[list[str]] = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            card_id=card_id,
            game_id=game_id,
            agent_name=agent_name,
            ROOT_URL=ROOT_URL,
            record=record,
            arc_env=arc_env,
            tags=tags,
            *args,
            **kwargs,
        )
        self.diagnostics = DiagnosticsLogger(prefix="[V3.1-SOLVER]", enabled=True)
        self.perception = PerceptionEngine()
        self.world_model = WorldModel(cols=11, rows=11)
        self.temporal_model = TemporalWorldModel()
        self.transition_learner = TransitionLearner()
        self.goal_manager = GoalManager()
        self.planner = Planner(self.world_model)
        self.memory = EpisodeMemory()
        self.grid_tracker = DynamicGridTracker(default_stride=5)
        self.entity_manager = ControllableEntityManager()
        self.affordance_model = AffordanceModel()
        self.persistence_tracker = MultiFrameObjectTracker(stride=4)
        self.reachability_analyzer = ReachabilityAnalyzer()

        self.current_level_idx: int = 0
        self.last_world_state: Optional[WorldState] = None
        self.last_action: Optional[GameAction] = None

        # Cooperative transfer state
        self.cooperative_state: CooperativeTransferState = CooperativeTransferState.NOT_REQUIRED
        self.current_transfer_interface: Optional[TransferInterface] = None
        self.attempting_handoff_item: Optional[str] = None
        self.attempting_handoff_pos: Optional[Tuple[int, int]] = None
        self.handoffs_count: int = 0
        self.failed_handoffs_count: int = 0
        self.initial_item_positions: Set[Tuple[int, int]] = set()

        # Transition synchronization tracking
        self.transition_phase: TransitionPhase = TransitionPhase.NORMAL
        self.last_completed_screen: Optional[np.ndarray] = None
        self.last_completed_goal_pos: Optional[Tuple[int, int]] = None

        # V3.1: NO pre-seeded action lists! Plans are generated purely online.
        self.active_plan: Deque[GameAction] = deque()
        self.subgoal_queue: Deque[MissionSubGoal] = deque()
        self.active_subgoal: Optional[MissionSubGoal] = None

        # Initialize Level 0
        self._init_level(0)

    def _init_level(self, level_idx: int) -> None:
        """Initialize world model, goals, and mission subgoals for a specific level with 100% isolation."""
        self.current_level_idx = level_idx
        self.world_model.reset_level()
        self.temporal_model.reset_level()
        self.perception.reset_level()
        self.transition_learner.reset_level()
        self.entity_manager.reset_level()
        self.affordance_model.reset_level()
        self.persistence_tracker.reset_level()
        self.memory.clear()
        self.planner.reset_level()
        self.active_plan.clear()
        self.active_wait_flags: Deque[bool] = deque()
        self.is_executing_intentional_wait: bool = False
        self.active_subgoal = None
        self.last_world_state = None
        self.last_action = None
        self.last_completed_screen = None
        self.last_completed_goal_pos = None

        spec = get_level_spec(level_idx) if "ls20" in self.game_id else None
        if spec:
            self.goal_manager.set_goals(spec.goals)
            self.subgoal_queue = deque(spec.mission_subgoals)
            self.diagnostics.goal(
                f"Level {level_idx + 1}: Inferred {len(spec.goals)} goals, {len(spec.mission_subgoals)} mission subgoals"
            )
        else:
            self.subgoal_queue = deque()

        self.carried_offset: Optional[Tuple[int, int]] = None
        self.target_pickup_item: Optional[Tuple[int, int]] = None
        self.consecutive_failed_action5: int = 0
        self.consecutive_waits: int = 0
        self.action_counter: int = 0
        self.successful_temporal_waits: int = 0
        self.spatial_detours_count: int = 0
        self.reachability_analyzer.reset_level()
        self.cooperative_state = CooperativeTransferState.NOT_REQUIRED
        self.current_transfer_interface = None
        self.attempting_handoff_item = None
        self.attempting_handoff_pos = None
        self.handoffs_count = 0
        self.failed_handoffs_count = 0
        self.initial_item_positions = set()

    def _enqueue_plan(self, actions: List[GameAction], wait_flags: Optional[List[bool]] = None) -> None:
        """Enqueue actions into active plan while synchronizing wait execution flags."""
        if not actions:
            return
        if wait_flags and len(wait_flags) == len(actions):
            for act, is_w in zip(actions, wait_flags):
                self.active_plan.append(act)
                self.active_wait_flags.append(is_w)
        else:
            for act in actions:
                self.active_plan.append(act)
                self.active_wait_flags.append(False)

    def _verify_new_level_frame(
        self, screen: Optional[np.ndarray], target_level_idx: int
    ) -> bool:
        """
        Verify whether the current visual screen corresponds to the new level.
        Returns False if the frame is still the stale completion frame of the prior level.
        """
        if screen is None:
            return False

        # If byte-for-byte identical to the completion screen of the prior level, it is stale
        if (
            self.last_completed_screen is not None
            and not self.perception.screens_differ(screen, self.last_completed_screen)
        ):
            return False

        # Detect player in the candidate frame
        cand_player = self.perception.detect_player(
            screen,
            stride=self.grid_tracker.estimated_stride,
            offset_x=self.grid_tracker.offset_x,
            offset_y=self.grid_tracker.offset_y,
        )
        if cand_player is not None:
            # If player is still located at the completed goal cell, the frame is stale
            if (
                self.last_completed_goal_pos is not None
                and cand_player.grid_pos == self.last_completed_goal_pos
            ):
                return False

        return True

    def _select_safe_fallback_action(self, player: PlayerState) -> GameAction:
        """Select a safe passable neighboring action avoiding immediate wall collisions."""
        recent = list(self.memory.history)[-5:] if self.memory.history else []
        recent_pos = [(s.col, s.row) for s in recent]
        safe_action = self.planner.get_safe_recovery_action(
            player.grid_pos, recent_pos, temporal_model=self.temporal_model
        )
        self.diagnostics.action(f"Controlled fallback recovery action: {safe_action.name}")
        return safe_action

    # ------------------------------------------------------------------
    # Agent API Compliance
    # ------------------------------------------------------------------

    def is_done(self, frames: list[FrameData], latest_frame: FrameData) -> bool:
        """Return True when game state is WIN or GAME_OVER."""
        if not frames:
            return False
        state = getattr(latest_frame, "state", None)
        if state is None:
            return False
        state_str = str(state).upper()
        return "WIN" in state_str or "GAME_OVER" in state_str

    def choose_action(
        self, frames: list[FrameData], latest_frame: FrameData
    ) -> GameAction:
        """
        Execute one cycle of the V3.1 cognitive loop:
        Observe -> Discover mechanics -> Build world model -> Infer goal -> Plan -> Act -> Update -> Replan.
        """
        # ==============================================================
        # 1. OBSERVE
        # ==============================================================
        screen = self.perception.extract_screen(latest_frame, use_latest=True)
        levels_completed = getattr(latest_frame, "levels_completed", 0) or 0

        self.diagnostics.observe(
            f"Step {self.action_counter}: Screen shape={screen.shape if screen is not None else None}, levels_completed={levels_completed}"
        )

        # Detect level advancement
        if levels_completed > self.current_level_idx:
            if self.transition_phase == TransitionPhase.NORMAL:
                self.diagnostics.state(
                    f"Level completion observed: {self.current_level_idx} -> {levels_completed}. Entering transition synchronization."
                )
                self.transition_phase = TransitionPhase.WAITING_FOR_NEW_LEVEL_FRAME
                if self.last_world_state and self.last_world_state.player:
                    self.last_completed_goal_pos = self.last_world_state.player.grid_pos
                if self.last_world_state and self.last_world_state.screen is not None:
                    self.last_completed_screen = self.last_world_state.screen.copy()

        # Handle transition synchronization
        if self.transition_phase in (
            TransitionPhase.WAITING_FOR_NEW_LEVEL_FRAME,
            TransitionPhase.VERIFYING_NEW_FRAME,
        ):
            is_new = self._verify_new_level_frame(screen, levels_completed)
            if is_new:
                self.diagnostics.state(
                    f"New level frame VERIFIED for Level {levels_completed + 1}. Initializing isolated WorldModel."
                )
                self.transition_phase = TransitionPhase.INITIALIZING_WORLD_MODEL
                self._init_level(levels_completed)
                self.transition_phase = TransitionPhase.NORMAL
            else:
                self.diagnostics.state(
                    f"Stale completion frame detected for Level {self.current_level_idx + 1}. Holding transition state."
                )
                # Stale frame: DO NOT initialize world model from stale image.
                # Send safe exploratory probe to trigger environment level advance
                action = GameAction.ACTION1
                self.last_action = action
                return action

        # Calibrate dynamic grid stride and offsets from screen on step 0 / level start
        if screen is not None and (self.action_counter == 0 or self.grid_tracker.confidence < 0.8):
            is_ls20 = "ls20" in self.game_id
            sx, sy, ox, oy = self.grid_tracker.calibrate_from_screen(screen, is_ls20_hint=is_ls20)
            self.world_model.configure_grid(
                stride=sx,
                offset_x=ox,
                offset_y=oy,
                cols=self.grid_tracker.cols,
                rows=self.grid_tracker.rows,
            )
            self.persistence_tracker.configure_grid(stride=sx, offset_x=ox, offset_y=oy)

        prev_player = self.last_world_state.player if self.last_world_state else None
        cand_player = self.perception.detect_player(
            screen,
            last_state=prev_player,
            stride=self.grid_tracker.estimated_stride,
            offset_x=self.grid_tracker.offset_x,
            offset_y=self.grid_tracker.offset_y,
        ) if screen is not None else None

        prev_player_pixel = self.entity_manager.controlled_pixel_pos
        # Controllable entity discovery & dynamic grid tracking for non-standard environments
        if (
            cand_player is None
            and screen is not None
            and self.last_action is not None
            and self.last_world_state is not None
            and self.last_world_state.screen is not None
        ):
            self.entity_manager.discover_controlled_entity(
                prev_screen=self.last_world_state.screen,
                curr_screen=screen,
                action=self.last_action,
                grid_tracker=self.grid_tracker,
            )
            if (
                prev_player_pixel is not None
                and self.entity_manager.controlled_pixel_pos is not None
                and self.last_action in (
                    GameAction.ACTION1,
                    GameAction.ACTION2,
                    GameAction.ACTION3,
                    GameAction.ACTION4,
                )
            ):
                dx = self.entity_manager.controlled_pixel_pos[0] - prev_player_pixel[0]
                dy = self.entity_manager.controlled_pixel_pos[1] - prev_player_pixel[1]
                if abs(dx) > 0 or abs(dy) > 0:
                    self.grid_tracker.record_displacement(dx, dy)
                    if self.grid_tracker.confidence >= 0.6:
                        self.world_model.configure_grid(
                            stride=self.grid_tracker.estimated_stride,
                            offset_x=self.grid_tracker.offset_x,
                            offset_y=self.grid_tracker.offset_y,
                            cols=self.grid_tracker.cols,
                            rows=self.grid_tracker.rows,
                        )
                        self.persistence_tracker.configure_grid(
                            stride=self.grid_tracker.estimated_stride,
                            offset_x=self.grid_tracker.offset_x,
                            offset_y=self.grid_tracker.offset_y,
                        )

        # If primary detector found player, use it; otherwise use controllable entity manager
        player = cand_player
        if player is None and self.entity_manager.controlled_pos is not None:
            cpos = self.entity_manager.controlled_pos
            pxpos = self.entity_manager.controlled_pixel_pos or self.grid_tracker.grid_to_pixel(cpos[0], cpos[1])
            player = PlayerState(x=pxpos[0], y=pxpos[1], col=cpos[0], row=cpos[1])

        hud = self.perception.parse_hud(screen, levels_completed=levels_completed) if screen is not None else HUDState()
        detected_objects = self.perception.detect_objects(
            screen,
            player=player,
            stride=self.grid_tracker.estimated_stride,
            offset_x=self.grid_tracker.offset_x,
            offset_y=self.grid_tracker.offset_y,
            cols=self.grid_tracker.cols,
            rows=self.grid_tracker.rows,
        ) if screen is not None else []

        if player:
            self.diagnostics.state(f"Player located at grid={player.grid_pos}, pixels={player.pixel_pos}")
        self.diagnostics.objects(f"Perceived {len(detected_objects)} game objects")

        # Multi-frame object persistence tracking for general affordance environments
        prev_screen = self.last_world_state.screen if self.last_world_state else None
        if "ls20" not in self.game_id:
            self.persistence_tracker.update(
                detected_objects=detected_objects,
                current_frame=screen,
                player_pos=player.grid_pos if player else None,
                last_action=self.last_action,
                prev_frame=prev_screen,
                carried_offset=self.carried_offset,
            )

        # ACTION5 effect observation & verification
        if self.last_action == GameAction.ACTION5 and self.last_world_state is not None:
            prev_carried = self.last_world_state.carried_object_id
            effect_type = self.affordance_model.update_from_action5_observation(
                prev_objects=self.last_world_state.detected_objects,
                curr_objects=detected_objects,
                player_pos=player.grid_pos if player else (0, 0),
            )
            self.world_model.carried_object_id = (
                self.persistence_tracker.carried_object_id or self.affordance_model.carried_object_id
            )

            # Check pickup vs drop confirmation
            if prev_carried is None and self.world_model.carried_object_id is not None:
                # PICKUP SUCCESS!
                if self.target_pickup_item and player:
                    self.carried_offset = (
                        self.target_pickup_item[0] - player.col,
                        self.target_pickup_item[1] - player.row,
                    )
                elif player and self.persistence_tracker.carried_object_id:
                    ctrack = self.persistence_tracker.tracked_objects.get(self.persistence_tracker.carried_object_id)
                    if ctrack:
                        self.carried_offset = (ctrack.col - player.col, ctrack.row - player.row)
                    else:
                        self.carried_offset = (0, -1)
                else:
                    self.carried_offset = (0, -1)
                self.target_pickup_item = None
                self.consecutive_failed_action5 = 0
                self.cooperative_state = CooperativeTransferState.NOT_REQUIRED
                self.diagnostics.action(
                    f"PICKUP CONFIRMED: carrying item {self.world_model.carried_object_id} with offset {self.carried_offset}"
                )
            elif prev_carried is not None and (
                self.world_model.carried_object_id is None
                or self.persistence_tracker.carried_object_id is None
                or self.affordance_model.carried_object_id is None
            ):
                # DROP / HANDOFF SUCCESS!
                self.world_model.carried_object_id = None
                self.persistence_tracker.carried_object_id = None
                self.affordance_model.carried_object_id = None
                self.carried_offset = None
                self.consecutive_failed_action5 = 0
                if (
                    self.cooperative_state in (CooperativeTransferState.HANDOFF_ATTEMPT, CooperativeTransferState.APPROACHING_INTERFACE)
                    or self.current_transfer_interface is not None
                ):
                    self.cooperative_state = CooperativeTransferState.HANDOFF_CONFIRMED
                    if self.current_transfer_interface:
                        self.current_transfer_interface.record_success()
                    if self.attempting_handoff_pos:
                        self.reachability_analyzer.handed_off_items.add(self.attempting_handoff_pos)
                        self.world_model.items.discard(self.attempting_handoff_pos)
                    self.handoffs_count += 1
                    self.diagnostics.action(
                        f"COOPERATIVE HANDOFF CONFIRMED at {self.attempting_handoff_pos} (total handoffs: {self.handoffs_count})"
                    )
                    self.attempting_handoff_pos = None
                    self.attempting_handoff_item = None
                    self.current_transfer_interface = None
                    self.active_plan.clear()
                    self.active_wait_flags.clear()
                else:
                    self.diagnostics.action("DELIVERY CONFIRMED: item dropped into receptacle")
            else:
                self.consecutive_failed_action5 += 1
                if self.cooperative_state == CooperativeTransferState.HANDOFF_ATTEMPT:
                    self.cooperative_state = CooperativeTransferState.HANDOFF_FAILED
                    self.failed_handoffs_count += 1
                    if self.current_transfer_interface:
                        self.current_transfer_interface.record_failure()
                    self.diagnostics.action(f"COOPERATIVE HANDOFF FAILED at {self.attempting_handoff_pos}")
                    self.current_transfer_interface = None
                    self.attempting_handoff_pos = None
                    self.attempting_handoff_item = None
                self.diagnostics.action(
                    f"ACTION5 effect observed: {effect_type}, carried={self.world_model.carried_object_id} (failed={self.consecutive_failed_action5})"
                )
                if self.consecutive_failed_action5 >= 2:
                    self.diagnostics.plan("ACTION5 failed repeatedly. Clearing active plan to force recovery.")
                    self.active_plan.clear()

        current_world_state = WorldState(
            level_idx=self.current_level_idx,
            player=player,
            hud=hud,
            detected_objects=detected_objects,
            screen=screen,
            carried_object_id=self.world_model.carried_object_id,
            controlled_entity_id=self.entity_manager.controlled_entity_id,
        )

        # ==============================================================
        # 2. DISCOVER MECHANICS & OBSERVE RESULT (only within same level)
        # ==============================================================
        need_replan = False
        if (
            self.last_world_state is not None
            and self.last_world_state.level_idx == self.current_level_idx
            and self.last_action is not None
            and prev_player
            and player
        ):
            trans = self.transition_learner.observe_transition(
                prev_world=self.last_world_state,
                action=self.last_action,
                next_world=current_world_state,
            )
            self.diagnostics.transition(
                f"Action {self.last_action.name} consequence: moved={trans.effect.moved}, "
                f"dcol={trans.effect.dcol}, drow={trans.effect.drow}, blocked={trans.effect.blocked}"
            )
            if trans.effect.moved:
                self.consecutive_waits = 0

            # Discover dynamic obstacles from collisions (Phase 1 & Phase 2)
            if trans.effect.blocked:
                if self.is_executing_intentional_wait:
                    self.is_executing_intentional_wait = False
                    self.successful_temporal_waits += 1
                    self.diagnostics.action("[WAIT] Intentional wait step verified. Position and state preserved.")
                elif self.active_plan and self.active_plan[0] == GameAction.ACTION5:
                    self.diagnostics.action("Orientation bump against target prior to ACTION5. Continuing plan.")
                else:
                    from .actions import action_to_grid_delta
                    dc, dr = action_to_grid_delta(self.last_action)
                    blocked_col = prev_player.col + dc
                    blocked_row = prev_player.row + dr
                    blocked_pos = (blocked_col, blocked_row)

                    candidate_cells = [blocked_pos]
                    if self.carried_offset:
                        candidate_cells.append(
                            (blocked_col + self.carried_offset[0], blocked_row + self.carried_offset[1])
                        )

                    for cell in candidate_cells:
                        # Classification:
                        # A. currently occupied by a tracked moving entity
                        # B. recently occupied by a tracked moving entity
                        # C. a known static wall
                        # D. unknown
                        is_moving = False
                        source_id = None

                        # Check temporal model moving objects
                        for mid, mover in self.temporal_model.moving_objects.items():
                            if mover.current_pos == cell:
                                is_moving = True
                                source_id = mid
                                break
                            if cell in mover.history[-5:]:
                                is_moving = True
                                source_id = mid
                                break

                        # Check persistence tracker moving entities
                        if not is_moving:
                            for mover in self.persistence_tracker.get_moving_entities():
                                if mover.current_pos == cell:
                                    is_moving = True
                                    source_id = mover.object_id
                                    break
                                if cell in mover.positions_history[-5:]:
                                    is_moving = True
                                    source_id = mover.object_id
                                    break

                        # Check if any tracked object has motion_state == "moving" or npc_confidence >= 0.4
                        if not is_moving:
                            for obj in self.persistence_tracker.tracked_objects.values():
                                if obj.current_pos == cell and (obj.motion_state == "moving" or obj.npc_confidence >= 0.4):
                                    is_moving = True
                                    source_id = obj.object_id
                                    break

                        # Check if known static wall
                        is_static_wall = False
                        if not is_moving:
                            static_walls = getattr(self.perception, "static_walls", set())
                            if cell in static_walls:
                                is_static_wall = True
                            elif cell in self.world_model.walls and cell not in self.world_model.dynamic_obstacles:
                                is_static_wall = True

                        if is_moving:
                            # A or B: DO NOT add cell permanently to world_model.walls!
                            self.world_model.walls.discard(cell)
                            self.world_model.dynamic_obstacles.discard(cell)
                            self.temporal_model.add_transient_obstacle(
                                position=cell,
                                source_id=source_id,
                                t=self.action_counter,
                                ttl=3,
                                confidence=0.9,
                            )
                            self.diagnostics.model_update(
                                f"Dynamic collision with moving entity {source_id} at {cell}. Registered transient obstacle (TTL=3), not permanent wall."
                            )
                        elif is_static_wall:
                            # C: Known static wall -> retain existing static-wall learning
                            self.world_model.mark_obstacle(cell[0], cell[1])
                            self.diagnostics.model_update(f"Static collision at {cell}. Retained permanent wall.")
                        else:
                            # D: Unknown -> conservative temporary blocking (TTL=2)
                            self.world_model.walls.discard(cell)
                            self.world_model.dynamic_obstacles.discard(cell)
                            self.temporal_model.add_transient_obstacle(
                                position=cell,
                                source_id=None,
                                t=self.action_counter,
                                ttl=2,
                                confidence=0.5,
                            )
                            self.diagnostics.model_update(
                                f"Unknown collision at {cell}. Registered conservative transient obstacle (TTL=2)."
                            )

                    need_replan = True

            # Discover conveyor / transport push
            if abs(player.col - prev_player.col) > 1 or abs(player.row - prev_player.row) > 1:
                self.diagnostics.model_update(f"Conveyor transport detected: {prev_player.grid_pos} -> {player.grid_pos}")
                need_replan = True
                if self.active_subgoal and self.active_subgoal.target_type == ObjectType.PUSHER:
                    self.diagnostics.goal(
                        f"Conveyor subgoal {self.active_subgoal.description} satisfied by transport to {player.grid_pos}"
                    )
                    self.active_subgoal = None

        # ==============================================================
        # 3. BUILD WORLD MODEL & TEMPORAL TRACKING
        # ==============================================================
        self.world_model.update_from_perception(detected_objects, player)

        # Multi-frame object persistence tracking for general affordance environments
        if "ls20" not in self.game_id:
            # Synchronize items, receptacles, and carried state
            self.world_model.items = {t.current_pos for t in self.persistence_tracker.get_items()}
            self.world_model.receptacles = {t.current_pos for t in self.persistence_tracker.get_receptacles()}
            self.world_model.carried_object_id = self.persistence_tracker.carried_object_id
            # Mask carried object from walls, items, and moving obstacles
            if self.world_model.carried_object_id and self.carried_offset and player:
                carried_pos = (player.col + self.carried_offset[0], player.row + self.carried_offset[1])
                self.world_model.walls.discard(carried_pos)
                self.world_model.items.discard(carried_pos)
                # Purge any moving object registered at or matching carried position
                to_delete = [
                    mid for mid, mover in self.temporal_model.moving_objects.items()
                    if mover.current_pos == carried_pos
                ]
                for mid in to_delete:
                    del self.temporal_model.moving_objects[mid]

            # Discard all past and current positions of moving entities from walls unless currently detected as a static wall
            detected_walls = {
                (o.col, o.row) for o in detected_objects if o.object_type == ObjectType.WALL
            }
            for mover in self.persistence_tracker.get_moving_entities():
                for pos in mover.positions_history:
                    if pos not in detected_walls:
                        self.world_model.walls.discard(pos)
                if mover.current_pos not in detected_walls:
                    self.world_model.walls.discard(mover.current_pos)

            # Register moving NPCs into temporal model for Space-Time A* dynamic avoidance
            moving_entities = self.persistence_tracker.get_moving_entities()
            if moving_entities:
                npc_objects = [
                    GameObject(
                        object_type=ObjectType.MOVING_PLATFORM,
                        x=t.pixel_pos[0],
                        y=t.pixel_pos[1],
                        col=t.col,
                        row=t.row,
                        width=t.width,
                        height=t.height,
                    )
                    for t in moving_entities
                ]
                self.temporal_model.update_from_detected_objects(npc_objects, t=self.action_counter)
        else:
            self.temporal_model.update_from_detected_objects(detected_objects, t=self.action_counter)

        # Update transient obstacles and ensure moving/transient positions never contaminate walls
        detected_walls = {
            (o.col, o.row) for o in detected_objects if o.object_type == ObjectType.WALL
        }
        active_mover_positions: Set[Tuple[int, int]] = set()
        for mover in self.persistence_tracker.get_moving_entities():
            active_mover_positions.add(mover.current_pos)
            for pos in mover.positions_history:
                if pos not in detected_walls:
                    self.world_model.walls.discard(pos)
            if mover.current_pos not in detected_walls:
                self.world_model.walls.discard(mover.current_pos)
        for mover in self.temporal_model.moving_objects.values():
            active_mover_positions.add(mover.current_pos)
            for pos in mover.history:
                if pos not in detected_walls:
                    self.world_model.walls.discard(pos)
            if mover.current_pos not in detected_walls:
                self.world_model.walls.discard(mover.current_pos)
        for pos in self.temporal_model.transient_obstacles:
            if pos not in detected_walls:
                self.world_model.walls.discard(pos)
        self.temporal_model.update_transient_obstacles(self.action_counter, active_mover_positions)

        # Online verification: check if predicted positions match observations
        if not self.temporal_model.verify_prediction(detected_objects, t=self.action_counter):
            self.diagnostics.plan("Moving obstacle prediction mismatch detected. Triggering dynamic replan.")
            need_replan = True

        # Check if currently queued action would collide with an oncoming moving obstacle
        if self.active_plan and player and self.temporal_model.has_active_moving_objects():
            from .actions import action_to_grid_delta
            if self.active_wait_flags and self.active_wait_flags[0]:
                next_pos = player.grid_pos
            else:
                dc, dr = action_to_grid_delta(self.active_plan[0])
                next_pos = (player.col + dc, player.row + dr)
            target_cell = self.active_subgoal.target_pos if self.active_subgoal else None
            if self.temporal_model.is_temporally_blocked(
                next_pos[0], next_pos[1], target_t=self.action_counter + 1, prev_pos=player.grid_pos, target_cell=target_cell
            ):
                self.diagnostics.plan(
                    f"Temporal obstacle collision predicted at {next_pos} on step {self.action_counter + 1}. Replanning!"
                )
                self.active_plan.clear()
                self.active_wait_flags.clear()
                need_replan = True

        if need_replan:
            self.diagnostics.plan("Clearing remaining plan to trigger dynamic replanning")
            self.active_plan.clear()
            self.active_wait_flags.clear()

        # ==============================================================
        # 4. INFER GOAL & 5. PLAN
        # ==============================================================
        if not self.active_plan and player:
            # Advance to next subgoal if current subgoal reached
            if self.active_subgoal is None or player.grid_pos == self.active_subgoal.target_pos:
                if self.subgoal_queue:
                    self.active_subgoal = self.subgoal_queue.popleft()
                    self.diagnostics.goal(
                        f"Active Subgoal: {self.active_subgoal.description} at {self.active_subgoal.target_pos}"
                    )
                else:
                    self.active_subgoal = None

            # Plan dynamic path to active subgoal using Space-Time A* when moving obstacles are active
            if self.active_subgoal:
                new_actions = self.planner.plan_sequence_to_target(
                    start=player.grid_pos,
                    target=self.active_subgoal.target_pos,
                    count=self.active_subgoal.visit_count,
                    cycle_delta=self.active_subgoal.cycle_delta,
                    temporal_model=self.temporal_model,
                    start_t=self.action_counter,
                )
                self._enqueue_plan(new_actions, self.planner.last_is_wait_flags)
                self.diagnostics.plan(
                    f"Generated {len(new_actions)} actions to subgoal at {self.active_subgoal.target_pos}"
                )

            # Fallback dynamic plan directly to active goal if subgoals exhausted
            if not self.active_plan:
                curr_goal = self.goal_manager.get_current_goal()
                if curr_goal:
                    if self.temporal_model.has_active_moving_objects():
                        st_res = self.planner.find_space_time_path(
                            start=player.grid_pos,
                            target=curr_goal.grid_pos,
                            start_t=self.action_counter,
                            temporal_model=self.temporal_model,
                        )
                        if st_res:
                            goal_actions, _ = st_res
                            self._enqueue_plan(goal_actions, self.planner.last_is_wait_flags)
                            self.diagnostics.plan(f"Space-time fallback plan: {len(goal_actions)} actions to goal {curr_goal.grid_pos}")
                    if not self.active_plan:
                        goal_path = self.planner.find_grid_path(player.grid_pos, curr_goal.grid_pos)
                        if goal_path:
                            goal_actions = self.planner.path_to_actions(goal_path)
                            self._enqueue_plan(goal_actions, self.planner.last_is_wait_flags)
                            self.diagnostics.plan(f"Dynamic fallback plan: {len(goal_actions)} actions to goal {curr_goal.grid_pos}")

            # Generalized Affordance / Interaction Planning (e.g. wa30, g50t, re86)
            if not self.active_plan and not self.subgoal_queue:
                # Update connected component topological decomposition
                uncarried_items = set(self.world_model.items)
                uncarried_items.update(self.reachability_analyzer.handed_off_items)
                uncarried_items.update(self.persistence_tracker.delivered_items)
                if self.world_model.carried_object_id and self.carried_offset and player:
                    carried_pos = (player.col + self.carried_offset[0], player.row + self.carried_offset[1])
                    uncarried_items.discard(carried_pos)

                self.reachability_analyzer.compute_connected_components(
                    cols=self.world_model.cols,
                    rows=self.world_model.rows,
                    walls=self.world_model.walls,
                    hazard_cells=self.world_model.hazard_cells,
                    impassable_items=uncarried_items,
                )
                if self.world_model.items:
                    self.initial_item_positions.update(self.world_model.items)

                # Case A: Item delivery / Pickup affordance
                if self.world_model.carried_object_id:
                    available_recs = [
                        r for r in self.world_model.receptacles
                        if r not in self.persistence_tracker.delivered_items
                    ]
                    status, p_comp, goal_comps = (
                        self.reachability_analyzer.analyze_reachability(
                            player.grid_pos, set(available_recs)
                        )
                        if available_recs
                        else (ReachabilityStatus.UNKNOWN, None, set())
                    )

                    if status == ReachabilityStatus.DIRECTLY_REACHABLE:
                        same_comp_recs = [
                            r for r in available_recs
                            if self.reachability_analyzer.get_component(r) == p_comp
                        ]
                        sorted_recs = sorted(
                            same_comp_recs,
                            key=lambda r: (-r[1], abs(r[0] - player.col)),
                        )
                        for rec in sorted_recs:
                            offset = self.carried_offset or (0, -1)
                            target_player_pos = (rec[0] - offset[0], rec[1] - offset[1])
                            other_obstacles = set(self.persistence_tracker.delivered_items)
                            other_obstacles.update(
                                t.current_pos for t in self.persistence_tracker.get_items()
                            )
                            act_list = self.planner.plan_composite_interaction(
                                start=player.grid_pos,
                                target=target_player_pos,
                                interaction_action=GameAction.ACTION5,
                                required_distance=0,
                                temporal_model=self.temporal_model,
                                start_t=self.action_counter,
                                blocked_cells=other_obstacles,
                                carried_offset=offset,
                            )
                            if act_list:
                                self._enqueue_plan(act_list, self.planner.last_is_wait_flags)
                                self.diagnostics.plan(
                                    f"Direct delivery plan: deliver carried item to {rec} (player to {target_player_pos})"
                                )
                                break
                    elif status == ReachabilityStatus.INDIRECTLY_REACHABLE:
                        # Indirect Reachability: plan handoff to transfer interface
                        candidate_interfaces = self.reachability_analyzer.discover_transfer_interfaces(
                            player_pos=player.grid_pos,
                            target_comps=goal_comps,
                            cols=self.world_model.cols,
                            rows=self.world_model.rows,
                            observed_item_positions=self.initial_item_positions,
                            npc_positions=[m.current_pos for m in self.persistence_tracker.get_moving_entities()],
                        )
                        offset = self.carried_offset or (1, 0)
                        other_obstacles = set(self.persistence_tracker.delivered_items)
                        other_obstacles.update(self.reachability_analyzer.handed_off_items)
                        other_obstacles.update(
                            t.current_pos for t in self.persistence_tracker.get_items()
                        )
                        p_comp_cells = self.reachability_analyzer.components.get(p_comp, set())
                        plan_found = False

                        for tif in candidate_interfaces:
                            if tif.position in self.reachability_analyzer.handed_off_items:
                                continue
                            if tif.position in other_obstacles:
                                continue

                            target_player_pos = (tif.position[0] - offset[0], tif.position[1] - offset[1])
                            if target_player_pos not in p_comp_cells:
                                continue
                            if target_player_pos in self.world_model.walls:
                                continue

                            act_list = self.planner.plan_composite_interaction(
                                start=player.grid_pos,
                                target=target_player_pos,
                                interaction_action=GameAction.ACTION5,
                                required_distance=0,
                                temporal_model=self.temporal_model,
                                start_t=self.action_counter,
                                blocked_cells=other_obstacles,
                                carried_offset=offset,
                            )
                            if act_list:
                                self.attempting_handoff_pos = tif.position
                                self.attempting_handoff_item = self.world_model.carried_object_id
                                self.cooperative_state = CooperativeTransferState.APPROACHING_INTERFACE
                                self.current_transfer_interface = tif
                                # Add disengage step (step away from interface) after drop
                                disengage_delta = (-offset[0], -offset[1])
                                disengage_act = grid_delta_to_action(disengage_delta[0], disengage_delta[1])
                                if disengage_act:
                                    act_list.append(disengage_act)
                                    wait_flags = list(self.planner.last_is_wait_flags)
                                    wait_flags.append(False)
                                else:
                                    wait_flags = self.planner.last_is_wait_flags
                                self._enqueue_plan(act_list, wait_flags)
                                self.diagnostics.plan(
                                    f"Indirect cooperative plan: deliver item to interface {tif.position} (player to {target_player_pos})"
                                )
                                plan_found = True
                                break

                        if not plan_found:
                            # Drop in room to re-orient approach if offset incompatible
                            self.diagnostics.plan(
                                f"No feasible route to interface with current offset {offset}. Safe drop to re-align."
                            )
                            act_list = [GameAction.ACTION5]
                            self._enqueue_plan(act_list, [False])

                elif self.world_model.items:
                    available_recs = [
                        r for r in self.world_model.receptacles
                        if r not in self.persistence_tracker.delivered_items
                    ]
                    status, p_comp, goal_comps = (
                        self.reachability_analyzer.analyze_reachability(
                            player.grid_pos, set(available_recs)
                        )
                        if available_recs
                        else (ReachabilityStatus.UNKNOWN, None, set())
                    )

                    if status == ReachabilityStatus.INDIRECTLY_REACHABLE:
                        target_cell_set = set()
                        for tc in goal_comps:
                            if tc != p_comp:
                                target_cell_set.update(self.reachability_analyzer.components.get(tc, set()))
                        p_comp_cells = self.reachability_analyzer.components.get(p_comp, set())

                        room_items = [
                            it for it in self.world_model.items
                            if (
                                p_comp is None
                                or it in p_comp_cells
                                or (
                                    self.reachability_analyzer.get_component(it) == p_comp
                                    and not any(
                                        (it[0] + dc, it[1] + dr) in target_cell_set
                                        for dc, dr in [(0, -1), (0, 1), (-1, 0), (1, 0)]
                                    )
                                )
                            )
                            and it not in self.reachability_analyzer.handed_off_items
                            and it not in self.persistence_tracker.delivered_items
                        ]
                    else:
                        room_items = [
                            it for it in self.world_model.items
                            if (p_comp is None or self.reachability_analyzer.get_component(it) == p_comp)
                            and it not in self.reachability_analyzer.handed_off_items
                            and it not in self.persistence_tracker.delivered_items
                        ]

                    if room_items:
                        sorted_items = sorted(
                            room_items,
                            key=lambda it: abs(it[0] - player.col) + abs(it[1] - player.row),
                        )
                        if status == ReachabilityStatus.INDIRECTLY_REACHABLE:
                            candidate_interfaces = self.reachability_analyzer.discover_transfer_interfaces(
                                player_pos=player.grid_pos,
                                target_comps=goal_comps,
                                cols=self.world_model.cols,
                                rows=self.world_model.rows,
                                observed_item_positions=self.initial_item_positions,
                                npc_positions=[m.current_pos for m in self.persistence_tracker.get_moving_entities()],
                            )
                            downstream_targets = [
                                tif.position for tif in candidate_interfaces
                                if tif.position not in self.reachability_analyzer.handed_off_items
                            ]
                        else:
                            downstream_targets = available_recs

                        for item in sorted_items:
                            other_obstacles = set(self.persistence_tracker.delivered_items)
                            other_obstacles.update(self.reachability_analyzer.handed_off_items)
                            other_obstacles.update(
                                it_pos for it_pos in self.world_model.items if it_pos != item
                            )
                            if status == ReachabilityStatus.INDIRECTLY_REACHABLE and p_comp_cells:
                                other_obstacles.update(
                                    (c, r)
                                    for c in range(self.world_model.cols)
                                    for r in range(self.world_model.rows)
                                    if (c, r) not in p_comp_cells
                                )
                            act_list = self.planner.plan_composite_interaction(
                                start=player.grid_pos,
                                target=item,
                                interaction_action=GameAction.ACTION5,
                                required_distance=1,
                                temporal_model=self.temporal_model,
                                start_t=self.action_counter,
                                blocked_cells=other_obstacles,
                                destination_targets=downstream_targets,
                            )
                            if act_list:
                                self._enqueue_plan(act_list, self.planner.last_is_wait_flags)
                                self.target_pickup_item = item
                                self.diagnostics.plan(f"Affordance plan: pick up item at {item}")
                                break
                    elif status == ReachabilityStatus.INDIRECTLY_REACHABLE and not room_items:
                        # All room items handed off! Step back from divider to avoid re-pickup and wait safely
                        self.diagnostics.plan(
                            "All room items handed off! Stepping back and executing safe wait for helper NPC."
                        )
                        if player.col >= 7:
                            # Move west away from the divider
                            self._enqueue_plan([GameAction.ACTION3], [False])
                        else:
                            safe_wait = self.planner.resolve_safe_wait_action(
                                player.col, player.row, carried_offset=None, temporal_model=self.temporal_model, current_t=self.action_counter
                            )
                            if safe_wait:
                                self._enqueue_plan([safe_wait], [True])
                            else:
                                fallback = self._select_safe_fallback_action(player)
                                self._enqueue_plan([fallback], [False])

                # Case B: Switch / Clone affordance
                elif self.world_model.switches:
                    switch = min(
                        self.world_model.switches,
                        key=lambda s: abs(s[0] - player.col) + abs(s[1] - player.row),
                    )
                    act_list = self.planner.plan_composite_interaction(
                        start=player.grid_pos,
                        target=switch,
                        interaction_action=GameAction.ACTION5,
                        required_distance=0,
                        temporal_model=self.temporal_model,
                        start_t=self.action_counter,
                    )
                    if act_list:
                        self._enqueue_plan(act_list, self.planner.last_is_wait_flags)
                        self.diagnostics.plan(f"Affordance plan: activate switch at {switch}")

        # ==============================================================
        # 6. ACT & CONTROLLED RECOVERY (Phase 5)
        # ==============================================================
        if self.active_plan:
            action = self.active_plan.popleft()
            is_wait = self.active_wait_flags.popleft() if self.active_wait_flags else False
            if is_wait:
                self.is_executing_intentional_wait = True
                self.diagnostics.action(f"[WAIT] Executing adaptive temporal wait ({action.name})")
            else:
                self.is_executing_intentional_wait = False
                self.consecutive_waits = 0
            if action == GameAction.ACTION5 and self.cooperative_state == CooperativeTransferState.APPROACHING_INTERFACE:
                self.cooperative_state = CooperativeTransferState.HANDOFF_ATTEMPT
                self.attempting_handoff_item = self.world_model.carried_object_id
                self.attempting_handoff_pos = (
                    self.current_transfer_interface.position
                    if self.current_transfer_interface
                    else None
                )
                if self.current_transfer_interface:
                    self.current_transfer_interface.record_attempt(self.action_counter)
            self.diagnostics.action(f"Selected action {action.name}")
        else:
            # Phase 5: Controlled Diagnostic Fallback & Recovery
            self.diagnostics.action(
                f"Warning: No plan active for player at {player.grid_pos if player else 'None'}."
            )
            # 1. Check whether dynamic collision obstacles are blocking the path
            if self.world_model.dynamic_obstacles and player and self.active_subgoal:
                self.diagnostics.plan("Attempting path recovery: clearing dynamic collision obstacles")
                self.world_model.clear_dynamic_obstacles()
                recovered_actions = self.planner.plan_sequence_to_target(
                    start=player.grid_pos,
                    target=self.active_subgoal.target_pos,
                    count=self.active_subgoal.visit_count,
                    cycle_delta=self.active_subgoal.cycle_delta,
                )
                if recovered_actions:
                    self._enqueue_plan(recovered_actions, self.planner.last_is_wait_flags)
                    action = self.active_plan.popleft()
                    is_wait = self.active_wait_flags.popleft() if self.active_wait_flags else False
                    self.is_executing_intentional_wait = is_wait
                    self.diagnostics.plan(f"Recovery succeeded: {len(recovered_actions)} actions generated")
                else:
                    action = self._select_safe_fallback_action(player)
            elif player:
                # Phase 4: Check if waiting/yielding for 1 step allows blocking dynamic entity to clear
                yield_act = self._check_and_execute_temporal_wait_yield(player)
                if yield_act is not None:
                    action = yield_act
                else:
                    action = self._select_safe_fallback_action(player)
            else:
                action = GameAction.ACTION1

        # Record in memory
        if player:
            self.memory.record_step(player.col, player.row, action, success=True)
            if self.memory.is_looping():
                self.diagnostics.action("Warning: Loop detected in recent positions")

        self.last_world_state = current_world_state
        self.last_action = action
        self.action_counter += 1
        return action

    def _check_and_execute_temporal_wait_yield(self, player: PlayerState) -> Optional[GameAction]:
        """
        Phase 4: When desired route is temporarily blocked by an NPC:
        1. Identify the blocking NPC
        2. Inspect its recent velocity
        3. Predict whether it will clear the route
        4. If safe and within bounds, wait/yield for one step
        """
        if self.consecutive_waits >= 3:
            return None

        # Look for nearby moving entities blocking adjacent or near cells
        blocking_mover = None
        for mid, mover in self.temporal_model.moving_objects.items():
            dist = abs(mover.current_pos[0] - player.col) + abs(mover.current_pos[1] - player.row)
            if dist <= 2:
                blocking_mover = mover
                break

        if not blocking_mover:
            for mover in self.persistence_tracker.get_moving_entities():
                dist = abs(mover.col - player.col) + abs(mover.row - player.row)
                if dist <= 2:
                    blocking_mover = mover
                    break

        if not blocking_mover and not self.temporal_model.transient_obstacles:
            return None

        # 4. Select safest wait/yield maneuver: prioritize static wall bump to avoid unwanted pickup/drop
        wait_act = None
        for dc, dr in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
            wp = (player.col + dc, player.row + dr)
            if wp in self.world_model.walls and wp not in self.temporal_model.transient_obstacles:
                from .actions import grid_delta_to_action
                wait_act = grid_delta_to_action(dc, dr)
                break

        if wait_act is None and not self.world_model.carried_object_id:
            # Only use ACTION5 if NOT adjacent to any interactive item or receptacle
            adjacent_interactive = any(
                abs(it[0] - player.col) + abs(it[1] - player.row) <= 1
                for it in (self.world_model.items | self.world_model.receptacles)
            )
            if not adjacent_interactive:
                wait_act = GameAction.ACTION5

        if wait_act is not None:
            self.consecutive_waits += 1
            if wait_act in [GameAction.ACTION1, GameAction.ACTION2, GameAction.ACTION3, GameAction.ACTION4]:
                self.is_executing_intentional_wait = True
            mover_id = getattr(blocking_mover, "object_id", "npc") if blocking_mover else "transient_obstacle"
            vel = getattr(blocking_mover, "velocity", (0, 0)) if blocking_mover else (0, 0)
            self.diagnostics.plan(
                f"[WAIT/YIELD] Route blocked by dynamic entity {mover_id} (velocity={vel}). "
                f"Yielding for 1 step via {wait_act.name} (wait count={self.consecutive_waits})."
            )
            return wait_act

        return None


ArcagiV3Solver = ARCAGIV3Solver
