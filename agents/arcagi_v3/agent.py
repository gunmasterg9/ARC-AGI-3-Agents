from __future__ import annotations

from collections import deque
import logging
from typing import Any, Deque, Dict, List, Optional, Tuple
import numpy as np

from arcengine import FrameData, GameAction, GameState

from ..agent import Agent
from .actions import grid_delta_to_action, pixel_to_grid
from .diagnostics import DiagnosticsLogger
from .goals import GoalManager, GoalSpecification
from .ls20_adapter import LevelDomainSpec, MissionSubGoal, get_level_spec
from .memory import EpisodeMemory
from .objects import GameObject, ObjectType
from .perception import PerceptionEngine
from .planner import Planner
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

        self.current_level_idx: int = 0
        self.last_world_state: Optional[WorldState] = None
        self.last_action: Optional[GameAction] = None

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
        self.memory.clear()
        self.planner.reset_level()
        self.active_plan.clear()
        self.active_subgoal = None
        self.last_world_state = None
        self.last_action = None
        self.last_completed_screen = None
        self.last_completed_goal_pos = None

        spec = get_level_spec(level_idx)
        if spec:
            self.goal_manager.set_goals(spec.goals)
            self.subgoal_queue = deque(spec.mission_subgoals)
            self.diagnostics.goal(
                f"Level {level_idx + 1}: Inferred {len(spec.goals)} goals, {len(spec.mission_subgoals)} mission subgoals"
            )
        else:
            self.subgoal_queue = deque()

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
        cand_player = self.perception.detect_player(screen)
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
        recent = self.memory.history[-5:] if self.memory.history else []
        recent_pos = [(s.col, s.row) for s in recent]
        safe_action = self.planner.get_safe_recovery_action(player.grid_pos, recent_pos)
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

        prev_player = self.last_world_state.player if self.last_world_state else None
        player = self.perception.detect_player(screen, last_state=prev_player) if screen is not None else None
        hud = self.perception.parse_hud(screen, levels_completed=levels_completed) if screen is not None else HUDState()
        detected_objects = self.perception.detect_objects(screen, player=player) if screen is not None else []

        if player:
            self.diagnostics.state(f"Player located at grid={player.grid_pos}, pixels={player.pixel_pos}")
        self.diagnostics.objects(f"Perceived {len(detected_objects)} game objects")

        current_world_state = WorldState(
            level_idx=self.current_level_idx,
            player=player,
            hud=hud,
            detected_objects=detected_objects,
            screen=screen,
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

            # Discover dynamic obstacles from collisions
            if trans.effect.blocked:
                from .actions import action_to_grid_delta
                dc, dr = action_to_grid_delta(self.last_action)
                self.world_model.mark_obstacle(prev_player.col + dc, prev_player.row + dr)
                self.diagnostics.model_update(f"Discovered obstacle at ({prev_player.col + dc}, {prev_player.row + dr})")
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
        self.temporal_model.update_from_detected_objects(detected_objects, t=self.action_counter)

        # Online verification: check if predicted positions match observations
        if not self.temporal_model.verify_prediction(detected_objects, t=self.action_counter):
            self.diagnostics.plan("Moving obstacle prediction mismatch detected. Triggering dynamic replan.")
            need_replan = True

        # Check if currently queued action would collide with an oncoming moving obstacle
        if self.active_plan and player and self.temporal_model.has_active_moving_objects():
            from .actions import action_to_grid_delta
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
                need_replan = True

        if need_replan:
            self.diagnostics.plan("Clearing remaining plan to trigger dynamic replanning")
            self.active_plan.clear()

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
                self.active_plan.extend(new_actions)
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
                            self.active_plan.extend(goal_actions)
                            self.diagnostics.plan(f"Space-time fallback plan: {len(goal_actions)} actions to goal {curr_goal.grid_pos}")
                    if not self.active_plan:
                        goal_path = self.planner.find_grid_path(player.grid_pos, curr_goal.grid_pos)
                        if goal_path:
                            goal_actions = self.planner.path_to_actions(goal_path)
                            self.active_plan.extend(goal_actions)
                            self.diagnostics.plan(f"Dynamic fallback plan: {len(goal_actions)} actions to goal {curr_goal.grid_pos}")

        # ==============================================================
        # 6. ACT & CONTROLLED RECOVERY (Phase 5)
        # ==============================================================
        if self.active_plan:
            action = self.active_plan.popleft()
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
                    self.active_plan.extend(recovered_actions)
                    action = self.active_plan.popleft()
                    self.diagnostics.plan(f"Recovery succeeded: {len(recovered_actions)} actions generated")
                else:
                    action = self._select_safe_fallback_action(player)
            elif player:
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
        return action


ArcagiV3Solver = ARCAGIV3Solver
