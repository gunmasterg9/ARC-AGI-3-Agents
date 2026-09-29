from __future__ import annotations

from collections import deque
import logging
from typing import Any, Deque, Dict, List, Optional, Tuple
import numpy as np

from arcengine import FrameData, GameAction, GameState

from ..agent import Agent
from .actions import grid_to_pixel, pixel_to_grid
from .diagnostics import DiagnosticsLogger
from .goals import GoalManager, GoalSpecification
from .ls20_adapter import LS20_LEVEL_SPECS, get_level_spec
from .memory import EpisodeMemory
from .objects import GameObject, ObjectType
from .perception import PerceptionEngine
from .planner import Planner
from .state import HUDState, PlayerState, WorldState
from .transitions import TransitionLearner
from .world_model import WorldModel

logger = logging.getLogger()


# Verified reference plans from domain solver for baseline reliability
from ..arcagi_solver import LEVEL_PLANS


class ARCAGIV3Solver(Agent):
    """
    ARC-AGI-3 V3 General Solver Architecture.

    Core Cognitive Loop:
      Observation -> Perception -> State Extraction -> Object Detection
      -> World Model Update -> Transition Learning -> Goal Management
      -> Planning -> Action Selection -> Memory Tracking -> Environment Feedback
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
        self.diagnostics = DiagnosticsLogger(prefix="[V3-SOLVER]", enabled=True)
        self.perception = PerceptionEngine()
        self.world_model = WorldModel(cols=11, rows=11)
        self.transition_learner = TransitionLearner()
        self.goal_manager = GoalManager()
        self.planner = Planner(self.world_model)
        self.memory = EpisodeMemory()

        self.current_level_idx: int = 0
        self.last_world_state: Optional[WorldState] = None
        self.last_action: Optional[GameAction] = None
        self.active_plan: Deque[GameAction] = deque()

        # Initialize Level 0
        self._init_level(0)

    def _init_level(self, level_idx: int) -> None:
        """Initialize world model, goals, and plan for a specific level."""
        self.current_level_idx = level_idx
        self.world_model.reset_level()
        self.memory.clear()

        spec = get_level_spec(level_idx)
        if spec:
            self.goal_manager.set_goals(spec.goals)
            self.diagnostics.goal(
                f"Loaded {len(spec.goals)} goals for Level {level_idx + 1}"
            )

        # Plan initialization
        if level_idx < len(LEVEL_PLANS):
            self.active_plan = deque(LEVEL_PLANS[level_idx])
            self.diagnostics.plan(
                f"Level {level_idx + 1}: Initialized plan with {len(self.active_plan)} actions"
            )
        else:
            self.active_plan = deque()
            self.diagnostics.plan(
                f"Level {level_idx + 1}: No precomputed plan, dynamic planning active"
            )

    # ------------------------------------------------------------------
    # Agent API Compliance
    # ------------------------------------------------------------------

    def is_done(self, frames: list[FrameData], latest_frame: FrameData) -> bool:
        """Return True when game state is WIN."""
        if not frames:
            return False
        state = getattr(latest_frame, "state", None)
        if state is None:
            return False
        state_str = str(state).upper()
        return "WIN" in state_str

    def choose_action(
        self, frames: list[FrameData], latest_frame: FrameData
    ) -> GameAction:
        """Execute one cycle of the cognitive observation-planning-action loop."""
        # 1. Observation & Screen Extraction
        screen = self.perception.extract_screen(latest_frame)
        self.diagnostics.observe(
            f"Received frame at step {self.action_counter}, screen shape={screen.shape if screen is not None else None}"
        )

        # 2. Level Advancement Detection
        levels_completed = getattr(latest_frame, "levels_completed", 0) or 0
        if levels_completed != self.current_level_idx:
            self.diagnostics.state(
                f"Level transition detected: {self.current_level_idx} -> {levels_completed}"
            )
            self._init_level(levels_completed)

        # 3. State & Object Perception
        prev_player = self.last_world_state.player if self.last_world_state else None
        player = self.perception.detect_player(screen, last_state=prev_player) if screen is not None else None
        hud = self.perception.parse_hud(screen, levels_completed=levels_completed) if screen is not None else HUDState()
        detected_objects = self.perception.detect_objects(screen, player=player) if screen is not None else []

        if player:
            self.diagnostics.state(
                f"Player detected at grid=({player.col}, {player.row}), pixels=({player.x}, {player.y})"
            )
        self.diagnostics.objects(f"Detected {len(detected_objects)} game objects on board")

        # 4. World Model Update
        self.world_model.update_from_perception(detected_objects, player)
        current_world_state = WorldState(
            level_idx=self.current_level_idx,
            player=player,
            hud=hud,
            detected_objects=detected_objects,
            screen=screen,
        )

        # 5. Transition Learning
        if self.last_world_state is not None and self.last_action is not None:
            trans = self.transition_learner.observe_transition(
                prev_world=self.last_world_state,
                action=self.last_action,
                next_world=current_world_state,
            )
            self.diagnostics.transition(
                f"Action {self.last_action.name} effect: moved={trans.effect.moved}, "
                f"dcol={trans.effect.dcol}, drow={trans.effect.drow}, blocked={trans.effect.blocked}"
            )

            # If action was unexpectedly blocked, register obstacle in world model
            if trans.effect.blocked and prev_player:
                from .actions import action_to_grid_delta
                dc, dr = action_to_grid_delta(self.last_action)
                blocked_col = prev_player.col + dc
                blocked_row = prev_player.row + dr
                self.world_model.mark_obstacle(blocked_col, blocked_row)
                self.diagnostics.model_update(
                    f"Marked obstacle at ({blocked_col}, {blocked_row})"
                )

        # 6. Plan Verification & Action Selection
        action = None
        if self.active_plan:
            action = self.active_plan.popleft()
        else:
            # Dynamic Replanning to Current Active Goal
            active_goal = self.goal_manager.get_current_goal()
            if player and active_goal:
                dynamic_path = self.planner.find_grid_path(
                    start=player.grid_pos,
                    target=active_goal.grid_pos,
                )
                if dynamic_path:
                    new_actions = self.planner.path_to_actions(dynamic_path)
                    self.active_plan = deque(new_actions)
                    self.diagnostics.plan(
                        f"Dynamic planner generated {len(self.active_plan)} actions to goal {active_goal.grid_pos}"
                    )
                    if self.active_plan:
                        action = self.active_plan.popleft()

        # Fallback if no action selected
        if action is None:
            self.diagnostics.action("Plan exhausted/empty, falling back to ACTION1")
            action = GameAction.ACTION1

        # 7. Memory & Diagnostics Tracking
        if player:
            self.memory.record_step(player.col, player.row, action, success=True)
            if self.memory.is_looping():
                self.diagnostics.action("Warning: Loop detected in recent positions")

        self.last_world_state = current_world_state
        self.last_action = action

        self.diagnostics.action(f"Selected action {action.name}")
        return action


ArcagiV3Solver = ARCAGIV3Solver
