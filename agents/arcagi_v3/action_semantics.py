from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from enum import Enum
import logging
from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np
from arcengine import GameAction

from .objects import GameObject, ObjectType
from .state import HUDState, PlayerState, WorldState

logger = logging.getLogger()


class ActionSemanticLabel(str, Enum):
    """Semantic interpretations of observed action effects."""

    MOVEMENT = "MOVEMENT"
    WAIT = "WAIT"
    PICKUP = "PICKUP"
    DROP = "DROP"
    FOCUS_SWITCH = "FOCUS_SWITCH"
    TOGGLE = "TOGGLE"
    TIME_REPLAY = "TIME_REPLAY"
    ENTITY_CREATION = "ENTITY_CREATION"
    ENTITY_REMOVAL = "ENTITY_REMOVAL"
    ENVIRONMENT_CHANGE = "ENVIRONMENT_CHANGE"
    UNKNOWN = "UNKNOWN"


class ProbeSafety(str, Enum):
    """Safety classification for proactive empirical action probes."""

    SAFE_TO_PROBE = "SAFE_TO_PROBE"
    CAUTION = "CAUTION"
    DO_NOT_PROBE = "DO_NOT_PROBE"


@dataclass
class ActionObservation:
    """
    Empirical evidence extracted from a before/after transition: (S_before, A, S_after).
    Contains raw physical, spatial, and relational deltas without premature categorization.
    """

    action: GameAction
    player_delta: Tuple[int, int] = (0, 0)  # (dcol, drow)
    player_pixel_delta: Tuple[int, int] = (0, 0)  # (dx, dy)
    orientation_delta: int = 0  # rot_idx delta
    time_delta: int = 1  # tick advance
    occupied_cells_delta: int = 0
    objects_appeared: List[str] = field(default_factory=list)
    objects_disappeared: List[str] = field(default_factory=list)
    object_displacements: Dict[str, Tuple[int, int]] = field(default_factory=dict)
    carried_before: Optional[str] = None
    carried_after: Optional[str] = None
    carried_changed: bool = False
    entities_created: List[str] = field(default_factory=list)
    entities_removed: List[str] = field(default_factory=list)
    active_entity_before: Optional[str] = None
    active_entity_after: Optional[str] = None
    focus_switched: bool = False
    environment_pixel_changes: int = 0
    environment_mutated: bool = False
    score_delta: int = 0
    levels_completed_delta: int = 0
    lives_delta: int = 0
    is_reset: bool = False
    is_game_over: bool = False
    step_counter_delta: int = -1
    preconditions: Dict[str, Any] = field(default_factory=dict)
    timestamp: int = 0
    follow_up_frames_count: int = 0


@dataclass
class ActionSemanticProfile:
    """
    Cumulative semantic representation of an action across multiple observations.
    Supports multiple competing or complementary hypotheses with evidence-based confidence.
    """

    action: GameAction
    observations: List[ActionObservation] = field(default_factory=list)
    confidence: Dict[ActionSemanticLabel, float] = field(default_factory=dict)
    classifications: Set[ActionSemanticLabel] = field(default_factory=set)
    preconditions: Dict[str, Any] = field(default_factory=dict)
    reversible: Optional[bool] = None
    time_advance: int = 1
    displacement: Tuple[int, int] = (0, 0)
    object_effects: Dict[str, Any] = field(default_factory=dict)
    entity_effects: Dict[str, Any] = field(default_factory=dict)
    environment_effects: Dict[str, Any] = field(default_factory=dict)
    goal_effects: Dict[str, Any] = field(default_factory=dict)
    safety_rating: ProbeSafety = ProbeSafety.SAFE_TO_PROBE

    def get_confidence(self, label: ActionSemanticLabel) -> float:
        """Return confidence score for a specific semantic label [0.0, 1.0]."""
        return self.confidence.get(label, 0.0)

    def best_hypothesis(self) -> Tuple[ActionSemanticLabel, float]:
        """Return the highest-confidence semantic hypothesis."""
        if not self.confidence:
            return (ActionSemanticLabel.UNKNOWN, 0.0)
        best_lbl = max(self.confidence.keys(), key=lambda l: self.confidence[l])
        return (best_lbl, self.confidence[best_lbl])


class ActionSemanticClassifier:
    """
    Maps empirical ActionObservations to semantic hypotheses with evidence scoring.
    Never relies on game names; reasons purely over physical deltas and object relations.
    """

    @staticmethod
    def classify(observation: ActionObservation) -> Dict[ActionSemanticLabel, float]:
        """
        Evaluate an observation and return a distribution of confidence scores
        for all applicable semantic labels.
        """
        scores: Dict[ActionSemanticLabel, float] = {}

        dcol, drow = observation.player_delta
        has_movement = (dcol != 0 or drow != 0)
        has_pixel_move = observation.player_pixel_delta != (0, 0)

        # 1. MOVEMENT
        if (has_movement or has_pixel_move) and not observation.is_reset:
            # Significant displacement of the avatar
            scores[ActionSemanticLabel.MOVEMENT] = 0.95
        elif has_movement and observation.is_reset:
            scores[ActionSemanticLabel.MOVEMENT] = 0.10

        # 2. WAIT / STATIONARY
        if not has_movement and not has_pixel_move:
            if (
                not observation.carried_changed
                and not observation.entities_created
                and not observation.environment_mutated
                and not observation.is_reset
                and not observation.is_game_over
                and not observation.focus_switched
            ):
                scores[ActionSemanticLabel.WAIT] = 0.90
            elif observation.environment_pixel_changes <= 5 and not observation.carried_changed:
                scores[ActionSemanticLabel.WAIT] = 0.70

        # 3. PICKUP
        if observation.carried_changed:
            if observation.carried_before is None and observation.carried_after is not None:
                scores[ActionSemanticLabel.PICKUP] = 0.95
            elif len(observation.objects_disappeared) > 0 and observation.carried_after is not None:
                scores[ActionSemanticLabel.PICKUP] = 0.90
        elif len(observation.objects_disappeared) > 0 and not has_movement:
            # An adjacent item vanished without carrying flag
            scores[ActionSemanticLabel.PICKUP] = 0.65

        # 4. DROP
        if observation.carried_changed:
            if observation.carried_before is not None and observation.carried_after is None:
                scores[ActionSemanticLabel.DROP] = 0.95
            elif len(observation.objects_appeared) > 0 and observation.carried_before is not None:
                scores[ActionSemanticLabel.DROP] = 0.90
        elif len(observation.objects_appeared) > 0 and not has_movement and observation.carried_before is not None:
            scores[ActionSemanticLabel.DROP] = 0.75

        # 5. FOCUS_SWITCH
        if observation.focus_switched:
            scores[ActionSemanticLabel.FOCUS_SWITCH] = 0.95
        elif (
            observation.active_entity_before is not None
            and observation.active_entity_after is not None
            and observation.active_entity_before != observation.active_entity_after
        ):
            scores[ActionSemanticLabel.FOCUS_SWITCH] = 0.90
        elif not has_movement and observation.environment_pixel_changes in (1, 2, 3) and not observation.carried_changed:
            # A tiny localized selector dot or indicator shifted
            scores[ActionSemanticLabel.FOCUS_SWITCH] = 0.60

        # 6. TIME_REPLAY
        if len(observation.entities_created) > 0 and (
            observation.is_reset or observation.occupied_cells_delta > 5 or observation.follow_up_frames_count > 5
        ):
            scores[ActionSemanticLabel.TIME_REPLAY] = 0.90
        elif observation.follow_up_frames_count > 10 and len(observation.entities_created) > 0:
            scores[ActionSemanticLabel.TIME_REPLAY] = 0.85

        # 7. ENTITY_CREATION
        if len(observation.entities_created) > 0:
            scores[ActionSemanticLabel.ENTITY_CREATION] = 0.92

        # 8. ENTITY_REMOVAL
        if len(observation.entities_removed) > 0:
            scores[ActionSemanticLabel.ENTITY_REMOVAL] = 0.92

        # 9. TOGGLE
        if (
            not has_movement
            and 0 < observation.environment_pixel_changes <= 15
            and not observation.carried_changed
            and not observation.focus_switched
            and not observation.entities_created
        ):
            scores[ActionSemanticLabel.TOGGLE] = 0.80

        # 10. ENVIRONMENT_CHANGE
        if observation.environment_pixel_changes > 15:
            scores[ActionSemanticLabel.ENVIRONMENT_CHANGE] = 0.85

        # 11. UNKNOWN / AMBIGUOUS
        if not scores:
            scores[ActionSemanticLabel.UNKNOWN] = 0.99
        else:
            # If all detected scores are low, include UNKNOWN
            max_s = max(scores.values())
            if max_s < 0.60:
                scores[ActionSemanticLabel.UNKNOWN] = 0.70

        return scores


class ActionProbeEngine:
    """
    Autonomous empirical probe engine.
    Observes state transitions, evaluates probe safety, extracts deltas, and maintains
    probabilistic semantic profiles for all available actions.
    """

    def __init__(self, observation_window: int = 5) -> None:
        self.observation_window: int = observation_window
        self.profiles: Dict[GameAction, ActionSemanticProfile] = {}
        self.frame_history: deque[Tuple[WorldState, Optional[GameAction]]] = deque(
            maxlen=observation_window + 5
        )
        self.step_counter: int = 0
        self.destructive_actions: Set[GameAction] = set()

    def reset_level(self) -> None:
        """Reset history and volatile level observations while preserving learned general profiles."""
        self.frame_history.clear()
        self.step_counter = 0

    def get_profile(self, action: GameAction) -> ActionSemanticProfile:
        """Retrieve or initialize semantic profile for a specific action."""
        if action not in self.profiles:
            self.profiles[action] = ActionSemanticProfile(action=action)
        return self.profiles[action]

    def determine_probe_safety(
        self, action: GameAction, current_state: Optional[WorldState] = None
    ) -> ProbeSafety:
        """
        Evaluate whether an action is safe to proactively probe.
        Checks destructiveness history, step budget urgency, and entity risks.
        """
        if action in self.destructive_actions:
            return ProbeSafety.DO_NOT_PROBE

        profile = self.get_profile(action)

        # Check past observations for catastrophic outcomes
        for obs in profile.observations:
            if obs.is_game_over or obs.lives_delta < 0:
                self.destructive_actions.add(action)
                profile.safety_rating = ProbeSafety.DO_NOT_PROBE
                return ProbeSafety.DO_NOT_PROBE

        # Check step budget urgency
        if current_state and current_state.hud:
            rem_steps = current_state.hud.remaining_steps
            if rem_steps is not None and rem_steps <= 8:
                return ProbeSafety.DO_NOT_PROBE
            if rem_steps is not None and rem_steps <= 15:
                return ProbeSafety.CAUTION

        # If previous observations caused irreversible large environment mutation
        for obs in profile.observations:
            if obs.environment_mutated and not profile.reversible:
                return ProbeSafety.CAUTION

        return ProbeSafety.SAFE_TO_PROBE

    def extract_observation(
        self,
        before_state: WorldState,
        action: GameAction,
        after_state: WorldState,
        follow_up_states: Optional[List[WorldState]] = None,
    ) -> ActionObservation:
        """
        Compute precise state deltas across spatial, inventory, entity, and environment dimensions.
        """
        p_before = before_state.player
        p_after = after_state.player

        # 1. Player spatial deltas
        dcol = 0
        drow = 0
        dx = 0
        dy = 0
        drot = 0
        if p_before and p_after:
            dcol = p_after.col - p_before.col
            drow = p_after.row - p_before.row
            dx = p_after.x - p_before.x
            dy = p_after.y - p_before.y
            drot = (p_after.rot_idx - p_before.rot_idx) % 4

        # 2. Carried / Inventory deltas
        c_before = before_state.carried_object_id
        c_after = after_state.carried_object_id
        carried_changed = c_before != c_after

        # 3. Object presence & displacement
        objs_before = {f"{o.col}_{o.row}": o for o in before_state.detected_objects}
        objs_after = {f"{o.col}_{o.row}": o for o in after_state.detected_objects}

        disappeared = [k for k in objs_before if k not in objs_after]
        appeared = [k for k in objs_after if k not in objs_before]

        displacements: Dict[str, Tuple[int, int]] = {}
        for k_b, o_b in objs_before.items():
            for k_a, o_a in objs_after.items():
                if o_b.object_type == o_a.object_type and (o_b.width, o_b.height) == (o_a.width, o_a.height):
                    if k_b != k_a and k_b in disappeared and k_a in appeared:
                        displacements[f"obj_{o_b.object_type.name}"] = (o_a.col - o_b.col, o_a.row - o_b.row)

        # 4. Entity presence & Focus switching
        ctrl_before = before_state.controlled_entity_id
        ctrl_after = after_state.controlled_entity_id
        focus_switched = False
        if ctrl_before is not None and ctrl_after is not None and ctrl_before != ctrl_after:
            focus_switched = True

        entities_created: List[str] = []
        entities_removed: List[str] = []

        # Check for new controllable or moving entities
        c_entities_before = [o for o in before_state.detected_objects if o.object_type == ObjectType.CONTROLLABLE]
        c_entities_after = [o for o in after_state.detected_objects if o.object_type == ObjectType.CONTROLLABLE]
        if len(c_entities_after) > len(c_entities_before):
            entities_created.append(f"entity_{len(c_entities_after)}")
        elif len(c_entities_after) < len(c_entities_before):
            entities_removed.append(f"entity_{len(c_entities_before)}")

        # 5. Visual environment changes
        screen_b = before_state.screen
        screen_a = after_state.screen
        env_pixels = 0
        if screen_b is not None and screen_a is not None and screen_b.shape == screen_a.shape:
            env_pixels = int(np.sum(screen_b != screen_a))

        # 6. HUD & Goal deltas
        score_delta = 0
        levels_delta = 0
        lives_delta = 0
        step_delta = -1
        if before_state.hud and after_state.hud:
            levels_delta = after_state.hud.levels_completed - before_state.hud.levels_completed
            lives_delta = after_state.hud.remaining_lives - before_state.hud.remaining_lives
            step_delta = after_state.hud.remaining_steps - before_state.hud.remaining_steps

        is_reset = lives_delta < 0 or (p_before and p_after and dcol == 0 and drow == 0 and lives_delta < 0)
        is_game_over = False
        if after_state.hud and after_state.hud.remaining_lives <= 0:
            is_game_over = True

        follow_up_count = len(follow_up_states) if follow_up_states else 0
        if follow_up_states:
            # Check for delayed animation or clone generation in follow-up frames
            for f_st in follow_up_states:
                if f_st.screen is not None and screen_a is not None and f_st.screen.shape == screen_a.shape:
                    env_pixels = max(env_pixels, int(np.sum(screen_a != f_st.screen)))

        self.step_counter += 1

        return ActionObservation(
            action=action,
            player_delta=(dcol, drow),
            player_pixel_delta=(dx, dy),
            orientation_delta=drot,
            time_delta=1 + follow_up_count,
            occupied_cells_delta=len(objs_after) - len(objs_before),
            objects_appeared=appeared,
            objects_disappeared=disappeared,
            object_displacements=displacements,
            carried_before=c_before,
            carried_after=c_after,
            carried_changed=carried_changed,
            entities_created=entities_created,
            entities_removed=entities_removed,
            active_entity_before=ctrl_before,
            active_entity_after=ctrl_after,
            focus_switched=focus_switched,
            environment_pixel_changes=env_pixels,
            environment_mutated=(env_pixels > 20),
            score_delta=score_delta,
            levels_completed_delta=levels_delta,
            lives_delta=lives_delta,
            is_reset=is_reset,
            is_game_over=is_game_over,
            step_counter_delta=step_delta,
            timestamp=self.step_counter,
            follow_up_frames_count=follow_up_count,
        )

    def record_transition(
        self,
        before_state: WorldState,
        action: GameAction,
        after_state: WorldState,
        follow_up_states: Optional[List[WorldState]] = None,
    ) -> ActionSemanticProfile:
        """
        Record a transition, extract observation deltas, accumulate evidence,
        and update the action's semantic profile.
        """
        obs = self.extract_observation(
            before_state=before_state,
            action=action,
            after_state=after_state,
            follow_up_states=follow_up_states,
        )

        profile = self.get_profile(action)
        profile.observations.append(obs)

        # Classify single observation
        single_scores = ActionSemanticClassifier.classify(obs)

        # Accumulate confidence asymptotically: C_{new} = C_{old} + (1 - C_{old}) * score * learning_rate
        lr = 0.40
        for label, score in single_scores.items():
            prev_c = profile.confidence.get(label, 0.0)
            updated_c = prev_c + (1.0 - prev_c) * (score * lr)
            profile.confidence[label] = round(min(1.0, updated_c), 3)

        # Dampen competing hypotheses when evidence contradicts them
        if ActionSemanticLabel.MOVEMENT in single_scores and single_scores[ActionSemanticLabel.MOVEMENT] > 0.8:
            if ActionSemanticLabel.WAIT in profile.confidence:
                profile.confidence[ActionSemanticLabel.WAIT] = round(
                    max(0.0, profile.confidence[ActionSemanticLabel.WAIT] * 0.5), 3
                )

        # Update confirmed classifications (confidence >= 0.50)
        profile.classifications = {
            lbl for lbl, conf in profile.confidence.items() if conf >= 0.50
        }

        # Update physical summary
        profile.displacement = obs.player_delta
        profile.time_advance = obs.time_delta

        # Update safety rating
        if obs.is_game_over or obs.lives_delta < 0:
            profile.safety_rating = ProbeSafety.DO_NOT_PROBE
            self.destructive_actions.add(action)
        elif obs.environment_mutated:
            profile.safety_rating = ProbeSafety.CAUTION

        return profile

    def is_action_type(
        self, action: GameAction, label: ActionSemanticLabel, threshold: float = 0.50
    ) -> bool:
        """Return True if the action has confidence >= threshold for the given semantic label."""
        profile = self.get_profile(action)
        return profile.get_confidence(label) >= threshold
