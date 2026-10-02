from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple
from arcengine import GameAction

from .objects import ObjectType
from .state import PlayerState, WorldState


@dataclass
class ActionEffect:
    """Observed consequence of an executed action."""

    moved: bool
    dcol: int
    drow: int
    blocked: bool
    reset_occurred: bool
    level_advanced: bool
    stepped_on: Optional[ObjectType] = None
    shape_changed: bool = False
    color_changed: bool = False
    rotation_changed: bool = False
    carried_changed: bool = False
    carried_object_id: Optional[str] = None
    active_entity_changed: bool = False
    interacted_object_id: Optional[str] = None
    interaction_type: Optional[str] = None


@dataclass
class Transition:
    """Historical tuple of (state_before, action, state_after, effect)."""

    prev_player: Optional[PlayerState]
    action: GameAction
    next_player: Optional[PlayerState]
    effect: ActionEffect
    confidence: float = 1.0


class TransitionLearner:
    """Learns action-effect dynamics, collision rules, and transformer behaviors."""

    def __init__(self) -> None:
        self.transitions: list[Transition] = []
        # Learned mapping of (col, row) -> effect type
        self.cell_effects: Dict[Tuple[int, int], ObjectType] = {}
        # Learned pusher movements: (col, row) -> (dcol, drow)
        self.pusher_deltas: Dict[Tuple[int, int], Tuple[int, int]] = {}

    def reset_level(self) -> None:
        """Reset learned level-local transitions, cell effects, and pusher mechanics."""
        self.transitions.clear()
        self.cell_effects.clear()
        self.pusher_deltas.clear()

    def observe_transition(
        self,
        prev_world: WorldState,
        action: GameAction,
        next_world: WorldState,
    ) -> Transition:
        """Compare pre- and post-action world states to determine empirical effects."""
        prev_p = prev_world.player
        next_p = next_world.player

        level_adv = next_world.level_idx > prev_world.level_idx
        reset = bool(
            prev_p
            and next_p
            and (next_world.hud.remaining_lives < prev_world.hud.remaining_lives)
        )

        dcol = 0
        drow = 0
        moved = False
        blocked = False

        if prev_p and next_p:
            dcol = next_p.col - prev_p.col
            drow = next_p.row - prev_p.row
            moved = (dcol != 0 or drow != 0)
            if action == GameAction.ACTION5:
                # ACTION5 does not displace spatial position unless observed
                blocked = False
            else:
                blocked = (not moved) and (not level_adv) and (not reset)

        # Check property changes on player
        rot_changed = bool(prev_p and next_p and prev_p.rot_idx != next_p.rot_idx)
        col_changed = bool(prev_p and next_p and prev_p.color_idx != next_p.color_idx)
        shp_changed = bool(prev_p and next_p and prev_p.shape_idx != next_p.shape_idx)

        # Interaction effects for ACTION5 or object state transitions
        carried_changed = False
        carried_obj_id = None
        active_ent_changed = False
        interaction_type = None

        prev_carried = getattr(prev_world, "carried_object_id", None)
        next_carried = getattr(next_world, "carried_object_id", None)
        if prev_carried != next_carried:
            carried_changed = True
            carried_obj_id = next_carried
            interaction_type = "pickup" if next_carried is not None else "drop"

        prev_ctrl = getattr(prev_world, "controlled_entity_id", None)
        next_ctrl = getattr(next_world, "controlled_entity_id", None)
        if prev_ctrl is not None and next_ctrl is not None and prev_ctrl != next_ctrl:
            active_ent_changed = True
            interaction_type = "switch_focus"

        if action == GameAction.ACTION5 and not interaction_type:
            if len(next_world.detected_objects) < len(prev_world.detected_objects):
                carried_changed = True
                interaction_type = "pickup"
            elif len(next_world.detected_objects) > len(prev_world.detected_objects):
                carried_changed = True
                interaction_type = "drop"
            else:
                interaction_type = "interact"

        stepped_on = None
        if next_p:
            pos = (next_p.col, next_p.row)
            if pos in self.cell_effects:
                stepped_on = self.cell_effects[pos]
            if rot_changed:
                self.cell_effects[pos] = ObjectType.TRANSFORMER_ROTATION
                stepped_on = ObjectType.TRANSFORMER_ROTATION
            elif col_changed:
                self.cell_effects[pos] = ObjectType.TRANSFORMER_COLOR
                stepped_on = ObjectType.TRANSFORMER_COLOR
            elif shp_changed:
                self.cell_effects[pos] = ObjectType.TRANSFORMER_SHAPE
                stepped_on = ObjectType.TRANSFORMER_SHAPE

        effect = ActionEffect(
            moved=moved,
            dcol=dcol,
            drow=drow,
            blocked=blocked,
            reset_occurred=reset,
            level_advanced=level_adv,
            stepped_on=stepped_on,
            shape_changed=shp_changed,
            color_changed=col_changed,
            rotation_changed=rot_changed,
            carried_changed=carried_changed,
            carried_object_id=carried_obj_id,
            active_entity_changed=active_ent_changed,
            interaction_type=interaction_type,
        )

        trans = Transition(
            prev_player=prev_p,
            action=action,
            next_player=next_p,
            effect=effect,
        )
        self.transitions.append(trans)
        return trans
