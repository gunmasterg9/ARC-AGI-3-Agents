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
            blocked = (not moved) and (not level_adv) and (not reset)

        # Check property changes on player
        rot_changed = bool(prev_p and next_p and prev_p.rot_idx != next_p.rot_idx)
        col_changed = bool(prev_p and next_p and prev_p.color_idx != next_p.color_idx)
        shp_changed = bool(prev_p and next_p and prev_p.shape_idx != next_p.shape_idx)

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
        )

        trans = Transition(
            prev_player=prev_p,
            action=action,
            next_player=next_p,
            effect=effect,
        )
        self.transitions.append(trans)
        return trans
