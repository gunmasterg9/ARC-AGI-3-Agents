from __future__ import annotations

from typing import Dict, Optional, Tuple
from arcengine import GameAction


# Action <-> Discrete Grid Delta (dcol, drow)
# Note: In grid space, col is X (0..10), row is Y (0..10).
# Up is row - 1, Down is row + 1, Left is col - 1, Right is col + 1.
ACTION_TO_GRID_DELTA: Dict[GameAction, Tuple[int, int]] = {
    GameAction.ACTION1: (0, -1),  # Up
    GameAction.ACTION2: (0, 1),   # Down
    GameAction.ACTION3: (-1, 0),  # Left
    GameAction.ACTION4: (1, 0),   # Right
}

GRID_DELTA_TO_ACTION: Dict[Tuple[int, int], GameAction] = {
    (0, -1): GameAction.ACTION1,
    (0, 1): GameAction.ACTION2,
    (-1, 0): GameAction.ACTION3,
    (1, 0): GameAction.ACTION4,
}

# Action <-> Pixel Movement Delta (dx, dy)
ACTION_TO_PIXEL_DELTA: Dict[GameAction, Tuple[int, int]] = {
    GameAction.ACTION1: (0, -5),
    GameAction.ACTION2: (0, 5),
    GameAction.ACTION3: (-5, 0),
    GameAction.ACTION4: (5, 0),
}


def action_to_grid_delta(action: GameAction) -> Tuple[int, int]:
    """Return (dcol, drow) for given action."""
    return ACTION_TO_GRID_DELTA.get(action, (0, 0))


def grid_delta_to_action(dcol: int, drow: int) -> Optional[GameAction]:
    """Return GameAction corresponding to (dcol, drow)."""
    return GRID_DELTA_TO_ACTION.get((dcol, drow))


def pixel_to_grid(x: int, y: int) -> Tuple[int, int]:
    """Convert screen pixel coords (x, y) to discrete grid (col, row)."""
    col = (x - 4) // 5
    row = y // 5
    return (col, row)


def grid_to_pixel(col: int, row: int) -> Tuple[int, int]:
    """Convert discrete grid coords (col, row) to screen pixel coords (x, y)."""
    x = 4 + col * 5
    y = row * 5
    return (x, y)


def is_valid_grid_pos(col: int, row: int) -> bool:
    """Check if discrete grid coordinate is within playable bounds."""
    return 0 <= col <= 10 and 0 <= row <= 10
