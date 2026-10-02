from __future__ import annotations

from typing import Dict, Optional, Tuple
from arcengine import GameAction


# Action <-> Discrete Grid Delta (dcol, drow)
# Note: In grid space, col is X, row is Y.
# Up is row - 1, Down is row + 1, Left is col - 1, Right is col + 1.
# ACTION5 represents in-place interaction with zero spatial displacement.
ACTION_TO_GRID_DELTA: Dict[GameAction, Tuple[int, int]] = {
    GameAction.ACTION1: (0, -1),  # Up
    GameAction.ACTION2: (0, 1),   # Down
    GameAction.ACTION3: (-1, 0),  # Left
    GameAction.ACTION4: (1, 0),   # Right
    GameAction.ACTION5: (0, 0),   # Interact / In-place action
}

GRID_DELTA_TO_ACTION: Dict[Tuple[int, int], GameAction] = {
    (0, -1): GameAction.ACTION1,
    (0, 1): GameAction.ACTION2,
    (-1, 0): GameAction.ACTION3,
    (1, 0): GameAction.ACTION4,
    (0, 0): GameAction.ACTION5,
}

# Action <-> Pixel Movement Delta (dx, dy)
ACTION_TO_PIXEL_DELTA: Dict[GameAction, Tuple[int, int]] = {
    GameAction.ACTION1: (0, -5),
    GameAction.ACTION2: (0, 5),
    GameAction.ACTION3: (-5, 0),
    GameAction.ACTION4: (5, 0),
    GameAction.ACTION5: (0, 0),
}


def action_to_grid_delta(action: GameAction) -> Tuple[int, int]:
    """Return (dcol, drow) for given action."""
    return ACTION_TO_GRID_DELTA.get(action, (0, 0))


def grid_delta_to_action(dcol: int, drow: int) -> Optional[GameAction]:
    """Return GameAction corresponding to (dcol, drow)."""
    return GRID_DELTA_TO_ACTION.get((dcol, drow))


def pixel_to_grid(
    x: int,
    y: int,
    stride: int = 5,
    offset_x: int = 4,
    offset_y: int = 0,
) -> Tuple[int, int]:
    """Convert screen pixel coords (x, y) to discrete grid (col, row)."""
    col = (x - offset_x) // stride
    row = (y - offset_y) // stride
    return (col, row)


def grid_to_pixel(
    col: int,
    row: int,
    stride: int = 5,
    offset_x: int = 4,
    offset_y: int = 0,
) -> Tuple[int, int]:
    """Convert discrete grid coords (col, row) to screen pixel coords (x, y)."""
    x = offset_x + col * stride
    y = offset_y + row * stride
    return (x, y)


def is_valid_grid_pos(col: int, row: int, cols: int = 11, rows: int = 11) -> bool:
    """Check if discrete grid coordinate is within playable bounds."""
    return 0 <= col < cols and 0 <= row < rows

