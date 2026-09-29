from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, List, Optional, Set, Tuple
from arcengine import GameAction


@dataclass
class MemoryEntry:
    col: int
    row: int
    action: GameAction
    success: bool = True


class EpisodeMemory:
    """Short-term episode memory and loop prevention."""

    def __init__(self, history_capacity: int = 20) -> None:
        self.history: Deque[MemoryEntry] = deque(maxlen=history_capacity)
        self.position_history: Deque[Tuple[int, int]] = deque(maxlen=history_capacity)
        self.failed_positions: Set[Tuple[int, int]] = set()

    def record_step(self, col: int, row: int, action: GameAction, success: bool = True) -> None:
        entry = MemoryEntry(col=col, row=row, action=action, success=success)
        self.history.append(entry)
        self.position_history.append((col, row))
        if not success:
            self.failed_positions.add((col, row))

    def is_looping(self) -> bool:
        """Detect if agent is stuck cycling in a small local loop (e.g. A-B-A-B)."""
        if len(self.position_history) < 6:
            return False
        recent = list(self.position_history)[-6:]
        # Check if only 2 unique positions in last 6 steps
        return len(set(recent)) <= 2

    def is_stuck(self) -> bool:
        """Detect if agent has not changed position for 3 consecutive steps."""
        if len(self.position_history) < 3:
            return False
        recent = list(self.position_history)[-3:]
        return len(set(recent)) == 1

    def clear(self) -> None:
        self.history.clear()
        self.position_history.clear()
        self.failed_positions.clear()
