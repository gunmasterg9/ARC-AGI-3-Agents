"""
ARC-AGI-3 V3.9 — Reachability, Connected Components, Transfer Interfaces & Cooperative Handoff.

Provides reusable topological analysis for multi-room environments:
- Connected component decomposition of passable grid
- Direct vs Indirect reachability classification
- Transfer interface discovery (doors, hatches, boundary cells, NPC transfer zones)
- Cooperative NPC classification (helper vs hazard vs obstacle)
- Multi-item handoff lifecycle tracking
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple

from arcengine import GameAction
from .objects import GameObject, ObjectType

logger = logging.getLogger("reachability")


class ReachabilityStatus(str, Enum):
    """Reachability status between player component and goal component."""

    DIRECTLY_REACHABLE = "DIRECTLY_REACHABLE"
    INDIRECTLY_REACHABLE = "INDIRECTLY_REACHABLE"
    UNREACHABLE = "UNREACHABLE"
    UNKNOWN = "UNKNOWN"


class CooperativeTransferState(str, Enum):
    """Generic cooperative interaction state machine."""

    NOT_REQUIRED = "NOT_REQUIRED"
    SEEKING_INTERFACE = "SEEKING_INTERFACE"
    APPROACHING_INTERFACE = "APPROACHING_INTERFACE"
    READY_FOR_HANDOFF = "READY_FOR_HANDOFF"
    HANDOFF_ATTEMPT = "HANDOFF_ATTEMPT"
    HANDOFF_CONFIRMED = "HANDOFF_CONFIRMED"
    HANDOFF_FAILED = "HANDOFF_FAILED"
    ABORTED = "ABORTED"


class NPCClassification(str, Enum):
    """Classification of detected non-player characters."""

    PASSIVE_OBSTACLE = "PASSIVE_OBSTACLE"
    MOVING_HAZARD = "MOVING_HAZARD"
    COMPETING_AGENT = "COMPETING_AGENT"
    HELPER_AGENT = "HELPER_AGENT"
    UNKNOWN = "UNKNOWN"


class ItemTransferState(str, Enum):
    """Lifecycle state of an item in multi-room cooperative environments."""

    UNACQUIRED = "UNACQUIRED"
    CARRIED = "CARRIED"
    AT_TRANSFER = "AT_TRANSFER"
    HANDED_OFF = "HANDED_OFF"
    DELIVERED = "DELIVERED"
    FAILED = "FAILED"


@dataclass
class TransferInterface:
    """Represents a discovered transfer interface between two spatial components."""

    position: Tuple[int, int]  # Grid position of the interface / dropzone cell
    boundary_a: Tuple[int, int]  # Player interaction cell in component A
    boundary_b: Tuple[int, int]  # Recipient interaction cell in component B
    width: int = 1
    passability: bool = False
    interaction_distance: int = 1
    required_action: GameAction = GameAction.ACTION5
    confidence: float = 0.5
    associated_npc: Optional[str] = None
    associated_item: Optional[str] = None
    associated_goal: Optional[Tuple[int, int]] = None
    status: str = "DISCOVERED"
    attempts: int = 0
    successes: int = 0
    failures: int = 0
    last_attempt_step: int = -1

    def record_attempt(self, step: int) -> None:
        self.attempts += 1
        self.last_attempt_step = step

    def record_success(self) -> None:
        self.successes += 1
        self.confidence = min(1.0, self.confidence + 0.2)
        self.status = "VERIFIED"

    def record_failure(self) -> None:
        self.failures += 1
        self.confidence = max(0.1, self.confidence - 0.25)
        if self.failures >= 3 and self.successes == 0:
            self.status = "UNRELIABLE"


class ReachabilityAnalyzer:
    """
    Topological analyzer for grid environments.
    Decomposes the passable grid into connected components and discovers
    inter-component transfer interfaces for cooperative objectives.
    """

    def __init__(self) -> None:
        self.component_map: Dict[Tuple[int, int], int] = {}
        self.components: Dict[int, Set[Tuple[int, int]]] = {}
        self.interfaces: Dict[Tuple[int, int], TransferInterface] = {}
        self.handed_off_items: Set[Tuple[int, int]] = set()
        self.npc_classifications: Dict[str, NPCClassification] = {}
        self.item_lifecycle: Dict[Tuple[int, int], ItemTransferState] = {}

    def reset_level(self) -> None:
        """Reset topological cache on level transition."""
        self.component_map.clear()
        self.components.clear()
        self.interfaces.clear()
        self.handed_off_items.clear()
        self.npc_classifications.clear()
        self.item_lifecycle.clear()

    def record_handoff(self, pos: Tuple[int, int]) -> None:
        """Record an item position as handed off at an interface."""
        self.handed_off_items.add(pos)

    def clear_handoff(self, pos: Tuple[int, int]) -> None:
        """Clear an item position from handed off set."""
        self.handed_off_items.discard(pos)

    def compute_connected_components(
        self,
        cols: int,
        rows: int,
        walls: Set[Tuple[int, int]],
        hazard_cells: Optional[Set[Tuple[int, int]]] = None,
        impassable_items: Optional[Set[Tuple[int, int]]] = None,
    ) -> Dict[int, Set[Tuple[int, int]]]:
        """
        Decomposes all passable grid cells into connected components using BFS flood fill.
        """
        self.component_map.clear()
        self.components.clear()
        hazards = hazard_cells or set()
        items = impassable_items or set()
        impassable = set(walls) | set(hazards) | set(items)

        visited: Set[Tuple[int, int]] = set()
        comp_id = 0

        for r in range(rows):
            for c in range(cols):
                cell = (c, r)
                if cell in impassable or cell in visited:
                    continue

                # Start new component flood fill
                current_comp: Set[Tuple[int, int]] = set()
                queue = deque([cell])
                visited.add(cell)

                while queue:
                    curr = queue.popleft()
                    current_comp.add(curr)
                    self.component_map[curr] = comp_id

                    for dc, dr in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
                        neighbor = (curr[0] + dc, curr[1] + dr)
                        if (
                            0 <= neighbor[0] < cols
                            and 0 <= neighbor[1] < rows
                            and neighbor not in impassable
                            and neighbor not in visited
                        ):
                            visited.add(neighbor)
                            queue.append(neighbor)

                self.components[comp_id] = current_comp
                comp_id += 1

        logger.debug(
            f"[REACHABILITY] Found {len(self.components)} connected components across {cols}x{rows} grid"
        )
        return self.components

    def get_component(self, pos: Tuple[int, int]) -> Optional[int]:
        """Return component ID for a given grid position, or component of adjacent walkable neighbor if pos is on a boundary/wall."""
        if pos in self.component_map:
            return self.component_map[pos]
        # If pos itself is impassable (e.g. wall receptacle, switch, or hatch), check adjacent passable cells
        for dc, dr in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
            nb = (pos[0] + dc, pos[1] + dr)
            if nb in self.component_map:
                return self.component_map[nb]
        return None

    def analyze_reachability(
        self,
        player_pos: Tuple[int, int],
        goal_positions: Set[Tuple[int, int]],
    ) -> Tuple[ReachabilityStatus, Optional[int], Set[int]]:
        """
        Classifies reachability from player to destination goals:
        - DIRECTLY_REACHABLE: At least one goal is in the same connected component.
        - INDIRECTLY_REACHABLE: No goal in player's component, but goals exist in other components.
        - UNREACHABLE: No goals exist anywhere or no components found.
        """
        player_comp = self.get_component(player_pos)
        if player_comp is None:
            return ReachabilityStatus.UNKNOWN, None, set()

        goal_comps = set()
        for g in goal_positions:
            comp = self.get_component(g)
            if comp is not None:
                goal_comps.add(comp)

        if not goal_comps:
            return ReachabilityStatus.UNREACHABLE, player_comp, set()

        if player_comp in goal_comps:
            return ReachabilityStatus.DIRECTLY_REACHABLE, player_comp, goal_comps

        return ReachabilityStatus.INDIRECTLY_REACHABLE, player_comp, goal_comps

    def discover_transfer_interfaces(
        self,
        player_pos: Tuple[int, int],
        target_comps: Set[int],
        cols: int,
        rows: int,
        observed_item_positions: Optional[Set[Tuple[int, int]]] = None,
        npc_positions: Optional[List[Tuple[int, int]]] = None,
    ) -> List[TransferInterface]:
        """
        Discovers candidate transfer interfaces between player's component A and goal components B.
        Examines:
        1. Shared boundary wall cells adjacent to both component A and target components.
        2. Interface cells where items have been observed residing or where NPCs operate.
        """
        player_comp = self.get_component(player_pos)
        if player_comp is None or not target_comps:
            return []

        comp_a_cells = self.components.get(player_comp, set())
        target_cells: Set[Tuple[int, int]] = set()
        for t_comp in target_comps:
            target_cells.update(self.components.get(t_comp, set()))

        observed_items = observed_item_positions or set()
        npcs = npc_positions or []

        # Find cells that bridge Component A and Target Component(s)
        candidate_positions: Set[Tuple[int, int]] = set()
        # Scan cells in and around component A boundaries
        for cell_a in comp_a_cells:
            for dc, dr in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
                cand = (cell_a[0] + dc, cell_a[1] + dr)
                if not (0 <= cand[0] < cols and 0 <= cand[1] < rows):
                    continue
                if cand in comp_a_cells:
                    continue  # Interface must be on the boundary/interface, not inside component A
                # If cand is adjacent to target_cells
                is_adjacent_to_b = any(
                    (cand[0] + d2c, cand[1] + d2r) in target_cells
                    for d2c, d2r in [(0, -1), (0, 1), (-1, 0), (1, 0)]
                )
                if is_adjacent_to_b:
                    candidate_positions.add(cand)

        # Also consider any observed items that sit on boundary cells
        for item_pos in observed_items:
            adj_a = any(
                (item_pos[0] + dc, item_pos[1] + dr) in comp_a_cells
                for dc, dr in [(0, -1), (0, 1), (-1, 0), (1, 0)]
            )
            adj_b = any(
                (item_pos[0] + dc, item_pos[1] + dr) in target_cells
                for dc, dr in [(0, -1), (0, 1), (-1, 0), (1, 0)]
            )
            if adj_a and adj_b:
                candidate_positions.add(item_pos)

        # Build or update TransferInterface objects
        discovered: List[TransferInterface] = []
        for pos in candidate_positions:
            # Find best boundary_a (walkable in comp_a) and boundary_b (walkable in target)
            neighbors_a = [
                (pos[0] + dc, pos[1] + dr)
                for dc, dr in [(0, -1), (0, 1), (-1, 0), (1, 0)]
                if (pos[0] + dc, pos[1] + dr) in comp_a_cells
            ]
            neighbors_b = [
                (pos[0] + dc, pos[1] + dr)
                for dc, dr in [(0, -1), (0, 1), (-1, 0), (1, 0)]
                if (pos[0] + dc, pos[1] + dr) in target_cells
            ]

            if not neighbors_a or not neighbors_b:
                continue

            # Closest neighbor in A to player
            best_a = min(neighbors_a, key=lambda n: abs(n[0] - player_pos[0]) + abs(n[1] - player_pos[1]))
            best_b = neighbors_b[0]

            # Base confidence calculation
            confidence = 0.5
            # High confidence if an item was observed placed at this interface
            if pos in observed_items:
                confidence = max(confidence, 0.95)
            # Confidence boost if an NPC in target component is close to boundary_b
            if npcs:
                min_npc_dist = min(abs(npc[0] - best_b[0]) + abs(npc[1] - best_b[1]) for npc in npcs)
                if min_npc_dist <= 2:
                    confidence = min(1.0, confidence + 0.2)

            if pos in self.interfaces:
                interface = self.interfaces[pos]
                interface.boundary_a = best_a
                interface.boundary_b = best_b
                # Preserve learned confidence unless initial item observed
                if pos in observed_items:
                    interface.confidence = max(interface.confidence, 0.95)
            else:
                interface = TransferInterface(
                    position=pos,
                    boundary_a=best_a,
                    boundary_b=best_b,
                    confidence=confidence,
                )
                self.interfaces[pos] = interface

            discovered.append(interface)

        # Score and sort interfaces
        def score_interface(tif: TransferInterface) -> float:
            dist_to_player = abs(tif.boundary_a[0] - player_pos[0]) + abs(tif.boundary_a[1] - player_pos[1])
            # Higher score = better. High confidence and low distance preferred.
            return tif.confidence * 100.0 - dist_to_player

        discovered.sort(key=score_interface, reverse=True)
        return discovered

    def classify_npc(
        self,
        npc_id: str,
        positions_history: List[Tuple[int, int]],
        items_delivered_count: int,
        interacted_with_items: bool = False,
    ) -> NPCClassification:
        """
        Classify NPC role based on observed behavioral history.
        """
        if items_delivered_count > 0 or interacted_with_items:
            cls = NPCClassification.HELPER_AGENT
        elif len(set(positions_history)) > 1:
            cls = NPCClassification.MOVING_HAZARD
        else:
            cls = NPCClassification.PASSIVE_OBSTACLE

        self.npc_classifications[npc_id] = cls
        return cls
