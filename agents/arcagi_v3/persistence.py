from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import logging
import math
from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np
from arcengine import GameAction

from .objects import GameObject, ObjectType

logger = logging.getLogger()


@dataclass
class TrackedObject:
    """Represents an object tracked across multiple consecutive visual frames."""

    object_id: str
    current_pos: Tuple[int, int]
    pixel_pos: Tuple[int, int]
    width: int
    height: int
    dominant_color: Optional[int] = None
    color_distribution: Dict[int, int] = field(default_factory=dict)
    positions_history: List[Tuple[int, int]] = field(default_factory=list)
    motion_state: str = "static"  # "static", "moving", "carried"
    velocity: Tuple[int, int] = (0, 0)
    persistence_count: int = 1
    last_seen_t: int = 0
    disappeared_count: int = 0
    item_confidence: float = 0.5
    receptacle_confidence: float = 0.5
    npc_confidence: float = 0.0
    wall_confidence: float = 0.0
    category: ObjectType = ObjectType.UNKNOWN
    is_carried: bool = False

    @property
    def col(self) -> int:
        return self.current_pos[0]

    @property
    def row(self) -> int:
        return self.current_pos[1]


class MultiFrameObjectTracker:
    """
    Maintains temporal associations between visual frames to reliably distinguish
    player, items, receptacles/drop zones, moving NPCs, and static walls.
    """

    MAX_HISTORY: int = 3
    DISAPPEAR_TIMEOUT: int = 4

    def __init__(self, stride: int = 4, offset_x: int = 0, offset_y: int = 0) -> None:
        self.stride: int = stride
        self.offset_x: int = offset_x
        self.offset_y: int = offset_y
        self.tracked_objects: Dict[str, TrackedObject] = {}
        self.frame_history: Deque[np.ndarray] = deque(maxlen=self.MAX_HISTORY)
        self.time_counter: int = 0
        self.carried_object_id: Optional[str] = None
        self.carried_offset: Optional[Tuple[int, int]] = None
        self.delivered_items: Set[Tuple[int, int]] = set()
        self.next_object_num: int = 1
        self.pickups_count: int = 0
        self.drops_count: int = 0
        self.failed_interactions_count: int = 0

    def configure_grid(self, stride: int = 4, offset_x: int = 0, offset_y: int = 0) -> None:
        """Configure grid geometric parameters."""
        self.stride = stride
        self.offset_x = offset_x
        self.offset_y = offset_y

    def reset_level(self) -> None:
        """Reset object tracking state between levels."""
        self.tracked_objects.clear()
        self.frame_history.clear()
        self.time_counter = 0
        self.carried_object_id = None
        self.carried_offset = None
        self.delivered_items.clear()
        self.next_object_num = 1
        self.pickups_count = 0
        self.drops_count = 0
        self.failed_interactions_count = 0

    def update(
        self,
        detected_objects: List[GameObject],
        current_frame: Optional[np.ndarray],
        player_pos: Optional[Tuple[int, int]],
        last_action: Optional[GameAction],
        prev_frame: Optional[np.ndarray] = None,
        carried_offset: Optional[Tuple[int, int]] = None,
    ) -> List[TrackedObject]:
        """
        Process current frame detections, update temporal tracking, and classify objects.
        """
        self.time_counter += 1
        if current_frame is not None:
            self.frame_history.append(current_frame.copy())

        # 1. Handle ACTION5 pickup/drop state transition first
        if last_action == GameAction.ACTION5 and player_pos is not None:
            self._handle_action5_transition(
                detected_objects,
                player_pos,
                prev_frame=prev_frame,
                current_frame=current_frame,
                carried_offset=carried_offset or self.carried_offset,
            )

        # 2. If carrying an object, update its relative attachment position
        effective_offset = carried_offset or self.carried_offset
        carried_pos = None
        carried_track = None
        if self.carried_object_id is not None:
            carried_track = self.tracked_objects.get(self.carried_object_id)
            if carried_track:
                carried_track.is_carried = True
                carried_track.motion_state = "carried"
                carried_track.category = ObjectType.ITEM
                if effective_offset is not None and player_pos is not None:
                    carried_pos = (player_pos[0] + effective_offset[0], player_pos[1] + effective_offset[1])
                    carried_track.current_pos = carried_pos
                    carried_track.pixel_pos = (
                        self.offset_x + carried_pos[0] * self.stride,
                        self.offset_y + carried_pos[1] * self.stride,
                    )
                    carried_track.positions_history.append(carried_pos)

        # 3. Associate incoming detections with existing tracked objects
        unmatched_detections = []
        matched_track_ids: Set[str] = set()

        for det in detected_objects:
            # Skip candidate if it matches the player position
            if player_pos and (det.col, det.row) == player_pos:
                continue

            # Skip candidate if it is already delivered
            if (det.col, det.row) in self.delivered_items:
                continue

            # If detection matches carried object position, associate directly to preserve visual tracking
            if carried_track is not None and (det.col, det.row) == carried_track.current_pos:
                carried_track.pixel_pos = (det.x, det.y)
                carried_track.dominant_color = det.color
                carried_track.last_seen_t = self.time_counter
                matched_track_ids.add(self.carried_object_id)
                continue

            best_match_id = None
            best_cost = float("inf")

            for obj_id, track in self.tracked_objects.items():
                if obj_id in matched_track_ids or track.is_carried:
                    continue

                dist = abs(det.col - track.current_pos[0]) + abs(det.row - track.current_pos[1])
                color_diff = 0 if det.color == track.dominant_color else 2

                # Static walls and receptacles never move on their own:
                # only match at exact same cell (dist == 0) and same color
                is_stationary = (
                    det.object_type in (ObjectType.WALL, ObjectType.RECEPTACLE)
                    or track.category in (ObjectType.WALL, ObjectType.RECEPTACLE)
                    or track.receptacle_confidence >= 0.5
                    or track.wall_confidence >= 0.5
                )
                if is_stationary:
                    if dist > 0 or det.color != track.dominant_color:
                        continue

                cost = dist + color_diff
                if dist <= 2 and cost < best_cost:
                    best_cost = cost
                    best_match_id = obj_id

            if best_match_id is not None and best_cost <= 3:
                # Update existing track
                track = self.tracked_objects[best_match_id]
                old_pos = track.current_pos
                new_pos = (det.col, det.row)
                vel = (new_pos[0] - old_pos[0], new_pos[1] - old_pos[1])

                track.current_pos = new_pos
                track.pixel_pos = (det.x, det.y)
                track.velocity = vel
                track.positions_history.append(new_pos)
                track.persistence_count += 1
                track.last_seen_t = self.time_counter
                track.disappeared_count = 0
                track.dominant_color = det.color
                matched_track_ids.add(best_match_id)

                # Motion state tracking
                if det.object_type == ObjectType.WALL or track.category == ObjectType.WALL:
                    track.category = ObjectType.WALL
                    track.wall_confidence = 1.0
                    track.npc_confidence = 0.0
                    track.motion_state = "static"
                    track.velocity = (0, 0)
                elif det.object_type == ObjectType.RECEPTACLE or track.category == ObjectType.RECEPTACLE or track.receptacle_confidence >= 0.5:
                    track.category = ObjectType.RECEPTACLE
                    track.receptacle_confidence = 1.0
                    track.npc_confidence = 0.0
                    track.motion_state = "static"
                    track.velocity = (0, 0)
                elif track.is_carried:
                    track.motion_state = "carried"
                elif vel != (0, 0):
                    track.motion_state = "moving"
                    track.category = ObjectType.MOVING_PLATFORM
                    track.npc_confidence = min(1.0, track.npc_confidence + 0.3)
                    track.item_confidence = max(0.0, track.item_confidence - 0.2)
                    track.receptacle_confidence = max(0.0, track.receptacle_confidence - 0.3)
                else:
                    track.motion_state = "static"
            else:
                unmatched_detections.append(det)

        # Handle disappearance of unmatched tracks
        for obj_id, track in list(self.tracked_objects.items()):
            if obj_id not in matched_track_ids and not track.is_carried:
                # Static receptacles and walls never disappear even if temporarily occluded by an NPC or item
                if (
                    track.category in (ObjectType.RECEPTACLE, ObjectType.WALL)
                    or track.receptacle_confidence >= 0.5
                    or track.wall_confidence >= 0.5
                ):
                    continue
                track.disappeared_count += 1
                if track.disappeared_count > self.DISAPPEAR_TIMEOUT:
                    # Only remove if not carried
                    logger.debug(f"[TRACKER] Pruning disappeared track {obj_id} after {track.disappeared_count} steps")
                    del self.tracked_objects[obj_id]

        # Spawn new tracks for unmatched detections
        for det in unmatched_detections:
            new_id = f"obj_{self.next_object_num}"
            self.next_object_num += 1

            new_track = TrackedObject(
                object_id=new_id,
                current_pos=(det.col, det.row),
                pixel_pos=(det.x, det.y),
                width=det.width,
                height=det.height,
                dominant_color=det.color,
                positions_history=[(det.col, det.row)],
                last_seen_t=self.time_counter,
                category=det.object_type,
            )
            self._initial_classify(new_track, det)
            self.tracked_objects[new_id] = new_track

        # Update categories based on multi-frame evidence
        self._refine_classifications()

        # Synchronize delivered items: any uncarried item sitting on a receptacle cell
        receptacle_cells = {
            t.current_pos for t in self.tracked_objects.values()
            if t.category == ObjectType.RECEPTACLE or t.receptacle_confidence >= 0.5
        }
        for det in detected_objects:
            if (
                det.object_type == ObjectType.ITEM
                and (det.col, det.row) in receptacle_cells
                and (carried_pos is None or (det.col, det.row) != carried_pos)
            ):
                self.delivered_items.add((det.col, det.row))
        for track in self.tracked_objects.values():
            if not track.is_carried and track.category == ObjectType.ITEM and track.current_pos in receptacle_cells:
                self.delivered_items.add(track.current_pos)

        return list(self.tracked_objects.values())

    def _initial_classify(self, track: TrackedObject, det: GameObject) -> None:
        """Heuristic initial classification based on static visual hints."""
        if det.object_type == ObjectType.WALL:
            track.wall_confidence = 0.9
            track.item_confidence = 0.0
            track.receptacle_confidence = 0.0
            track.category = ObjectType.WALL
        elif det.object_type == ObjectType.RECEPTACLE:
            track.receptacle_confidence = 0.8
            track.item_confidence = 0.2
            track.category = ObjectType.RECEPTACLE
        elif det.object_type == ObjectType.ITEM:
            track.item_confidence = 0.8
            track.receptacle_confidence = 0.2
            track.category = ObjectType.ITEM
        else:
            # By default, smaller single-cell objects are candidate items,
            # while multi-cell clusters or border shapes are candidate receptacles
            track.item_confidence = 0.6
            track.receptacle_confidence = 0.4
            track.category = ObjectType.ITEM

    def _refine_classifications(self) -> None:
        """Refine item vs receptacle vs moving NPC classification using temporal history."""
        # Detect connected receptacle clusters (zones that share boundary or multi-cell structure)
        receptacle_cells = set()
        for track in self.tracked_objects.values():
            if track.receptacle_confidence >= 0.5 and track.motion_state == "static":
                receptacle_cells.add(track.current_pos)

        for track in self.tracked_objects.values():
            if track.is_carried:
                track.category = ObjectType.ITEM
                track.item_confidence = 1.0
                continue

            if track.category == ObjectType.WALL or track.wall_confidence >= 0.8:
                continue

            # Moving objects cannot be static items or receptacles
            if track.npc_confidence >= 0.5 or (len(set(track.positions_history[-3:])) > 1 and track.wall_confidence < 0.5):
                track.category = ObjectType.MOVING_PLATFORM
                track.item_confidence = 0.0
                track.receptacle_confidence = 0.0
                continue

            # Delivered items resting on receptacles must retain ITEM classification
            if track.current_pos in self.delivered_items and track.item_confidence > track.receptacle_confidence:
                track.category = ObjectType.ITEM
                track.item_confidence = 0.95
                track.receptacle_confidence = 0.0
                continue

            # If stationary across multiple observations, check if it's adjacent to other receptacle cells
            if track.persistence_count >= 2:
                # Check spatial clustering with other receptacle-like cells
                has_receptacle_neighbor = any(
                    abs(track.col - rc[0]) + abs(track.row - rc[1]) == 1
                    for rc in receptacle_cells
                    if rc != track.current_pos
                )
                if has_receptacle_neighbor:
                    track.receptacle_confidence = min(1.0, track.receptacle_confidence + 0.25)
                    track.item_confidence = max(0.0, track.item_confidence - 0.25)

            if track.item_confidence >= track.receptacle_confidence:
                track.category = ObjectType.ITEM
            else:
                track.category = ObjectType.RECEPTACLE

    def _handle_action5_transition(
        self,
        current_detections: List[GameObject],
        player_pos: Tuple[int, int],
        prev_frame: Optional[np.ndarray] = None,
        current_frame: Optional[np.ndarray] = None,
        carried_offset: Optional[Tuple[int, int]] = None,
    ) -> None:
        """
        Evaluate before/after state when ACTION5 was executed at player_pos.
        Learns pickup or drop dynamically from visual diffs or detection changes.
        """
        curr_positions = {(d.col, d.row) for d in current_detections}

        if self.carried_object_id is None:
            # Look for an adjacent item (dist <= 1)
            candidates = []
            for obj_id, track in list(self.tracked_objects.items()):
                if (
                    track.is_carried
                    or track.category in (ObjectType.WALL, ObjectType.RECEPTACLE)
                    or track.receptacle_confidence >= 0.5
                    or track.current_pos in self.delivered_items
                ):
                    continue
                dist = abs(track.col - player_pos[0]) + abs(track.row - player_pos[1])
                if dist <= 1:
                    candidates.append((dist, obj_id, track))

            candidates.sort(key=lambda c: c[0])
            for _, obj_id, track in candidates:
                disappeared = track.current_pos not in curr_positions
                visual_change = False
                if prev_frame is not None and current_frame is not None:
                    px = self.offset_x + track.col * self.stride
                    py = self.offset_y + track.row * self.stride
                    if px + self.stride <= current_frame.shape[1] and py + self.stride <= current_frame.shape[0]:
                        visual_change = not np.array_equal(
                            prev_frame[py : py + self.stride, px : px + self.stride],
                            current_frame[py : py + self.stride, px : px + self.stride],
                        )

                if disappeared or visual_change:
                    self.carried_object_id = obj_id
                    self.carried_offset = (track.col - player_pos[0], track.row - player_pos[1])
                    track.is_carried = True
                    track.item_confidence = 1.0
                    track.category = ObjectType.ITEM
                    self.pickups_count += 1
                    logger.info(
                        f"[TRACKER] Confirmed PICKUP of {obj_id} at {track.current_pos} with offset {self.carried_offset} (total pickups: {self.pickups_count})"
                    )
                    return
        else:
            # Carrying an item: check if item dropped
            carried_track = self.tracked_objects.get(self.carried_object_id)
            receptacles = self.get_receptacles()
            receptacle_cells = {t.current_pos for t in receptacles}

            drop_pos = None
            if carried_offset is not None:
                drop_pos = (player_pos[0] + carried_offset[0], player_pos[1] + carried_offset[1])
            elif carried_track:
                drop_pos = carried_track.current_pos

            # Check if dropped in or near a receptacle
            # Check if dropped in a receptacle
            if drop_pos and drop_pos in receptacle_cells:
                if carried_track:
                    carried_track.is_carried = False
                    carried_track.current_pos = drop_pos
                    carried_track.last_seen_t = self.time_counter
                self.delivered_items.add(drop_pos)
                self.carried_object_id = None
                self.carried_offset = None
                self.drops_count += 1
                logger.info(
                    f"[TRACKER] Confirmed DELIVERY of item into receptacle at {drop_pos} (total drops: {self.drops_count})"
                )
                return

            for d in current_detections:
                dist = abs(d.col - player_pos[0]) + abs(d.row - player_pos[1])
                if dist <= 1:
                    if carried_track:
                        carried_track.is_carried = False
                        carried_track.current_pos = (d.col, d.row)
                        carried_track.last_seen_t = self.time_counter
                    if (d.col, d.row) in receptacle_cells:
                        self.delivered_items.add((d.col, d.row))
                    self.carried_object_id = None
                    self.carried_offset = None
                    self.drops_count += 1
                    logger.info(
                        f"[TRACKER] Confirmed DROP of item at {(d.col, d.row)} (total drops: {self.drops_count})"
                    )
                    return

            # Generic drop fallback: in ARC-AGI-3, executing ACTION5 while carrying unconditionally drops the item
            if carried_track and drop_pos:
                carried_track.is_carried = False
                carried_track.current_pos = drop_pos
                self.carried_object_id = None
                self.carried_offset = None
                self.drops_count += 1
                logger.info(
                    f"[TRACKER] Confirmed generic DROP of item at {drop_pos} (total drops: {self.drops_count})"
                )
                return

    def get_items(self) -> List[TrackedObject]:
        """Return all active, uncarried candidate items that have not yet been delivered."""
        receptacle_cells = {t.current_pos for t in self.get_receptacles()}
        return [
            t for t in self.tracked_objects.values()
            if not t.is_carried
            and t.object_id != self.carried_object_id
            and t.category == ObjectType.ITEM
            and t.item_confidence >= 0.4
            and t.motion_state != "carried"
            and t.current_pos not in receptacle_cells
            and t.current_pos not in self.delivered_items
        ]

    def get_receptacles(self) -> List[TrackedObject]:
        """Return candidate receptacle / drop zone entities."""
        return [
            t for t in self.tracked_objects.values()
            if not t.is_carried
            and t.object_id != self.carried_object_id
            and t.category == ObjectType.RECEPTACLE
            and t.receptacle_confidence >= 0.4
        ]

    def get_moving_entities(self) -> List[TrackedObject]:
        """Return moving NPCs or platforms for collision avoidance, strictly excluding carried items and static walls."""
        return [
            t for t in self.tracked_objects.values()
            if not t.is_carried
            and t.object_id != self.carried_object_id
            and t.category != ObjectType.WALL
            and t.wall_confidence < 0.5
            and (t.motion_state == "moving" or t.npc_confidence >= 0.5)
        ]
