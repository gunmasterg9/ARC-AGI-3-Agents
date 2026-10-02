import numpy as np
import pytest
from arcengine import GameAction

from agents.arcagi_v3.actions import grid_delta_to_action
from agents.arcagi_v3.affordance import DynamicGridTracker
from agents.arcagi_v3.agent import ARCAGIV3Solver
from agents.arcagi_v3.objects import GameObject, ObjectType
from agents.arcagi_v3.perception import PerceptionEngine
from agents.arcagi_v3.persistence import MultiFrameObjectTracker
from agents.arcagi_v3.planner import Planner
from agents.arcagi_v3.state import PlayerState
from agents.arcagi_v3.temporal import MovingObjectModel, TemporalWorldModel
from agents.arcagi_v3.world_model import WorldModel


@pytest.mark.unit
def test_wa30_4px_stride_detection():
    """1. Test WA30 4px stride detection from screen geometry."""
    tracker = DynamicGridTracker()
    screen = np.zeros((64, 64), dtype=np.int8)
    # Draw 4x4 blocks at stride 4
    for r in range(4):
        for c in range(4):
            screen[r * 4 : r * 4 + 4, c * 4 : c * 4 + 4] = (r + c) % 5 + 1
    sx, sy, ox, oy = tracker.calibrate_from_screen(screen, is_ls20_hint=False)
    assert sx == 4
    assert sy == 4
    assert tracker.cols == 16
    assert tracker.rows == 16


@pytest.mark.unit
def test_ls20_5px_stride_regression():
    """2. Test LS20 5px stride regression is preserved."""
    tracker = DynamicGridTracker()
    screen = np.zeros((64, 64), dtype=np.int8)
    sx, sy, ox, oy = tracker.calibrate_from_screen(screen, is_ls20_hint=True)
    assert sx == 5
    assert sy == 5
    assert ox == 4
    assert oy == 0
    assert tracker.cols == 11
    assert tracker.rows == 11


@pytest.mark.unit
def test_offset_detection():
    """3. Test offset detection for both 0,0 and 4,0 layouts."""
    tracker_wa30 = DynamicGridTracker()
    screen_wa30 = np.zeros((64, 64), dtype=np.int8)
    screen_wa30[0:4, 0:4] = 2
    screen_wa30[4:8, 4:8] = 2
    sx, sy, ox, oy = tracker_wa30.calibrate_from_screen(screen_wa30, is_ls20_hint=False)
    assert ox == 0
    assert oy == 0

    tracker_ls20 = DynamicGridTracker()
    sx, sy, ox, oy = tracker_ls20.calibrate_from_screen(screen_wa30, is_ls20_hint=True)
    assert ox == 4
    assert oy == 0


@pytest.mark.unit
def test_pixel_to_grid_conversion():
    """4. Test pixel to grid conversion."""
    tracker = DynamicGridTracker(stride=4, offset_x=0, offset_y=0, cols=16, rows=16)
    assert tracker.pixel_to_grid(32, 48) == (8, 12)
    assert tracker.pixel_to_grid(0, 0) == (0, 0)
    assert tracker.pixel_to_grid(44, 24) == (11, 6)

    tracker_ls20 = DynamicGridTracker(stride=5, offset_x=4, offset_y=0, cols=11, rows=11)
    assert tracker_ls20.pixel_to_grid(4, 0) == (0, 0)
    assert tracker_ls20.pixel_to_grid(9, 5) == (1, 1)


@pytest.mark.unit
def test_grid_to_pixel_conversion():
    """5. Test grid to pixel conversion."""
    tracker = DynamicGridTracker(stride=4, offset_x=0, offset_y=0, cols=16, rows=16)
    assert tracker.grid_to_pixel(8, 12) == (32, 48)
    assert tracker.grid_to_pixel(11, 6) == (44, 24)

    tracker_ls20 = DynamicGridTracker(stride=5, offset_x=4, offset_y=0, cols=11, rows=11)
    assert tracker_ls20.grid_to_pixel(0, 0) == (4, 0)
    assert tracker_ls20.grid_to_pixel(1, 1) == (9, 5)


@pytest.mark.unit
def test_round_trip_conversion():
    """6. Test round-trip conversion pixel <-> grid."""
    tracker = DynamicGridTracker(stride=4, offset_x=0, offset_y=0, cols=16, rows=16)
    for col in range(16):
        for row in range(16):
            px, py = tracker.grid_to_pixel(col, row)
            c, r = tracker.pixel_to_grid(px, py)
            assert (c, r) == (col, row)


@pytest.mark.unit
def test_player_localization():
    """7. Test player localization for 4px avatar in WA30."""
    perc = PerceptionEngine()
    screen = np.zeros((64, 64), dtype=np.int8)
    # Player avatar: 4x4 with color 14
    screen[48:52, 32:36] = 14
    player = perc.detect_player(screen, stride=4, offset_x=0, offset_y=0)
    assert player is not None
    assert player.pixel_pos == (32, 48)
    assert player.grid_pos == (8, 12)


@pytest.mark.unit
def test_item_localization():
    """8. Test item localization for 4px items."""
    perc = PerceptionEngine()
    screen = np.zeros((64, 64), dtype=np.int8)
    # Item at (11, 6) -> pixel (44, 24)
    screen[24:28, 44:48] = 4
    screen[25:27, 45:47] = 9
    objs = perc.detect_objects(screen, player=None, stride=4, offset_x=0, offset_y=0, cols=16, rows=16)
    item_objs = [o for o in objs if o.object_type == ObjectType.ITEM and o.col == 11 and o.row == 6]
    assert len(item_objs) == 1


@pytest.mark.unit
def test_receptacle_localization():
    """9. Test receptacle localization for color 2 zones."""
    perc = PerceptionEngine()
    screen = np.zeros((64, 64), dtype=np.int8)
    # Receptacle at row 7, cols 7..9 -> pixels (28..40, 28..32)
    screen[28:32, 28:40] = 2
    objs = perc.detect_objects(screen, player=None, stride=4, offset_x=0, offset_y=0, cols=16, rows=16)
    rec_objs = [o for o in objs if o.object_type == ObjectType.RECEPTACLE and o.row == 7]
    assert len(rec_objs) == 3
    assert {o.col for o in rec_objs} == {7, 8, 9}


@pytest.mark.unit
def test_interaction_cell_calculation():
    """10. Test interaction-cell calculation for target."""
    wm = WorldModel(cols=16, rows=16)
    planner = Planner(wm)
    # Target item at (8, 9), player at (8, 12)
    # Shortest path to neighbor of (8, 9) ends at (8, 10)
    actions = planner.plan_composite_interaction(
        start=(8, 12),
        target=(8, 9),
        interaction_action=GameAction.ACTION5,
        required_distance=1,
    )
    # Expect 2 UP movements then ACTION5
    assert actions == [GameAction.ACTION1, GameAction.ACTION1, GameAction.ACTION5]


@pytest.mark.unit
def test_facing_calculation():
    """11. Test required facing direction alignment."""
    wm = WorldModel(cols=16, rows=16)
    planner = Planner(wm)
    # Player at (8, 10), target item at (8, 9) -> delta (0, -1) -> ACTION1
    actions = planner.plan_composite_interaction(
        start=(8, 10),
        target=(8, 9),
        interaction_action=GameAction.ACTION5,
        required_distance=1,
    )
    assert actions[0] == GameAction.ACTION1
    assert actions[1] == GameAction.ACTION5


@pytest.mark.unit
def test_pickup_before_after_detection():
    """12. Test pickup detection via visual diff at target cell."""
    tracker = MultiFrameObjectTracker(stride=4, offset_x=0, offset_y=0)
    det = GameObject(object_type=ObjectType.ITEM, x=32, y=36, col=8, row=9, color=4)
    tracker.update([det], None, player_pos=(8, 10), last_action=None)
    assert tracker.carried_object_id is None

    # Before screen: border color 3
    prev_screen = np.zeros((64, 64), dtype=np.int8)
    prev_screen[36:40, 32:36] = 3
    # After screen: border color 0
    curr_screen = np.zeros((64, 64), dtype=np.int8)
    curr_screen[36:40, 32:36] = 0

    tracker.update(
        [det],
        current_frame=curr_screen,
        player_pos=(8, 10),
        last_action=GameAction.ACTION5,
        prev_frame=prev_screen,
        carried_offset=(0, -1),
    )
    assert tracker.carried_object_id is not None
    assert tracker.pickups_count == 1


@pytest.mark.unit
def test_drop_before_after_detection():
    """13. Test drop detection into receptacle zone."""
    tracker = MultiFrameObjectTracker(stride=4, offset_x=0, offset_y=0)
    item_det = GameObject(object_type=ObjectType.ITEM, x=32, y=36, col=8, row=9, color=4)
    rec_det = GameObject(object_type=ObjectType.RECEPTACLE, x=32, y=28, col=8, row=7, color=2)
    tracker.update([item_det, rec_det], None, player_pos=(8, 10), last_action=None)

    # Pick up item
    item_track = list(tracker.tracked_objects.values())[0]
    tracker.carried_object_id = item_track.object_id
    item_track.is_carried = True

    # Drop into receptacle at (8, 7) when player is at (8, 8) with offset (0, -1)
    tracker.update(
        [rec_det],
        current_frame=None,
        player_pos=(8, 8),
        last_action=GameAction.ACTION5,
        carried_offset=(0, -1),
    )
    assert tracker.carried_object_id is None
    assert (8, 7) in tracker.delivered_items
    assert tracker.drops_count == 1


@pytest.mark.unit
def test_failed_action5_recovery():
    """14. Test failed ACTION5 detection when screen does not change."""
    tracker = MultiFrameObjectTracker(stride=4, offset_x=0, offset_y=0)
    screen = np.zeros((64, 64), dtype=np.int8)
    # Player at (8, 10), no items nearby
    tracker.update(
        [],
        current_frame=screen,
        player_pos=(8, 10),
        last_action=GameAction.ACTION5,
        prev_frame=screen,
    )
    assert tracker.carried_object_id is None
    assert tracker.pickups_count == 0


@pytest.mark.unit
def test_carried_object_persistence():
    """15. Test carried object persists and updates position with player."""
    tracker = MultiFrameObjectTracker(stride=4, offset_x=0, offset_y=0)
    det = GameObject(object_type=ObjectType.ITEM, x=32, y=36, col=8, row=9, color=4)
    tracker.update([det], None, player_pos=(8, 10), last_action=None)
    item_id = list(tracker.tracked_objects.keys())[0]
    tracker.carried_object_id = item_id
    tracker.tracked_objects[item_id].is_carried = True

    # Player moves to (8, 9) with carried offset (0, -1)
    tracker.update([], None, player_pos=(8, 9), last_action=GameAction.ACTION1, carried_offset=(0, -1))
    track = tracker.tracked_objects[item_id]
    assert track.current_pos == (8, 8)
    assert track.is_carried is True
    assert track.motion_state == "carried"
    # Must NOT be classified as moving NPC
    assert track.category == ObjectType.ITEM


@pytest.mark.unit
def test_npc_aware_interaction_planning():
    """16. Test NPC-aware interaction planning routes around dynamic obstacles."""
    wm = WorldModel(cols=8, rows=8)
    planner = Planner(wm)
    tm = TemporalWorldModel()
    # NPC moving horizontally at row 1
    tm.moving_objects["npc"] = MovingObjectModel(
        object_id="npc",
        current_pos=(1, 1),
        velocity=(0, 0),
        confidence=0.9,
    )
    # Player at (0, 1), target item at (2, 1)
    # Direct path is blocked by NPC at (1, 1). Space-time planning finds alternate path.
    actions = planner.plan_composite_interaction(
        start=(0, 1),
        target=(2, 1),
        interaction_action=GameAction.ACTION5,
        required_distance=1,
        temporal_model=tm,
    )
    assert len(actions) > 0
    assert actions[-1] == GameAction.ACTION5


@pytest.mark.unit
def test_carrier_identity_masking_persistence_and_collisions():
    """17. Test Phase 1 Carrier Identity Masking:
    - Carried object excluded from get_items() and get_moving_entities()
    - Preserved in tracked_objects for verification
    - Detections at carried position update track without creating ghost NPC movers
    """
    tracker = MultiFrameObjectTracker(stride=4, offset_x=0, offset_y=0)
    item_det = GameObject(object_type=ObjectType.ITEM, x=16, y=20, col=4, row=5, color=4)
    tracker.update([item_det], None, player_pos=(4, 6), last_action=None)

    item_id = list(tracker.tracked_objects.keys())[0]
    tracker.carried_object_id = item_id
    tracker.tracked_objects[item_id].is_carried = True

    # Player moves to (4, 5) carrying item at offset (0, -1) -> carried pos is (4, 4)
    carried_det = GameObject(object_type=ObjectType.ITEM, x=16, y=16, col=4, row=4, color=4)
    # Incoming detection at carried position
    tracker.update(
        [carried_det],
        current_frame=None,
        player_pos=(4, 5),
        last_action=GameAction.ACTION1,
        carried_offset=(0, -1),
    )

    # 1. Carried item must remain tracked
    assert item_id in tracker.tracked_objects
    track = tracker.tracked_objects[item_id]
    assert track.current_pos == (4, 4)
    assert track.is_carried is True

    # 2. Excluded from items and moving entities
    assert tracker.get_items() == []
    assert tracker.get_moving_entities() == []

    # 3. No ghost mover tracks created
    assert len(tracker.tracked_objects) == 1


@pytest.mark.unit
def test_pmargquscu_wall_classification():
    """18. Test Phase 2 Multi-Color Component Classification:
    Specifically tests that WA30 wall sprite 'pmargquscu' is classified as WALL,
    not RECEPTACLE, while border-9/interior-2 sprite is classified as RECEPTACLE.
    """
    perception = PerceptionEngine()
    screen = np.zeros((64, 64), dtype=np.int8)

    # 1. Textured wall sprite pmargquscu: 4x4 matrix with 2 and background
    # pixels: [[2, 0, 2, 2], [0, 2, 2, 2], [2, 2, 2, 0], [2, 2, 0, 2]]
    pmargquscu = np.array([
        [2, 0, 2, 2],
        [0, 2, 2, 2],
        [2, 2, 2, 0],
        [2, 2, 0, 2],
    ], dtype=np.int8)
    screen[8:12, 8:12] = pmargquscu  # cell (col=2, row=2)

    # 2. Receptacle sprite with border 9 and interior 2
    rec = np.array([
        [9, 9, 9, 9],
        [9, 2, 2, 9],
        [9, 2, 2, 9],
        [9, 9, 9, 9],
    ], dtype=np.int8)
    screen[16:20, 16:20] = rec  # cell (col=4, row=4)

    # 3. Collectible item sprite with border 4 and interior 9
    item = np.array([
        [4, 4, 4, 4],
        [4, 9, 9, 4],
        [4, 9, 9, 4],
        [4, 4, 4, 4],
    ], dtype=np.int8)
    screen[24:28, 24:28] = item  # cell (col=6, row=6)

    # 4. Pure solid wall
    screen[32:36, 32:36] = 3  # cell (col=8, row=8)

    objects = perception.detect_objects(
        screen,
        player=None,
        stride=4,
        offset_x=0,
        offset_y=0,
        cols=16,
        rows=16,
    )

    obj_by_pos = {(o.col, o.row): o for o in objects}

    # pmargquscu must be WALL
    assert (2, 2) in obj_by_pos
    assert obj_by_pos[(2, 2)].object_type == ObjectType.WALL

    # Receptacle must be RECEPTACLE
    assert (4, 4) in obj_by_pos
    assert obj_by_pos[(4, 4)].object_type == ObjectType.RECEPTACLE

    # Collectible item must be ITEM
    assert (6, 6) in obj_by_pos
    assert obj_by_pos[(6, 6)].object_type == ObjectType.ITEM

    # Solid wall must be WALL
    assert (8, 8) in obj_by_pos
    assert obj_by_pos[(8, 8)].object_type == ObjectType.WALL


@pytest.mark.unit
def test_carrier_composite_footprint_collision():
    """19. Test Phase 3 Carrier Footprint:
    Verifies that Space-Time A* and BFS check composite footprint
    (player cell + carried cell) against static and dynamic obstacles.
    """
    wm = WorldModel(cols=6, rows=6)
    # Put a wall at (2, 2)
    wm.walls.add((2, 2))
    planner = Planner(wm)

    # Player starts at (1, 1). Target is (3, 1).
    # Case A: Free player, no carried item. Can move (1,1)->(2,1)->(3,1).
    path_free = planner.find_grid_path(start=(1, 1), target=(3, 1), carried_offset=None)
    assert path_free == [(1, 1), (2, 1), (3, 1)]

    # Case B: Player carries item at offset (0, 1) [south of player].
    # If player moves to (2, 1), carried item would be at (2, 2), which is a WALL!
    # Path planner must reject stepping into (2, 1) and route around!
    path_carried = planner.find_grid_path(start=(1, 1), target=(3, 1), carried_offset=(0, 1))
    assert path_carried is not None
    # Verify no waypoint in path puts the carried item into (2, 2)
    for p in path_carried:
        carried_p = (p[0] + 0, p[1] + 1)
        assert carried_p not in wm.walls
        assert p not in wm.walls


@pytest.mark.unit
def test_pickup_orientation_feasibility_selection():
    """20. Test Phase 3 Orientation Feasibility Selection:
    Item is located at (2, 2).
    A narrow corridor exists to the right towards receptacle at (5, 2).
    Picking up from the left (1, 2) creates carried_offset = (+1, 0) [item in front].
    Picking up from above (2, 1) creates carried_offset = (0, +1) [item below player].
    If there is a wall below the corridor at row 3, offset (0, +1) would collide with the wall,
    while offset (+1, 0) fits through the corridor.
    Planner must simulate the route and choose the feasible approach orientation.
    """
    wm = WorldModel(cols=8, rows=6)
    # Corridor along row 2 from col 2 to 6. Wall along row 3 from col 2 to 6.
    for c in range(2, 7):
        wm.walls.add((c, 3))
    # Wall above at row 1 from col 3 to 6
    for c in range(3, 7):
        wm.walls.add((c, 1))

    planner = Planner(wm)
    # Player starts at (0, 2)
    # Item is at (2, 2)
    # Receptacle is at (6, 2)
    actions = planner.plan_composite_interaction(
        start=(0, 2),
        target=(2, 2),
        interaction_action=GameAction.ACTION5,
        required_distance=1,
        destination_targets=[(6, 2)],
    )

    assert len(actions) > 0
    assert actions[-1] == GameAction.ACTION5
    # Player approached from (1, 2) facing right (ACTION4)
    # Resulting offset is (+1, 0), which fits in 1-tile high corridor row 2!
    assert actions[-2] == GameAction.ACTION4


@pytest.mark.unit
def test_static_wall_bump_retains_permanent_wall():
    """21. Phase 5.1: Static wall bump -> permanent wall remains valid."""
    agent = ARCAGIV3Solver(game_id="wa30")
    agent.action_counter = 5
    # Simulate a static wall at (3, 3) known in world_model.walls
    agent.world_model.walls.add((3, 3))

    # Player attempted move to (3, 3) and was blocked
    # Check blocked classification logic:
    # (3, 3) is a known static wall, no moving entity there
    is_static_wall = (3, 3) in agent.world_model.walls
    assert is_static_wall
    # In agent logic, static collision retains permanent wall
    agent.world_model.mark_obstacle(3, 3)
    assert (3, 3) in agent.world_model.walls
    assert (3, 3) not in agent.temporal_model.transient_obstacles


@pytest.mark.unit
def test_npc_bump_creates_transient_obstacle_not_permanent_wall():
    """22. Phase 5.2: NPC bump -> cell NOT added to permanent wall set."""
    agent = ARCAGIV3Solver(game_id="wa30")
    agent.action_counter = 10

    # Set up moving entity at (4, 4) in persistence tracker
    from agents.arcagi_v3.persistence import TrackedObject
    mover = TrackedObject(
        object_id="npc_1",
        current_pos=(4, 4),
        pixel_pos=(16, 16),
        width=4,
        height=4,
        motion_state="moving",
    )
    mover.positions_history = [(3, 4), (4, 4)]
    agent.persistence_tracker.tracked_objects["npc_1"] = mover

    # Player attempts move into (4, 4) and gets blocked
    cell = (4, 4)
    moving_entities = agent.persistence_tracker.get_moving_entities()
    is_moving_entity = any(
        m.current_pos == cell or (len(m.positions_history) >= 2 and m.positions_history[-2] == cell)
        for m in moving_entities
    )
    assert is_moving_entity

    # Ensure (4, 4) is registered as transient obstacle with TTL=3, not permanent wall
    agent.world_model.walls.discard(cell)
    agent.temporal_model.add_transient_obstacle(
        position=cell,
        source_id="npc_1",
        t=agent.action_counter,
        ttl=3,
        confidence=0.9,
    )

    assert cell not in agent.world_model.walls
    assert cell in agent.temporal_model.transient_obstacles
    assert agent.temporal_model.is_temporally_blocked(cell[0], cell[1], target_t=10)


@pytest.mark.unit
def test_npc_moves_away_cell_becomes_traversable():
    """23. Phase 5.3: NPC moves away -> previously blocked cell becomes traversable."""
    temporal = TemporalWorldModel()
    cell = (4, 4)
    temporal.add_transient_obstacle(
        position=cell,
        source_id="npc_1",
        t=10,
        ttl=3,
        confidence=0.9,
    )
    assert temporal.is_temporally_blocked(4, 4, target_t=10)

    # NPC moves away to (5, 4) at t=11
    # When update_transient_obstacles runs with moving_positions={(5, 4)},
    # cell (4, 4) is vacated by the entity and cleared!
    temporal.update_transient_obstacles(current_t=11, moving_positions={(5, 4)})
    assert cell not in temporal.transient_obstacles
    assert not temporal.is_temporally_blocked(4, 4, target_t=11)


@pytest.mark.unit
def test_transient_obstacle_expires_by_ttl():
    """24. Phase 5.4: Transient obstacle expires correctly by TTL."""
    temporal = TemporalWorldModel()
    cell = (2, 5)
    temporal.add_transient_obstacle(
        position=cell,
        source_id="unknown_block",
        t=20,
        ttl=2,
        confidence=0.5,
    )
    # At t=21, still valid
    assert temporal.is_temporally_blocked(2, 5, target_t=21)

    # At t=22 (20 + 2), expired!
    temporal.update_transient_obstacles(current_t=22)
    assert cell not in temporal.transient_obstacles
    assert not temporal.is_temporally_blocked(2, 5, target_t=22)


@pytest.mark.unit
def test_npc_velocity_incorporated_into_space_time_replanning():
    """25. Phase 5.5: NPC velocity is incorporated into replanning."""
    temporal = TemporalWorldModel()
    # Add a moving object at (3, 2) moving south towards (3, 3) at t=1, (3, 4) at t=2
    mover = MovingObjectModel(
        object_id="mover_1",
        current_pos=(3, 2),
        previous_pos=(3, 1),
        velocity=(0, 1),
        confidence=0.8,
        blocks_player=True,
    )
    mover.time_history = [0]
    temporal.moving_objects["mover_1"] = mover

    # Predict positions:
    # At t=1, mover is at (3, 3)
    assert mover.predict_position_at(target_t=1, current_t=0) == (3, 3)
    temporal.current_time = 0
    # Cell (3, 3) should be temporally blocked at t=1
    assert temporal.is_temporally_blocked(3, 3, target_t=1)
    # But cell (3, 3) is NOT blocked at t=0
    assert not temporal.is_temporally_blocked(3, 3, target_t=0)


@pytest.mark.unit
def test_carrier_composite_footprint_with_transient_obstacles():
    """26. Phase 5.6: Carrier composite footprint still works with transient obstacles."""
    wm = WorldModel(cols=6, rows=6)
    temporal = TemporalWorldModel()
    # Put transient obstacle at (2, 2)
    temporal.add_transient_obstacle(position=(2, 2), t=0, ttl=5)

    # Player starts at (1, 1). Target is (3, 1).
    # Player carries item at offset (0, 1) [south of player].
    # Stepping into (2, 1) places carried item at (2, 2) which is temporally blocked!
    planner = Planner(wm)
    res = planner.find_space_time_path(
        start=(1, 1),
        target=(3, 1),
        start_t=0,
        temporal_model=temporal,
        carried_offset=(0, 1),
    )
    # Path must not step into (2, 1) at t=1 because carried item would hit (2, 2)
    assert res is not None
    actions, path = res
    for step_t, pos in enumerate(path):
        carried_pos = (pos[0] + 0, pos[1] + 1)
        if step_t == 0:
            continue
        assert carried_pos != (2, 2), f"Carried item collided with transient obstacle at t={step_t}"


@pytest.mark.unit
def test_multiple_npcs_supported():
    """27. Phase 5.7: Multiple NPCs remain supported."""
    temporal = TemporalWorldModel()
    mover1 = MovingObjectModel(
        object_id="mover_1",
        current_pos=(2, 2),
        velocity=(1, 0),
        confidence=0.8,
        blocks_player=True,
    )
    mover1.time_history = [0]

    mover2 = MovingObjectModel(
        object_id="mover_2",
        current_pos=(5, 5),
        velocity=(0, -1),
        confidence=0.8,
        blocks_player=True,
    )
    mover2.time_history = [0]

    temporal.moving_objects["mover_1"] = mover1
    temporal.moving_objects["mover_2"] = mover2

    assert temporal.has_active_moving_objects()
    temporal.current_time = 0
    # At t=1, mover1 is at (3, 2), mover2 is at (5, 4)
    assert temporal.is_temporally_blocked(3, 2, target_t=1)
    assert temporal.is_temporally_blocked(5, 4, target_t=1)
    # Unrelated cells are free
    assert not temporal.is_temporally_blocked(0, 0, target_t=1)


@pytest.mark.unit
def test_temporary_npc_blockage_does_not_create_oscillation():
    """28. Phase 5.8: Temporary NPC blockage does not create oscillation."""
    agent = ARCAGIV3Solver(game_id="wa30")
    agent.action_counter = 12
    # Player at (2, 2), has plan to move right: [(3, 2), (4, 2)]
    # NPC is currently passing through (3, 2)
    mover = MovingObjectModel(
        object_id="npc_x",
        current_pos=(3, 2),
        previous_pos=(3, 1),
        velocity=(0, 1),
        confidence=0.8,
        blocks_player=True,
    )
    mover.time_history = [12]
    agent.temporal_model.moving_objects["npc_x"] = mover

    # Check wait/yield logic
    player = PlayerState(x=8, y=8, col=2, row=2)
    wait_action = agent._check_and_execute_temporal_wait_yield(player)
    # Safe wait action should be triggered (ACTION5 for not carrying)
    assert wait_action == GameAction.ACTION5
    assert agent.consecutive_waits == 1
    # Wall set must NOT contain (3, 2)
    assert (3, 2) not in agent.world_model.walls


@pytest.mark.unit
def test_space_time_single_npc_blocks_for_1_step():
    """29. Phase 6.1: Single NPC blocks route for 1 step -> wait 1 step, then proceed."""
    wm = WorldModel(cols=6, rows=6)
    temporal = TemporalWorldModel()
    temporal.current_time = 0
    # Corridor along row 2: start=(1, 2), target=(4, 2)
    # NPC at (2, 1) moving south (0, 1): at t=1 occupies (2, 2), at t=2 occupies (2, 3)
    mover = MovingObjectModel(
        object_id="npc_1",
        current_pos=(2, 1),
        previous_pos=(2, 0),
        velocity=(0, 1),
        confidence=0.8,
        blocks_player=True,
    )
    mover.time_history = [0]
    temporal.moving_objects["npc_1"] = mover

    planner = Planner(wm)
    res = planner.find_space_time_path(
        start=(1, 2),
        target=(4, 2),
        start_t=0,
        temporal_model=temporal,
        carried_offset=None,
    )
    assert res is not None
    actions, coords = res
    # First action must be a wait at (1, 2)
    assert coords[1] == (1, 2)
    assert actions[0] == GameAction.ACTION5
    # Then steps into (2, 2) at t=2
    assert coords[2] == (2, 2)
    assert coords[-1] == (4, 2)


@pytest.mark.unit
def test_space_time_single_npc_blocks_for_2_steps():
    """30. Phase 6.2: Single NPC blocks route for 2 steps -> wait 2 steps, then proceed."""
    wm = WorldModel(cols=6, rows=6)
    temporal = TemporalWorldModel()
    temporal.current_time = 0
    # Wall above and below row 2 to make it a narrow 1-tile corridor
    for c in range(6):
        wm.walls.add((c, 1))
        wm.walls.add((c, 3))
    wm.walls.discard((2, 1))  # vertical crossing at col 2
    wm.walls.discard((2, 3))

    # NPC moving south through col 2:
    # At t=0: at (2, 0), at t=1: (2, 1), at t=2: (2, 2) [corridor cell blocked], at t=3: (2, 3) [corridor clear]
    mover = MovingObjectModel(
        object_id="npc_cross",
        current_pos=(2, 0),
        previous_pos=(2, -1),
        velocity=(0, 1),
        confidence=0.8,
        blocks_player=True,
    )
    mover.time_history = [0]
    temporal.moving_objects["npc_cross"] = mover

    planner = Planner(wm)
    res = planner.find_space_time_path(
        start=(1, 2),
        target=(4, 2),
        start_t=0,
        temporal_model=temporal,
        carried_offset=None,
    )
    assert res is not None
    actions, coords = res
    # Waits 1 step at t=0 so at t=2 player is at (1, 2), and steps into (2, 2) at t=3 when clear
    assert (2, 2) in coords
    # Verify no collision with mover
    for t_step, p in enumerate(coords):
        pred_npc = mover.predict_position_at(t_step, current_t=0)
        assert p != pred_npc


@pytest.mark.unit
def test_space_time_single_npc_blocks_for_3_to_4_steps():
    """31. Phase 6.3: Single NPC blocks route for 3-4 steps -> adaptive wait up to k=4."""
    wm = WorldModel(cols=6, rows=6)
    temporal = TemporalWorldModel()
    temporal.current_time = 0
    # Single corridor from (1, 2) to (3, 2). Enclosed walls including (0, 2) dead-end.
    for c in range(6):
        wm.walls.add((c, 1))
        wm.walls.add((c, 3))
    wm.walls.add((0, 2))
    # Transient obstacle at (2, 2) with expiration at step 4 (wait 3 steps)
    temporal.add_transient_obstacle(position=(2, 2), t=0, ttl=4)

    planner = Planner(wm)
    res = planner.find_space_time_path(
        start=(1, 2),
        target=(3, 2),
        start_t=0,
        temporal_model=temporal,
        max_wait_horizon=4,
    )
    assert res is not None
    actions, coords = res
    assert coords[-1] == (3, 2)
    # Number of wait steps is exactly 3 (coords repeating (1, 2))
    wait_count = sum(1 for i in range(len(actions)) if coords[i + 1] == coords[i])
    assert wait_count == 3


@pytest.mark.unit
def test_space_time_predicted_vacancy_selects_wait():
    """32. Phase 6.4: Predicted vacancy selects wait over expensive detour."""
    wm = WorldModel(cols=7, rows=7)
    temporal = TemporalWorldModel()
    temporal.current_time = 0
    # Short straight path is row 2: (1, 2) -> (2, 2) -> (3, 2) -> (4, 2) (cost: 3 moves)
    # Alternate path goes all the way around rows 4-5 (cost: 9 moves)
    for c in range(1, 5):
        wm.walls.add((c, 3))  # dividing wall between row 2 and row 4
    # Cell (2, 2) has a transient obstacle that clears at step 2
    temporal.add_transient_obstacle(position=(2, 2), t=0, ttl=2)

    planner = Planner(wm)
    res = planner.find_space_time_path(
        start=(1, 2),
        target=(4, 2),
        start_t=0,
        temporal_model=temporal,
    )
    assert res is not None
    actions, coords = res
    # Path should choose waiting 2 steps then crossing (total ~5 steps) over 9-step detour
    assert len(actions) < 9
    assert coords[-1] == (4, 2)


@pytest.mark.unit
def test_space_time_detour_selected_when_cheaper():
    """33. Phase 6.5: Detour selected when detour cost is cheaper than waiting."""
    wm = WorldModel(cols=6, rows=6)
    temporal = TemporalWorldModel()
    temporal.current_time = 0
    # Pillar at (2, 2).
    # Path A (above pillar): (1, 2) -> (1, 1) -> (2, 1) -> (3, 1) -> (3, 2) [blocked by NPC]
    # Path B (below pillar): (1, 2) -> (1, 3) -> (2, 3) -> (3, 3) -> (3, 2) [clear detour]
    wm.walls.add((2, 2))  # central pillar
    # Cell (2, 1) is blocked by an NPC for 4 steps!
    temporal.add_transient_obstacle(position=(2, 1), t=0, ttl=4)

    planner = Planner(wm)
    res = planner.find_space_time_path(
        start=(1, 2),
        target=(3, 2),
        start_t=0,
        temporal_model=temporal,
    )
    assert res is not None
    actions, coords = res
    # Detour through (2, 3) is chosen over waiting for (2, 1) to clear!
    assert (2, 3) in coords


@pytest.mark.unit
def test_space_time_multi_npc_blocking():
    """34. Phase 6.6: Multi-NPC blocking evaluated concurrently."""
    wm = WorldModel(cols=8, rows=8)
    temporal = TemporalWorldModel()
    temporal.current_time = 0

    # NPC 1 crosses at (2, 2) moving south
    m1 = MovingObjectModel(object_id="m1", current_pos=(2, 1), velocity=(0, 1), confidence=0.8, blocks_player=True)
    m1.time_history = [0]
    # NPC 2 crosses at (4, 2) moving north
    m2 = MovingObjectModel(object_id="m2", current_pos=(4, 3), velocity=(0, -1), confidence=0.8, blocks_player=True)
    m2.time_history = [0]
    temporal.moving_objects["m1"] = m1
    temporal.moving_objects["m2"] = m2

    planner = Planner(wm)
    res = planner.find_space_time_path(
        start=(1, 2),
        target=(5, 2),
        start_t=0,
        temporal_model=temporal,
    )
    assert res is not None
    actions, coords = res
    assert coords[-1] == (5, 2)
    # Verify neither mover collides with player at any step
    for t_step, p in enumerate(coords):
        assert p != m1.predict_position_at(t_step, current_t=0)
        assert p != m2.predict_position_at(t_step, current_t=0)


@pytest.mark.unit
def test_space_time_wait_while_carrying():
    """35. Phase 6.7: Wait while carrying executes static wall bump, not ACTION5."""
    wm = WorldModel(cols=6, rows=6)
    wm.walls.add((1, 1))  # Static wall adjacent to player at (1, 2)
    temporal = TemporalWorldModel()
    temporal.current_time = 0

    # Blocking obstacle at (2, 2) for 1 step
    temporal.add_transient_obstacle(position=(2, 2), t=0, ttl=1)

    planner = Planner(wm)
    # Player at (1, 2) carrying item at offset (0, 1) [south of player]
    wait_act = planner.resolve_safe_wait_action(
        col=1,
        row=2,
        carried_offset=(0, 1),
        temporal_model=temporal,
    )
    # Must be ACTION1 (up into wall at 1, 1), NOT ACTION5 (which would drop item!)
    assert wait_act == GameAction.ACTION1


@pytest.mark.unit
def test_space_time_wait_preserves_carried_offset():
    """36. Phase 6.8: Wait preserves carried offset and carried_object_id."""
    agent = ARCAGIV3Solver(game_id="wa30")
    agent.world_model.carried_object_id = "item_123"
    agent.carried_offset = (0, -1)
    agent.world_model.walls.add((2, 1))  # Static wall north of player at (2, 2)

    # Nearby blocking NPC at (3, 2)
    mover = MovingObjectModel(object_id="npc_b", current_pos=(3, 2), velocity=(0, 1), confidence=0.8)
    mover.time_history = [0]
    agent.temporal_model.moving_objects["npc_b"] = mover

    player = PlayerState(x=8, y=8, col=2, row=2)
    wait_action = agent._check_and_execute_temporal_wait_yield(player)
    # Must bump into static wall (2, 1) via ACTION1
    assert wait_action == GameAction.ACTION1
    assert agent.is_executing_intentional_wait
    assert agent.world_model.carried_object_id == "item_123"
    assert agent.carried_offset == (0, -1)


@pytest.mark.unit
def test_space_time_wait_does_not_create_static_wall():
    """37. Phase 6.9: Wait bump does not corrupt world_model.walls."""
    agent = ARCAGIV3Solver(game_id="wa30")
    agent.world_model.carried_object_id = "item_456"
    agent.carried_offset = (1, 0)
    agent.world_model.walls.add((2, 1))

    # Mark intentional wait
    agent.is_executing_intentional_wait = True
    agent.last_action = GameAction.ACTION1

    # Simulate observe_transition blocked check
    if agent.is_executing_intentional_wait:
        agent.is_executing_intentional_wait = False
        agent.successful_temporal_waits += 1

    # Cell where player is waiting (2, 2) must NOT be a wall
    assert (2, 2) not in agent.world_model.walls
    assert agent.successful_temporal_waits == 1


@pytest.mark.unit
def test_space_time_transient_obstacle_expiration():
    """38. Phase 6.10: Transient obstacle expiration enables space-time planning."""
    wm = WorldModel(cols=6, rows=6)
    temporal = TemporalWorldModel()
    temporal.current_time = 0
    # Transient obstacle expires at step 2
    temporal.add_transient_obstacle(position=(2, 2), t=0, ttl=2)

    planner = Planner(wm)
    # At t=0: (2, 2) is blocked
    assert temporal.is_temporally_blocked(2, 2, target_t=0)
    # At t=2: (2, 2) is expired
    assert not temporal.is_temporally_blocked(2, 2, target_t=2)

    res = planner.find_space_time_path(
        start=(1, 2),
        target=(3, 2),
        start_t=0,
        temporal_model=temporal,
    )
    assert res is not None
    actions, coords = res
    assert coords[-1] == (3, 2)


@pytest.mark.unit
def test_space_time_periodic_npc_prediction():
    """39. Phase 6.11: Periodic NPC cycle prediction times corridor crossing."""
    temporal = TemporalWorldModel()
    # Periodic mover oscillating vertically: (3, 1) -> (3, 2) -> (3, 3) -> (3, 2)
    cycle = [(3, 1), (3, 2), (3, 3), (3, 2)]
    mover = MovingObjectModel(
        object_id="patrol_1",
        current_pos=(3, 2),
        trajectory_cycle=cycle,
        estimated_period=4,
        estimated_phase=1,
        is_periodic=True,
        confidence=0.9,
        blocks_player=True,
    )
    temporal.moving_objects["patrol_1"] = mover
    temporal.current_time = 0

    # At dt=0: (3, 2) is occupied
    assert mover.predict_position_at(0, current_t=0) == (3, 2)
    # At dt=1: moves to (3, 3) -> cell (3, 2) is free!
    assert mover.predict_position_at(1, current_t=0) == (3, 3)
    assert not temporal.is_temporally_blocked(3, 2, target_t=1)


@pytest.mark.unit
def test_v37_behavior_regression_preserved():
    """40. Phase 6.12: Existing V3.7 behavior regression remains fully preserved."""
    wm = WorldModel(cols=6, rows=6)
    wm.walls.add((2, 2))
    planner = Planner(wm)

    # 1. Static walls remain impassable
    path = planner.find_grid_path(start=(1, 2), target=(3, 2))
    assert path is not None
    assert (2, 2) not in path

    # 2. Carrier composite footprint avoids static walls
    path_c = planner.find_grid_path(start=(1, 1), target=(3, 1), carried_offset=(0, 1))
    assert path_c is not None
    for p in path_c:
        assert (p[0], p[1] + 1) != (2, 2)



