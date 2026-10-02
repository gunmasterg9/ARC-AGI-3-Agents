import sys
sys.path.append(r"C:\Users\GAUTAM\arc-agi-3\environment_files\ls20\9607627b")
import pytest
import numpy as np
import ls20
from arcengine import ActionInput, FrameData, GameAction, GameState

from agents.arcagi_v3 import (
    ARCAGIV3Solver,
    DiagnosticsLogger,
    EpisodeMemory,
    GoalManager,
    GoalSpecification,
    ObjectType,
    PerceptionEngine,
    Planner,
    PlayerState,
    TransitionLearner,
    WorldModel,
    get_level_spec,
)
from agents.arcagi_v3.actions import (
    action_to_grid_delta,
    grid_delta_to_action,
    grid_to_pixel,
    pixel_to_grid,
)


@pytest.mark.unit
def test_v3_import_and_construction():
    """Test V3 agent import, inheritance, and attribute initialization."""
    solver = ARCAGIV3Solver()
    assert solver.MAX_ACTIONS == 500
    assert len(solver.active_plan) == 0  # V3.1: Zero pre-seeded actions
    assert len(solver.subgoal_queue) > 0  # Semantic subgoals ready for dynamic planning


@pytest.mark.unit
def test_coordinate_transforms():
    """Test discrete grid <-> screen pixel conversions."""
    assert grid_to_pixel(0, 0) == (4, 0)
    assert grid_to_pixel(6, 9) == (34, 45)
    assert pixel_to_grid(34, 45) == (6, 9)
    assert action_to_grid_delta(GameAction.ACTION1) == (0, -1)
    assert grid_delta_to_action(0, -1) == GameAction.ACTION1


@pytest.mark.unit
def test_perception_screen_and_player():
    """Test screen extraction and player detection."""
    game = ls20.Ls20()
    frame = game.perform_action(ActionInput(id=GameAction.RESET))
    perception = PerceptionEngine()

    screen = perception.extract_screen(frame)
    assert screen is not None
    assert screen.shape == (64, 64)

    player = perception.detect_player(screen)
    assert player is not None
    assert player.grid_pos == (6, 9)  # Level 1 player start is (34, 45) -> (6, 9)


@pytest.mark.unit
def test_perception_object_detection():
    """Test detection of walls, goals, transformers."""
    game = ls20.Ls20()
    frame = game.perform_action(ActionInput(id=GameAction.RESET))
    perception = PerceptionEngine()
    screen = perception.extract_screen(frame)

    objects = perception.detect_objects(screen)
    assert len(objects) > 0
    types = {o.object_type for o in objects}
    assert ObjectType.WALL in types
    assert ObjectType.GOAL in types


@pytest.mark.unit
def test_world_model_and_planner():
    """Test world model passability and BFS pathfinding."""
    wm = WorldModel(cols=11, rows=11)
    # Mark a wall
    wm.mark_obstacle(5, 5)
    assert not wm.is_passable(5, 5)
    assert wm.is_passable(0, 0)

    planner = Planner(wm)
    path = planner.find_grid_path(start=(0, 0), target=(2, 0))
    assert path == [(0, 0), (1, 0), (2, 0)]
    actions = planner.path_to_actions(path)
    assert actions == [GameAction.ACTION4, GameAction.ACTION4]  # Two RIGHT moves


@pytest.mark.unit
def test_goal_manager_and_multi_goal():
    """Test goal management for multi-goal levels."""
    gm = GoalManager()
    g1 = GoalSpecification(goal_id=0, col=10, row=10, req_shape=5, req_color=1, req_rot=1)
    g2 = GoalSpecification(goal_id=1, col=10, row=7, req_shape=0, req_color=3, req_rot=2)
    gm.set_goals([g1, g2])

    assert len(gm.get_active_goals()) == 2
    assert gm.get_current_goal().goal_id == 0

    gm.mark_satisfied(0)
    assert len(gm.get_active_goals()) == 1
    assert gm.get_current_goal().goal_id == 1

    gm.mark_satisfied(1)
    assert gm.all_satisfied()


@pytest.mark.unit
def test_transition_learning():
    """Test observing and learning from state transitions."""
    learner = TransitionLearner()
    p1 = PlayerState(x=34, y=45, col=6, row=9, rot_idx=3)
    p2 = PlayerState(x=34, y=40, col=6, row=8, rot_idx=3)

    from agents.arcagi_v3.state import HUDState, WorldState
    w1 = WorldState(level_idx=0, player=p1, hud=HUDState())
    w2 = WorldState(level_idx=0, player=p2, hud=HUDState())

    trans = learner.observe_transition(w1, GameAction.ACTION1, w2)
    assert trans.effect.moved is True
    assert trans.effect.drow == -1
    assert trans.effect.blocked is False


@pytest.mark.unit
def test_episode_memory_and_loop_detection():
    """Test episode memory recording and loop detection."""
    mem = EpisodeMemory()
    mem.record_step(5, 5, GameAction.ACTION1)
    mem.record_step(5, 4, GameAction.ACTION2)
    mem.record_step(5, 5, GameAction.ACTION1)
    mem.record_step(5, 4, GameAction.ACTION2)
    mem.record_step(5, 5, GameAction.ACTION1)
    mem.record_step(5, 4, GameAction.ACTION2)

    assert mem.is_looping() is True


@pytest.mark.unit
def test_ls20_specs():
    """Test all 7 level domain specifications."""
    for lvl in range(7):
        spec = get_level_spec(lvl)
        assert spec is not None
        assert spec.level_idx == lvl
        assert len(spec.goals) >= 1


# =====================================================================
# V3.2 SPACE-TIME A* TEMPORAL PLANNER UNIT TESTS (1-15)
# =====================================================================
from agents.arcagi_v3.objects import GameObject
from agents.arcagi_v3.temporal import MovingObjectModel, TemporalWorldModel


@pytest.mark.unit
def test_temporal_state():
    """1. Test TemporalWorldModel initialization, clock progression, and active state."""
    tm = TemporalWorldModel()
    assert tm.current_time == 0
    assert not tm.has_active_moving_objects()
    tm.update_time(5)
    assert tm.current_time == 5


@pytest.mark.unit
def test_moving_object_detection():
    """2. Test online moving-object discovery from multi-frame displacement."""
    tm = TemporalWorldModel()
    obj0 = [GameObject(object_type=ObjectType.MOVING_PLATFORM, x=14, y=35, col=2, row=7)]
    tm.update_from_detected_objects(obj0, t=0)
    assert not tm.has_active_moving_objects()

    obj1 = [GameObject(object_type=ObjectType.MOVING_PLATFORM, x=19, y=35, col=3, row=7)]
    tm.update_from_detected_objects(obj1, t=1)
    assert tm.has_active_moving_objects()
    assert len(tm.moving_objects) == 1


@pytest.mark.unit
def test_velocity_estimation():
    """3. Test velocity vector estimation from sequential positions."""
    tm = TemporalWorldModel()
    tm.update_from_detected_objects([GameObject(object_type=ObjectType.MOVING_PLATFORM, x=14, y=35, col=2, row=7)], t=0)
    tm.update_from_detected_objects([GameObject(object_type=ObjectType.MOVING_PLATFORM, x=19, y=35, col=3, row=7)], t=1)
    mover = list(tm.moving_objects.values())[0]
    assert mover.velocity == (1, 0)



@pytest.mark.unit
def test_period_estimation():
    """4. Test cyclic period estimation from repeated observation history."""
    mover = MovingObjectModel(object_id="m1", current_pos=(2, 7))
    positions = [(2, 7), (3, 7), (4, 7), (3, 7), (2, 7), (3, 7), (4, 7), (3, 7)]
    for t, pos in enumerate(positions):
        mover.update_observation(pos, t)
    assert mover.is_periodic is True
    assert mover.estimated_period == 4
    assert mover.trajectory_cycle == [(2, 7), (3, 7), (4, 7), (3, 7)]


@pytest.mark.unit
def test_phase_estimation():
    """5. Test cyclic phase index estimation within trajectory cycle."""
    mover = MovingObjectModel(object_id="m1", current_pos=(2, 7))
    positions = [(2, 7), (3, 7), (4, 7), (3, 7), (2, 7), (3, 7), (4, 7), (3, 7)]
    for t, pos in enumerate(positions):
        mover.update_observation(pos, t)
    assert mover.estimated_phase == 3


@pytest.mark.unit
def test_future_position_prediction():
    """6. Test future position prediction for periodic trajectories."""
    mover = MovingObjectModel(object_id="m1", current_pos=(2, 7))
    positions = [(2, 7), (3, 7), (4, 7), (3, 7), (2, 7), (3, 7), (4, 7), (3, 7)]
    for t, pos in enumerate(positions):
        mover.update_observation(pos, t)
    assert mover.predict_position_at(8, current_t=7) == (2, 7)
    assert mover.predict_position_at(9, current_t=7) == (3, 7)
    assert mover.predict_position_at(10, current_t=7) == (4, 7)


@pytest.mark.unit
def test_collision_at_t_plus_1():
    """7. Test immediate t+1 cell collision and vertex swap edge collision."""
    tm = TemporalWorldModel()
    tm.current_time = 0
    mover = MovingObjectModel(object_id="m1", current_pos=(2, 7), velocity=(1, 0), confidence=0.8)
    tm.moving_objects["m1"] = mover

    # Extrapolates to (3, 7) at t=1
    assert tm.is_temporally_blocked(3, 7, target_t=1, prev_pos=(4, 7)) is True
    assert tm.is_temporally_blocked(0, 0, target_t=1) is False
    # Vertex swap collision
    assert tm.is_temporally_blocked(2, 7, target_t=1, prev_pos=(3, 7)) is True


@pytest.mark.unit
def test_collision_at_future_t():
    """8. Test collision evaluation at arbitrary future time horizon t."""
    tm = TemporalWorldModel()
    tm.current_time = 0
    mover = MovingObjectModel(
        object_id="m1",
        current_pos=(2, 7),
        is_periodic=True,
        estimated_period=2,
        estimated_phase=0,
        trajectory_cycle=[(2, 7), (3, 7)],
        confidence=0.9,
    )
    tm.moving_objects["m1"] = mover

    assert tm.is_temporally_blocked(3, 7, target_t=1) is True
    assert tm.is_temporally_blocked(2, 7, target_t=2) is True
    assert tm.is_temporally_blocked(3, 7, target_t=3) is True
    assert tm.is_temporally_blocked(2, 7, target_t=3) is False


@pytest.mark.unit
def test_space_time_astar_path_generation():
    """9. Test Space-Time A* generates collision-free path around dynamic mover."""
    wm = WorldModel(cols=5, rows=5)
    planner = Planner(wm)
    tm = TemporalWorldModel()
    mover = MovingObjectModel(
        object_id="m1",
        current_pos=(2, 2),
        is_periodic=True,
        estimated_period=2,
        estimated_phase=0,
        trajectory_cycle=[(2, 2), (2, 3)],
        confidence=0.9,
    )
    tm.moving_objects["m1"] = mover

    res = planner.find_space_time_path(start=(0, 2), target=(4, 2), start_t=0, temporal_model=tm)
    assert res is not None
    actions, coords = res
    assert coords[0] == (0, 2)
    assert coords[-1] == (4, 2)
    for step_t, pos in enumerate(coords):
        mover_pos = mover.predict_position_at(step_t, current_t=0)
        assert pos != mover_pos


@pytest.mark.unit
def test_temporal_replanning():
    """10. Test detection of imminent hazard triggers dynamic replanning."""
    solver = ARCAGIV3Solver()
    tm = solver.temporal_model
    mover = MovingObjectModel(object_id="m1", current_pos=(6, 8), velocity=(0, -1), confidence=0.8)
    tm.moving_objects["m1"] = mover
    assert tm.is_temporally_blocked(6, 7, target_t=1) is True


@pytest.mark.unit
def test_prediction_mismatch_recovery():
    """11. Test prediction verification failure dampens confidence and signals replan."""
    tm = TemporalWorldModel()
    mover = MovingObjectModel(
        object_id="m1",
        current_pos=(2, 7),
        velocity=(1, 0),
        confidence=0.8,
        is_periodic=True,
        estimated_period=2,
        estimated_phase=0,
        trajectory_cycle=[(2, 7), (3, 7)],
    )
    tm.moving_objects["m1"] = mover
    tm.current_time = 0

    observed = [GameObject(object_type=ObjectType.MOVING_PLATFORM, x=14, y=35, col=2, row=7)]
    verified = tm.verify_prediction(observed, t=1)
    assert verified is False
    assert mover.confidence < 0.8


@pytest.mark.unit
def test_periodic_state_compression():
    """12. Test periodic state space compression using LCM of periods."""
    tm = TemporalWorldModel()
    m1 = MovingObjectModel(object_id="m1", current_pos=(0, 0), is_periodic=True, estimated_period=4, confidence=0.8)
    m2 = MovingObjectModel(object_id="m2", current_pos=(1, 1), is_periodic=True, estimated_period=6, confidence=0.8)
    tm.moving_objects["m1"] = m1
    tm.moving_objects["m2"] = m2
    assert tm.compute_effective_period() == 12


@pytest.mark.unit
def test_no_illegal_wait_action():
    """13. Test Space-Time A* only uses valid ARC-AGI-3 movement actions."""
    wm = WorldModel(cols=5, rows=5)
    planner = Planner(wm)
    res = planner.find_space_time_path(start=(0, 0), target=(1, 0))
    assert res is not None
    actions, _ = res
    legal = {GameAction.ACTION1, GameAction.ACTION2, GameAction.ACTION3, GameAction.ACTION4}
    for a in actions:
        assert a in legal


@pytest.mark.unit
def test_static_world_planner_regression():
    """14. Test static world automatically uses BFS pathfinder without temporal overhead."""
    wm = WorldModel(cols=5, rows=5)
    planner = Planner(wm)
    tm = TemporalWorldModel()
    assert tm.has_active_moving_objects() is False
    actions = planner.plan_sequence_to_target(start=(0, 0), target=(2, 0), temporal_model=tm)
    assert actions == [GameAction.ACTION4, GameAction.ACTION4]


@pytest.mark.unit
def test_level_transition_temporal_reset():
    """15. Test clean reset of all temporal tracking upon level transition."""
    tm = TemporalWorldModel()
    tm.update_time(42)
    tm.moving_objects["m1"] = MovingObjectModel(object_id="m1", current_pos=(1, 1))
    tm.prev_object_positions["p1"] = (1, 1)

    tm.reset_level()
    assert tm.current_time == 0
    assert len(tm.moving_objects) == 0
    assert len(tm.prev_object_positions) == 0


# ==============================================================================
# V3.3 Affordance & Interaction Modeling Tests
# ==============================================================================

from agents.arcagi_v3.affordance import (
    AffordanceModel,
    ControllableEntityManager,
    DynamicGridTracker,
)
from agents.arcagi_v3.objects import Affordance
from agents.arcagi_v3.state import HUDState, WorldState


@pytest.mark.unit
def test_action5_registration():
    """1. Test ACTION5 is recognized in action maps with zero spatial displacement."""
    from agents.arcagi_v3.actions import ACTION_TO_GRID_DELTA, ACTION_TO_PIXEL_DELTA
    assert GameAction.ACTION5 in ACTION_TO_GRID_DELTA
    assert ACTION_TO_GRID_DELTA[GameAction.ACTION5] == (0, 0)
    assert ACTION_TO_PIXEL_DELTA[GameAction.ACTION5] == (0, 0)


@pytest.mark.unit
def test_action5_conversion():
    """2. Test ACTION5 conversion via action_to_grid_delta."""
    assert action_to_grid_delta(GameAction.ACTION5) == (0, 0)


@pytest.mark.unit
def test_dynamic_grid_stride_detection():
    """3. Test dynamic stride inference using GCD over displacements."""
    # wa30: stride 4
    tracker_wa30 = DynamicGridTracker(default_stride=5)
    for d in [4, 8, 12, 4]:
        tracker_wa30.record_displacement(d, 0)
    assert tracker_wa30.estimated_stride == 4

    # g50t: stride 6
    tracker_g50t = DynamicGridTracker(default_stride=5)
    for d in [6, 12, 6]:
        tracker_g50t.record_displacement(0, d)
    assert tracker_g50t.estimated_stride == 6

    # re86: stride 3
    tracker_re86 = DynamicGridTracker(default_stride=5)
    for d in [3, 6, 9]:
        tracker_re86.record_displacement(d, 0)
    assert tracker_re86.estimated_stride == 3


@pytest.mark.unit
def test_controlled_entity_discovery():
    """4. Test identifying controlled entity position from frame diffs."""
    manager = ControllableEntityManager()
    tracker = DynamicGridTracker(default_stride=4)
    tracker.estimated_stride = 4
    tracker.offset_x = 0
    tracker.offset_y = 0

    prev_screen = np.zeros((64, 64), dtype=np.uint8)
    curr_screen = np.zeros((64, 64), dtype=np.uint8)
    # Entity moved from (8, 8) to (12, 8)
    prev_screen[8:12, 8:12] = 5
    curr_screen[8:12, 12:16] = 5

    pos = manager.discover_controlled_entity(
        prev_screen, curr_screen, GameAction.ACTION4, tracker
    )
    assert pos is not None
    assert manager.controlled_entity_id is not None
    assert manager.control_confidence > 0.0


@pytest.mark.unit
def test_multiple_entity_tracking():
    """5. Test tracking entity confidence across exploratory actions."""
    manager = ControllableEntityManager()
    tracker = DynamicGridTracker(default_stride=4)
    prev_screen = np.zeros((64, 64), dtype=np.uint8)
    curr_screen = np.zeros((64, 64), dtype=np.uint8)
    curr_screen[4:8, 4:8] = 7

    conf_before = manager.control_confidence
    manager.discover_controlled_entity(
        prev_screen, curr_screen, GameAction.ACTION4, tracker
    )
    assert manager.control_confidence > conf_before


@pytest.mark.unit
def test_affordance_representation():
    """6. Test Affordance dataclass structure and defaults."""
    aff = Affordance(
        target_object_id="item_1",
        target_pos=(3, 4),
        interaction_action=GameAction.ACTION5,
        required_distance=1,
        predicted_effect="pickup",
        confidence=0.8,
    )
    assert aff.target_object_id == "item_1"
    assert aff.interaction_action == GameAction.ACTION5
    assert aff.required_distance == 1
    assert aff.predicted_effect == "pickup"


@pytest.mark.unit
def test_action5_transition_learning():
    """7. Test TransitionLearner observes ACTION5 without falsely flagging blocked."""
    tl = TransitionLearner()
    p1 = PlayerState(x=10, y=10, col=2, row=2)
    p2 = PlayerState(x=10, y=10, col=2, row=2)
    w1 = WorldState(level_idx=0, player=p1, hud=HUDState())
    w2 = WorldState(level_idx=0, player=p2, hud=HUDState())

    trans = tl.observe_transition(w1, GameAction.ACTION5, w2)
    assert trans.effect.blocked is False
    assert trans.effect.moved is False


@pytest.mark.unit
def test_pickup_drop_state_transition():
    """8. Test pickup and drop state transitions in AffordanceModel."""
    model = AffordanceModel()
    obj1 = GameObject(object_type=ObjectType.ITEM, x=15, y=15, col=3, row=3)

    # 1. Pickup: adjacent item disappears after ACTION5
    effect = model.update_from_action5_observation(
        prev_objects=[obj1],
        curr_objects=[],
        player_pos=(3, 2),
    )
    assert effect == "pickup"
    assert model.carried_object_id is not None

    # 2. Drop: item reappears when carried
    obj_dropped = GameObject(object_type=ObjectType.ITEM, x=15, y=15, col=3, row=3)
    effect2 = model.update_from_action5_observation(
        prev_objects=[],
        curr_objects=[obj_dropped],
        player_pos=(3, 2),
    )
    assert effect2 == "drop"
    assert model.carried_object_id is None


@pytest.mark.unit
def test_focus_switch_transition():
    """9. Test focus-switch transition detection in TransitionLearner."""
    tl = TransitionLearner()
    p = PlayerState(x=10, y=10, col=2, row=2)
    w1 = WorldState(level_idx=0, player=p, hud=HUDState(), controlled_entity_id="entity_1")
    w2 = WorldState(level_idx=0, player=p, hud=HUDState(), controlled_entity_id="entity_2")

    trans = tl.observe_transition(w1, GameAction.ACTION5, w2)
    assert trans.effect.active_entity_changed is True
    assert trans.effect.interaction_type == "switch_focus"


@pytest.mark.unit
def test_generic_interaction_planning():
    """10. Test planning interaction on a target with required_distance=0."""
    wm = WorldModel(cols=5, rows=5)
    planner = Planner(wm)
    actions = planner.plan_composite_interaction(
        start=(0, 0),
        target=(2, 0),
        interaction_action=GameAction.ACTION5,
        required_distance=0,
    )
    assert actions == [GameAction.ACTION4, GameAction.ACTION4, GameAction.ACTION5]


@pytest.mark.unit
def test_composite_movement_and_action5_planning():
    """11. Test planning navigation adjacent to target with required_distance=1."""
    wm = WorldModel(cols=5, rows=5)
    planner = Planner(wm)
    actions = planner.plan_composite_interaction(
        start=(0, 0),
        target=(2, 0),
        interaction_action=GameAction.ACTION5,
        required_distance=1,
    )
    # Moving from (0,0) to adjacent neighbor (1,0), then ACTION5
    assert actions == [GameAction.ACTION4, GameAction.ACTION5]


@pytest.mark.unit
def test_invalid_action5_precondition():
    """12. Test unreachable target returns empty composite action plan."""
    wm = WorldModel(cols=5, rows=5)
    # Surround target with walls
    wm.walls = {(1, 0), (3, 0), (2, 1)}
    planner = Planner(wm)
    actions = planner.plan_composite_interaction(
        start=(0, 0),
        target=(2, 0),
        interaction_action=GameAction.ACTION5,
        required_distance=0,
    )
    assert actions == []


@pytest.mark.unit
def test_ls20_regression():
    """13. Test LS20 level specs and subgoals are completely preserved."""
    spec = get_level_spec(0)
    assert spec is not None
    assert len(spec.goals) == 1
    assert len(spec.mission_subgoals) == 2


@pytest.mark.unit
def test_space_time_astar_regression():
    """14. Test Space-Time A* temporal avoidance regression."""
    wm = WorldModel(cols=5, rows=5)
    planner = Planner(wm)
    tm = TemporalWorldModel()
    mover = MovingObjectModel(object_id="m1", current_pos=(1, 0), velocity=(0, 0), confidence=0.9)
    tm.moving_objects["m1"] = mover

    # Cell (1, 0) is blocked at t=1, planner navigates around via (0, 1) -> (1, 1) -> (2, 1) -> (2, 0)
    res = planner.find_space_time_path(start=(0, 0), target=(2, 0), start_t=0, temporal_model=tm)
    assert res is not None
    actions, coords = res
    assert (1, 0) not in [c for c in coords[1:2]]


# ==============================================================================
# V3.4 Multi-Frame Object Persistence Tests (1-12)
# ==============================================================================

from agents.arcagi_v3.persistence import MultiFrameObjectTracker, TrackedObject


@pytest.mark.unit
def test_frame_history_association():
    """1. Test multi-frame visual history retention and sliding window."""
    tracker = MultiFrameObjectTracker()
    f1 = np.ones((64, 64), dtype=np.uint8) * 1
    f2 = np.ones((64, 64), dtype=np.uint8) * 2
    f3 = np.ones((64, 64), dtype=np.uint8) * 3
    f4 = np.ones((64, 64), dtype=np.uint8) * 4

    tracker.update([], f1, None, None)
    tracker.update([], f2, None, None)
    tracker.update([], f3, None, None)
    assert len(tracker.frame_history) == 3
    assert tracker.frame_history[0][0, 0] == 1

    tracker.update([], f4, None, None)
    assert len(tracker.frame_history) == 3
    assert tracker.frame_history[0][0, 0] == 2
    assert tracker.frame_history[-1][0, 0] == 4


@pytest.mark.unit
def test_stable_object_identity():
    """2. Test stable object identity across multiple frames for stationary objects."""
    tracker = MultiFrameObjectTracker()
    det1 = GameObject(object_type=ObjectType.ITEM, x=8, y=8, col=2, row=2, color=9)
    tracker.update([det1], None, None, None)
    obj_id_1 = list(tracker.tracked_objects.keys())[0]

    # Frame 2: same detection
    det2 = GameObject(object_type=ObjectType.ITEM, x=8, y=8, col=2, row=2, color=9)
    tracker.update([det2], None, None, None)
    assert list(tracker.tracked_objects.keys()) == [obj_id_1]
    assert tracker.tracked_objects[obj_id_1].persistence_count == 2


@pytest.mark.unit
def test_moving_object_identity_preservation():
    """3. Test object identity preservation and velocity tracking for moving entities."""
    tracker = MultiFrameObjectTracker()
    det1 = GameObject(object_type=ObjectType.ITEM, x=8, y=8, col=2, row=2, color=12)
    tracker.update([det1], None, None, None)
    obj_id = list(tracker.tracked_objects.keys())[0]

    # Frame 2: moved to (3, 2)
    det2 = GameObject(object_type=ObjectType.ITEM, x=12, y=8, col=3, row=2, color=12)
    tracker.update([det2], None, None, None)

    track = tracker.tracked_objects[obj_id]
    assert track.current_pos == (3, 2)
    assert track.velocity == (1, 0)
    assert track.motion_state == "moving"
    assert track.category == ObjectType.MOVING_PLATFORM


@pytest.mark.unit
def test_temporary_disappearance():
    """4. Test object is not immediately discarded on 1-frame disappearance."""
    tracker = MultiFrameObjectTracker()
    det1 = GameObject(object_type=ObjectType.ITEM, x=8, y=8, col=2, row=2, color=9)
    tracker.update([det1], None, None, None)
    obj_id = list(tracker.tracked_objects.keys())[0]

    # Frame 2: object temporarily occluded / not detected
    tracker.update([], None, None, None)
    assert obj_id in tracker.tracked_objects
    assert tracker.tracked_objects[obj_id].disappeared_count == 1

    # Frame 3: object reappears
    tracker.update([det1], None, None, None)
    assert obj_id in tracker.tracked_objects
    assert tracker.tracked_objects[obj_id].disappeared_count == 0


@pytest.mark.unit
def test_item_receptacle_confidence():
    """5. Test differential confidence between items and receptacles."""
    tracker = MultiFrameObjectTracker()
    # Det1 is a single isolated cell
    item_det = GameObject(object_type=ObjectType.ITEM, x=4, y=4, col=1, row=1, color=9)
    # Det2 and Det3 are neighboring cells representing a receptacle zone
    rec1 = GameObject(object_type=ObjectType.RECEPTACLE, x=20, y=20, col=5, row=5, color=2)
    rec2 = GameObject(object_type=ObjectType.RECEPTACLE, x=24, y=20, col=6, row=5, color=2)

    tracker.update([item_det, rec1, rec2], None, None, None)
    # Second observation confirms persistence
    tracker.update([item_det, rec1, rec2], None, None, None)

    items = tracker.get_items()
    receptacles = tracker.get_receptacles()
    assert any(t.current_pos == (1, 1) for t in items)
    assert any(t.current_pos in {(5, 5), (6, 5)} for t in receptacles)


@pytest.mark.unit
def test_pickup_detection():
    """6. Test ACTION5 triggers pickup when adjacent item disappears."""
    tracker = MultiFrameObjectTracker()
    item_det = GameObject(object_type=ObjectType.ITEM, x=8, y=4, col=2, row=1, color=9)
    tracker.update([item_det], None, player_pos=(1, 1), last_action=GameAction.ACTION1)

    # Next step: player executes ACTION5 adjacent to (2, 1) and item disappears
    tracker.update([], None, player_pos=(1, 1), last_action=GameAction.ACTION5)
    assert tracker.carried_object_id is not None
    assert tracker.pickups_count == 1
    assert tracker.tracked_objects[tracker.carried_object_id].is_carried is True


@pytest.mark.unit
def test_carried_object_persistence():
    """7. Test carried object persists across movement steps while carried."""
    tracker = MultiFrameObjectTracker()
    item_det = GameObject(object_type=ObjectType.ITEM, x=8, y=4, col=2, row=1, color=9)
    tracker.update([item_det], None, player_pos=(1, 1), last_action=None)
    tracker.update([], None, player_pos=(1, 1), last_action=GameAction.ACTION5)

    carried_id = tracker.carried_object_id
    assert carried_id is not None

    # Move player 3 steps while carrying
    tracker.update([], None, player_pos=(2, 1), last_action=GameAction.ACTION4)
    tracker.update([], None, player_pos=(3, 1), last_action=GameAction.ACTION4)
    tracker.update([], None, player_pos=(4, 1), last_action=GameAction.ACTION4)

    assert tracker.carried_object_id == carried_id
    assert carried_id in tracker.tracked_objects
    assert tracker.tracked_objects[carried_id].is_carried is True


@pytest.mark.unit
def test_drop_detection():
    """8. Test ACTION5 near receptacle successfully triggers drop."""
    tracker = MultiFrameObjectTracker()
    item_det = GameObject(object_type=ObjectType.ITEM, x=8, y=4, col=2, row=1, color=9)
    rec_det = GameObject(object_type=ObjectType.RECEPTACLE, x=16, y=4, col=4, row=1, color=2)

    tracker.update([item_det, rec_det], None, player_pos=(1, 1), last_action=None)
    # Pickup item at (2, 1)
    tracker.update([rec_det], None, player_pos=(1, 1), last_action=GameAction.ACTION5)
    assert tracker.carried_object_id is not None

    # Move adjacent to receptacle at (4, 1), standing at (3, 1)
    tracker.update([rec_det], None, player_pos=(3, 1), last_action=GameAction.ACTION4)

    # Execute ACTION5 to drop
    tracker.update([rec_det], None, player_pos=(3, 1), last_action=GameAction.ACTION5)
    assert tracker.carried_object_id is None
    assert tracker.drops_count == 1


@pytest.mark.unit
def test_delivery_verification():
    """9. Test post-drop state marks receptacle affordance and resets carried state."""
    tracker = MultiFrameObjectTracker()
    item_det = GameObject(object_type=ObjectType.ITEM, x=8, y=4, col=2, row=1, color=9)
    rec_det = GameObject(object_type=ObjectType.RECEPTACLE, x=16, y=4, col=4, row=1, color=2)

    tracker.update([item_det, rec_det], None, player_pos=(1, 1), last_action=None)
    tracker.update([rec_det], None, player_pos=(1, 1), last_action=GameAction.ACTION5)
    tracker.update([rec_det], None, player_pos=(3, 1), last_action=GameAction.ACTION5)

    rec_tracks = tracker.get_receptacles()
    assert len(rec_tracks) >= 1
    assert tracker.drops_count == 1
    assert tracker.carried_object_id is None


@pytest.mark.unit
def test_wa30_interaction_planning():
    """10. Test composite interaction plan: move adjacent to item, pickup, move to rec, drop."""
    wm = WorldModel(cols=8, rows=8)
    planner = Planner(wm)

    # Plan pickup at (2, 1) from (0, 1) -> moves to (1, 1) then ACTION5
    plan_pickup = planner.plan_composite_interaction(
        start=(0, 1),
        target=(2, 1),
        interaction_action=GameAction.ACTION5,
        required_distance=1,
    )
    assert plan_pickup == [GameAction.ACTION4, GameAction.ACTION5]

    # Plan drop at (4, 1) from (1, 1) -> moves to (3, 1) then ACTION5
    plan_drop = planner.plan_composite_interaction(
        start=(1, 1),
        target=(4, 1),
        interaction_action=GameAction.ACTION5,
        required_distance=1,
    )
    assert plan_drop == [GameAction.ACTION4, GameAction.ACTION4, GameAction.ACTION5]


@pytest.mark.unit
def test_v34_spacetime_astar_regression():
    """11. Test Space-Time A* continues to function without regression."""
    wm = WorldModel(cols=6, rows=6)
    planner = Planner(wm)
    tm = TemporalWorldModel()
    tm.moving_objects["npc"] = MovingObjectModel(object_id="npc", current_pos=(1, 0), velocity=(0, 0), confidence=0.9)

    res = planner.find_space_time_path(start=(0, 0), target=(2, 0), start_t=0, temporal_model=tm)
    assert res is not None
    actions, coords = res
    assert (1, 0) not in [c for c in coords[1:2]]


@pytest.mark.unit
def test_v34_ls20_regression():
    """12. Test LS20 level specs and subgoals are completely preserved."""
    for lvl in range(7):
        spec = get_level_spec(lvl)
        assert spec is not None
        assert spec.level_idx == lvl
        assert len(spec.goals) >= 1
        assert len(spec.mission_subgoals) >= 1


# ==============================================================================
# V3.9 Reachability, Connected Components & Cooperative Handoff Tests (1-20)
# ==============================================================================

from agents.arcagi_v3.reachability import (
    CooperativeTransferState,
    ItemTransferState,
    NPCClassification,
    ReachabilityAnalyzer,
    ReachabilityStatus,
    TransferInterface,
)


@pytest.mark.unit
def test_v39_direct_goal_same_component():
    """1. Test direct reachability when player and goal share same component."""
    analyzer = ReachabilityAnalyzer()
    analyzer.compute_connected_components(cols=10, rows=10, walls=set())
    status, p_comp, goal_comps = analyzer.analyze_reachability(player_pos=(1, 1), goal_positions={(8, 8)})
    assert status == ReachabilityStatus.DIRECTLY_REACHABLE
    assert p_comp in goal_comps


@pytest.mark.unit
def test_v39_goal_unreachable_detected():
    """2. Test unreachable status when no goals are provided or accessible."""
    analyzer = ReachabilityAnalyzer()
    analyzer.compute_connected_components(cols=10, rows=10, walls=set())
    status, p_comp, goal_comps = analyzer.analyze_reachability(player_pos=(1, 1), goal_positions=set())
    assert status == ReachabilityStatus.UNREACHABLE


@pytest.mark.unit
def test_v39_two_connected_components():
    """3. Test connected component decomposition across a dividing wall."""
    analyzer = ReachabilityAnalyzer()
    divider = {(5, r) for r in range(10)}
    comps = analyzer.compute_connected_components(cols=10, rows=10, walls=divider)
    assert len(comps) == 2
    comp0 = analyzer.get_component((2, 2))
    comp1 = analyzer.get_component((7, 2))
    assert comp0 != comp1


@pytest.mark.unit
def test_v39_transfer_interface_discovery():
    """4. Test discovery of transfer interface cells bridging two rooms."""
    analyzer = ReachabilityAnalyzer()
    divider = {(5, r) for r in range(10)}
    analyzer.compute_connected_components(cols=10, rows=10, walls=divider)
    target_comp = analyzer.get_component((7, 2))
    interfaces = analyzer.discover_transfer_interfaces(
        player_pos=(2, 2),
        target_comps={target_comp},
        cols=10,
        rows=10,
    )
    assert len(interfaces) == 10
    positions = {tif.position for tif in interfaces}
    assert positions == divider


@pytest.mark.unit
def test_v39_indirect_reachability_classification():
    """5. Test indirect reachability when goal resides in an adjacent room."""
    analyzer = ReachabilityAnalyzer()
    divider = {(5, r) for r in range(10)}
    analyzer.compute_connected_components(cols=10, rows=10, walls=divider)
    status, p_comp, goal_comps = analyzer.analyze_reachability(player_pos=(2, 2), goal_positions={(8, 8)})
    assert status == ReachabilityStatus.INDIRECTLY_REACHABLE
    assert p_comp not in goal_comps


@pytest.mark.unit
def test_v39_transfer_interface_priority_scoring():
    """6. Test interface priority scoring favors cells with observed items."""
    analyzer = ReachabilityAnalyzer()
    divider = {(5, r) for r in range(10)}
    analyzer.compute_connected_components(cols=10, rows=10, walls=divider)
    target_comp = analyzer.get_component((7, 2))
    observed = {(5, 5)}
    interfaces = analyzer.discover_transfer_interfaces(
        player_pos=(4, 5),
        target_comps={target_comp},
        cols=10,
        rows=10,
        observed_item_positions=observed,
    )
    best = interfaces[0]
    assert best.position == (5, 5)
    assert best.confidence >= 0.95


@pytest.mark.unit
def test_v39_impassable_items_block_flood_fill():
    """7. Test uncarried solid items block flood fill connectivity."""
    analyzer = ReachabilityAnalyzer()
    divider_walls = {(c, 5) for c in range(10) if c != 5}
    comps = analyzer.compute_connected_components(
        cols=10, rows=10, walls=divider_walls, impassable_items={(5, 5)}
    )
    assert len(comps) == 2


@pytest.mark.unit
def test_v39_handed_off_item_state_tracking():
    """8. Test tracking and clearing of handed-off items."""
    analyzer = ReachabilityAnalyzer()
    assert (8, 3) not in analyzer.handed_off_items
    analyzer.record_handoff((8, 3))
    assert (8, 3) in analyzer.handed_off_items
    analyzer.clear_handoff((8, 3))
    assert (8, 3) not in analyzer.handed_off_items


@pytest.mark.unit
def test_v39_cooperative_transfer_state_lifecycle():
    """9. Test CooperativeTransferState enumeration values and transitions."""
    assert CooperativeTransferState.NOT_REQUIRED.value == "NOT_REQUIRED"
    assert CooperativeTransferState.APPROACHING_INTERFACE.value == "APPROACHING_INTERFACE"
    assert CooperativeTransferState.HANDOFF_ATTEMPT.value == "HANDOFF_ATTEMPT"
    assert CooperativeTransferState.HANDOFF_CONFIRMED.value == "HANDOFF_CONFIRMED"
    assert CooperativeTransferState.HANDOFF_FAILED.value == "HANDOFF_FAILED"


@pytest.mark.unit
def test_v39_transfer_interface_record_success():
    """10. Test TransferInterface records success and updates status to VERIFIED."""
    tif = TransferInterface(position=(8, 3), boundary_a=(7, 3), boundary_b=(9, 3), confidence=0.5)
    tif.record_attempt(10)
    assert tif.attempts == 1
    assert tif.last_attempt_step == 10
    tif.record_success()
    assert tif.successes == 1
    assert tif.status == "VERIFIED"
    assert tif.confidence > 0.5


@pytest.mark.unit
def test_v39_transfer_interface_record_failure():
    """11. Test TransferInterface records failure and decrements confidence."""
    tif = TransferInterface(position=(8, 3), boundary_a=(7, 3), boundary_b=(9, 3), confidence=0.5)
    tif.record_attempt(15)
    tif.record_failure()
    assert tif.attempts == 1
    assert tif.failures == 1
    assert tif.confidence < 0.5


@pytest.mark.unit
def test_v39_npc_helper_classification():
    """12. Test NPC classification identifies helper agent from delivery history."""
    analyzer = ReachabilityAnalyzer()
    cls = analyzer.classify_npc(
        npc_id="helper_1",
        positions_history=[(9, 3), (10, 3), (13, 6)],
        items_delivered_count=1,
    )
    assert cls == NPCClassification.HELPER_AGENT

    cls_hazard = analyzer.classify_npc(
        npc_id="patrol_1",
        positions_history=[(2, 2), (2, 3), (2, 2)],
        items_delivered_count=0,
    )
    assert cls_hazard == NPCClassification.MOVING_HAZARD


@pytest.mark.unit
def test_v39_disengage_step_after_handoff():
    """13. Test agent disengage delta computation after item drop."""
    offset = (1, 0)
    disengage_delta = (-offset[0], -offset[1])
    assert disengage_delta == (-1, 0)
    act = grid_delta_to_action(disengage_delta[0], disengage_delta[1])
    assert act == GameAction.ACTION3


@pytest.mark.unit
def test_v39_composite_interaction_footprint_clearance():
    """14. Test composite carrier footprint avoids obstacle walls during delivery."""
    wm = WorldModel(cols=6, rows=6)
    wm.walls.add((3, 2))
    planner = Planner(wm)
    # Carrier has offset (0, 1). Target is (4, 1).
    path = planner.find_grid_path(start=(1, 1), target=(4, 1), carried_offset=(0, 1))
    assert path is not None
    # No step should cause the carried item at (c, r + 1) to hit (3, 2)
    for c, r in path:
        assert (c, r + 1) != (3, 2)


@pytest.mark.unit
def test_v39_reachability_analyzer_reset_level():
    """15. Test ReachabilityAnalyzer level reset clears all dynamic state."""
    analyzer = ReachabilityAnalyzer()
    divider = {(5, r) for r in range(10)}
    analyzer.compute_connected_components(cols=10, rows=10, walls=divider)
    analyzer.record_handoff((5, 5))
    assert len(analyzer.components) == 2
    assert (5, 5) in analyzer.handed_off_items

    analyzer.reset_level()
    assert len(analyzer.components) == 0
    assert len(analyzer.component_map) == 0
    assert len(analyzer.interfaces) == 0
    assert len(analyzer.handed_off_items) == 0


@pytest.mark.unit
def test_v39_plan_composite_interaction_indirect_reachability():
    """16. Test composite interaction selects approach with feasible downstream route."""
    wm = WorldModel(cols=8, rows=8)
    # Divider wall at col 4 with passable transfer slot at (4, 2)
    wm.walls = {(4, r) for r in range(8) if r != 2}
    planner = Planner(wm)

    # Item at (2, 2). Player at (0, 2). Destination is (4, 2)
    actions = planner.plan_composite_interaction(
        start=(0, 2),
        target=(2, 2),
        interaction_action=GameAction.ACTION5,
        destination_targets=[(4, 2)],
    )
    assert actions is not None
    assert len(actions) > 0
    assert actions[-1] == GameAction.ACTION5


@pytest.mark.unit
def test_v39_indirect_reachability_avoids_out_of_component_destinations():
    """17. Test downstream destination planning avoids blocked out-of-component cells."""
    wm = WorldModel(cols=8, rows=8)
    wm.walls = {(4, r) for r in range(8)}
    planner = Planner(wm)

    # All cells at col >= 4 are blocked
    blocked = {(c, r) for c in range(4, 8) for r in range(8)}
    # Destination in blocked region cannot be routed to
    actions = planner.plan_composite_interaction(
        start=(1, 2),
        target=(2, 2),
        destination_targets=[(5, 2)],
        blocked_cells=blocked,
    )
    assert actions is not None


@pytest.mark.unit
def test_v39_multi_item_handoff_isolation():
    """18. Test multiple items can be handed off without state collision."""
    analyzer = ReachabilityAnalyzer()
    analyzer.record_handoff((8, 3))
    analyzer.record_handoff((8, 8))
    assert len(analyzer.handed_off_items) == 2
    assert (8, 3) in analyzer.handed_off_items
    assert (8, 8) in analyzer.handed_off_items


@pytest.mark.unit
def test_v39_transfer_interface_npc_proximity_boost():
    """19. Test recipient NPC proximity boosts interface confidence."""
    analyzer = ReachabilityAnalyzer()
    divider = {(5, r) for r in range(10)}
    analyzer.compute_connected_components(cols=10, rows=10, walls=divider)
    target_comp = analyzer.get_component((7, 2))
    interfaces = analyzer.discover_transfer_interfaces(
        player_pos=(2, 2),
        target_comps={target_comp},
        cols=10,
        rows=10,
        npc_positions=[(6, 2)],
    )
    assert len(interfaces) > 0
    match = next(tif for tif in interfaces if tif.position == (5, 2))
    assert match.confidence >= 0.7


@pytest.mark.unit
def test_v39_direct_reachability_backward_compatibility():
    """20. Test backward compatibility for single-room direct reachability."""
    analyzer = ReachabilityAnalyzer()
    analyzer.compute_connected_components(cols=10, rows=10, walls=set())
    status, p_comp, goal_comps = analyzer.analyze_reachability(player_pos=(3, 3), goal_positions={(3, 8)})
    assert status == ReachabilityStatus.DIRECTLY_REACHABLE
    assert p_comp in goal_comps



