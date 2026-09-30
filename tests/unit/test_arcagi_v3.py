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

