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
    assert solver.current_level_idx == 0
    assert len(solver.active_plan) > 0


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
