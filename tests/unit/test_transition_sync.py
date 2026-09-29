import numpy as np
import pytest
from arcengine import FrameData, GameAction

from agents.arcagi_v3 import ARCAGIV3Solver
from agents.arcagi_v3.state import TransitionPhase


def create_mock_level_screen(wall_pixel_val: int = 3, num_walls: int = 20, player_pos: tuple[int, int] = (6, 2)) -> np.ndarray:
    """Create a synthetic 64x64 screen with distinct wall and player patterns."""
    screen = np.zeros((64, 64), dtype=np.int32)
    # Add border walls
    screen[0, :] = wall_pixel_val
    screen[63, :] = wall_pixel_val
    screen[:, 0] = wall_pixel_val
    screen[:, 63] = wall_pixel_val

    # Add specific wall pattern
    for i in range(num_walls):
        screen[10 + i, 10 + i] = wall_pixel_val

    # Place player at specific grid position
    # Grid coordinates convert to pixels: x = 4 + col * 5, y = row * 5
    px = 4 + player_pos[0] * 5
    py = player_pos[1] * 5
    screen[py:py+2, px:px+5] = 12  # Top color (COLORS[0])
    screen[py+2:py+5, px:px+5] = 9  # Bottom color (COLORS[1])
    return screen


def test_level_transition_synchronization_regression() -> None:
    """
    PHASE 6 REGRESSION TEST:
    Reproduce the exact transition timing mismatch:
    - Frame N: Level 1 completion, levels_completed increments to 1, but image still represents Level 1.
    - Frame N+1: Actual Level 2 image arrives with new player position and new wall geometry.
    Verify:
    1. Frame N does NOT initialize Level 2 model (transition state held).
    2. Frame N+1 initializes Level 2 model.
    3. Level 1 objects do not appear in Level 2 model (isolated world model).
    4. Level 2 player position is correct.
    5. Level 2 goal is detected.
    6. Level 2 planner finds the route.
    """
    solver = ARCAGIV3Solver()

    # Create distinct visual frames for Level 1 and Level 2
    l1_screen = create_mock_level_screen(wall_pixel_val=3, num_walls=10, player_pos=(6, 2))
    l2_screen = create_mock_level_screen(wall_pixel_val=3, num_walls=25, player_pos=(5, 8))

    # Add an obstacle unique to Level 1 at (6, 4)
    # In pixels: px = 4 + 6*5 = 34, py = 4*5 = 20
    l1_screen[20:25, 34:39] = 3

    # Simulate Level 1 active state
    solver.current_level_idx = 0
    solver.last_completed_screen = l1_screen.copy()
    solver.last_completed_goal_pos = (6, 2)

    # Frame N: Level 1 completion frame where levels_completed has incremented to 1,
    # but the image still shows the old Level 1 screen (stale completion frame).
    frame_n = FrameData(
        game_id="ls20",
        frame=[l1_screen.tolist()],
        levels_completed=1,
    )

    action_n = solver.choose_action([frame_n], frame_n)

    # Verification 1: Frame N MUST NOT initialize Level 2 WorldModel
    assert solver.current_level_idx == 0, "Level index should remain 0 during stale Frame N"
    assert solver.transition_phase == TransitionPhase.WAITING_FOR_NEW_LEVEL_FRAME, (
        "Transition phase should be WAITING_FOR_NEW_LEVEL_FRAME"
    )
    # Level 1 obstacle (6, 4) must NOT have been recorded into Level 2
    assert (6, 4) not in solver.world_model.walls, "Level 1 obstacles must not pollute model on stale frame"

    # Frame N+1: The actual Level 2 frame arrives
    frame_n1 = FrameData(
        game_id="ls20",
        frame=[l2_screen.tolist()],
        levels_completed=1,
    )

    action_n1 = solver.choose_action([frame_n, frame_n1], frame_n1)

    # Verification 2: Frame N+1 MUST initialize Level 2
    assert solver.current_level_idx == 1, "Level index should advance to 1 on verified Frame N+1"
    assert solver.transition_phase == TransitionPhase.NORMAL, "Transition phase should return to NORMAL"

    # Verification 3: Level 1 objects do not appear in Level 2 model
    assert (6, 4) not in solver.world_model.walls, "Level 1 obstacle (6, 4) must not appear in Level 2 model"

    # Verification 4: Level 2 player position is correct
    assert solver.last_world_state is not None, "World state should be recorded"
    assert solver.last_world_state.player is not None, "Player should be detected"
    assert solver.last_world_state.player.grid_pos == (5, 8), (
        f"Expected Level 2 player at (5, 8), got {solver.last_world_state.player.grid_pos}"
    )

    # Verification 5: Level 2 goal is detected
    current_goal = solver.goal_manager.get_current_goal()
    assert current_goal is not None, "Level 2 goal should be loaded"
    assert (current_goal.col, current_goal.row) == (2, 8), (
        f"Expected Level 2 goal at (2, 8), got {(current_goal.col, current_goal.row)}"
    )

    # Verification 6: Level 2 planner finds the route
    subgoal = solver.active_subgoal
    assert subgoal is not None, "Level 2 active subgoal should be set"
    path = solver.planner.find_grid_path((5, 8), subgoal.target_pos)
    assert path is not None, f"Planner must find a route from (5, 8) to {subgoal.target_pos}"
    assert len(path) > 1, "Path should contain steps"
