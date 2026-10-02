import numpy as np
import pytest
from arcengine import GameAction

from agents.arcagi_v3.action_semantics import (
    ActionObservation,
    ActionProbeEngine,
    ActionSemanticClassifier,
    ActionSemanticLabel,
    ActionSemanticProfile,
    ProbeSafety,
)
from agents.arcagi_v3.objects import GameObject, ObjectType
from agents.arcagi_v3.reachability import ReachabilityAnalyzer, ReachabilityStatus
from agents.arcagi_v3.state import HUDState, PlayerState, WorldState


def create_mock_world_state(
    col: int = 2,
    row: int = 2,
    carried_id: str | None = None,
    ctrl_id: str | None = None,
    objects: list[GameObject] | None = None,
    remaining_steps: int = 50,
    remaining_lives: int = 3,
    levels_completed: int = 0,
    screen: np.ndarray | None = None,
) -> WorldState:
    player = PlayerState(x=col * 4, y=row * 4, col=col, row=row)
    hud = HUDState(
        remaining_steps=remaining_steps,
        remaining_lives=remaining_lives,
        levels_completed=levels_completed,
    )
    if screen is None:
        screen = np.zeros((32, 32), dtype=np.int8)
    return WorldState(
        level_idx=levels_completed,
        player=player,
        hud=hud,
        detected_objects=objects or [],
        screen=screen,
        carried_object_id=carried_id,
        controlled_entity_id=ctrl_id,
    )


@pytest.mark.unit
def test_player_delta_detection():
    """1. Test that player movement between states is accurately computed."""
    engine = ActionProbeEngine()
    s_before = create_mock_world_state(col=2, row=2)
    s_after = create_mock_world_state(col=3, row=2)

    obs = engine.extract_observation(s_before, GameAction.ACTION4, s_after)
    assert obs.player_delta == (1, 0)
    assert obs.player_pixel_delta == (4, 0)
    scores = ActionSemanticClassifier.classify(obs)
    assert scores[ActionSemanticLabel.MOVEMENT] >= 0.90


@pytest.mark.unit
def test_zero_displacement_detection():
    """2. Test that zero displacement is identified when player does not move."""
    engine = ActionProbeEngine()
    s_before = create_mock_world_state(col=2, row=2)
    s_after = create_mock_world_state(col=2, row=2)

    obs = engine.extract_observation(s_before, GameAction.ACTION5, s_after)
    assert obs.player_delta == (0, 0)
    assert obs.player_pixel_delta == (0, 0)


@pytest.mark.unit
def test_time_advance_detection():
    """3. Test that time advance and step counter decrement are detected."""
    engine = ActionProbeEngine()
    s_before = create_mock_world_state(remaining_steps=50)
    s_after = create_mock_world_state(remaining_steps=49)

    obs = engine.extract_observation(s_before, GameAction.ACTION1, s_after)
    assert obs.time_delta == 1
    assert obs.step_counter_delta == -1


@pytest.mark.unit
def test_object_appearance_detection():
    """4. Test that newly appeared objects on grid are detected."""
    engine = ActionProbeEngine()
    s_before = create_mock_world_state(objects=[])
    obj_new = GameObject(object_type=ObjectType.ITEM, x=16, y=16, col=4, row=4)
    s_after = create_mock_world_state(objects=[obj_new])

    obs = engine.extract_observation(s_before, GameAction.ACTION5, s_after)
    assert len(obs.objects_appeared) == 1
    assert "4_4" in obs.objects_appeared[0]


@pytest.mark.unit
def test_object_disappearance_detection():
    """5. Test that disappeared objects from grid are identified."""
    engine = ActionProbeEngine()
    obj_old = GameObject(object_type=ObjectType.ITEM, x=16, y=16, col=4, row=4)
    s_before = create_mock_world_state(objects=[obj_old])
    s_after = create_mock_world_state(objects=[])

    obs = engine.extract_observation(s_before, GameAction.ACTION5, s_after)
    assert len(obs.objects_disappeared) == 1
    assert "4_4" in obs.objects_disappeared[0]


@pytest.mark.unit
def test_object_position_change_detection():
    """6. Test detection of displaced objects."""
    engine = ActionProbeEngine()
    obj_b = GameObject(object_type=ObjectType.ITEM, x=16, y=16, col=4, row=4)
    obj_a = GameObject(object_type=ObjectType.ITEM, x=20, y=16, col=5, row=4)
    s_before = create_mock_world_state(objects=[obj_b])
    s_after = create_mock_world_state(objects=[obj_a])

    obs = engine.extract_observation(s_before, GameAction.ACTION4, s_after)
    assert "obj_ITEM" in obs.object_displacements
    assert obs.object_displacements["obj_ITEM"] == (1, 0)


@pytest.mark.unit
def test_carried_state_change_detection():
    """7. Test detection of carried ownership changes."""
    engine = ActionProbeEngine()
    s_before = create_mock_world_state(carried_id=None)
    s_after = create_mock_world_state(carried_id="item_99")

    obs = engine.extract_observation(s_before, GameAction.ACTION5, s_after)
    assert obs.carried_changed is True
    assert obs.carried_before is None
    assert obs.carried_after == "item_99"


@pytest.mark.unit
def test_entity_creation_detection():
    """8. Test detection of newly created controllable entities."""
    engine = ActionProbeEngine()
    e1 = GameObject(object_type=ObjectType.CONTROLLABLE, x=4, y=4, col=1, row=1)
    e2 = GameObject(object_type=ObjectType.CONTROLLABLE, x=8, y=8, col=2, row=2)
    s_before = create_mock_world_state(objects=[e1])
    s_after = create_mock_world_state(objects=[e1, e2])

    obs = engine.extract_observation(s_before, GameAction.ACTION5, s_after)
    assert len(obs.entities_created) == 1


@pytest.mark.unit
def test_entity_removal_detection():
    """9. Test detection of entity removal."""
    engine = ActionProbeEngine()
    e1 = GameObject(object_type=ObjectType.CONTROLLABLE, x=4, y=4, col=1, row=1)
    s_before = create_mock_world_state(objects=[e1])
    s_after = create_mock_world_state(objects=[])

    obs = engine.extract_observation(s_before, GameAction.ACTION5, s_after)
    assert len(obs.entities_removed) == 1


@pytest.mark.unit
def test_environment_change_detection():
    """10. Test detection of widespread screen pixel changes."""
    engine = ActionProbeEngine()
    scr_b = np.zeros((32, 32), dtype=np.int8)
    scr_a = np.zeros((32, 32), dtype=np.int8)
    scr_a[0:5, 0:5] = 3  # 25 pixels changed

    s_before = create_mock_world_state(screen=scr_b)
    s_after = create_mock_world_state(screen=scr_a)

    obs = engine.extract_observation(s_before, GameAction.ACTION5, s_after)
    assert obs.environment_pixel_changes == 25
    assert obs.environment_mutated is True
    scores = ActionSemanticClassifier.classify(obs)
    assert scores[ActionSemanticLabel.ENVIRONMENT_CHANGE] >= 0.80


@pytest.mark.unit
def test_goal_change_detection():
    """11. Test detection of level advance and score delta."""
    engine = ActionProbeEngine()
    s_before = create_mock_world_state(levels_completed=1)
    s_after = create_mock_world_state(levels_completed=2)

    obs = engine.extract_observation(s_before, GameAction.ACTION1, s_after)
    assert obs.levels_completed_delta == 1


@pytest.mark.unit
def test_pickup_hypothesis():
    """12. Test that pickup observation triggers high PICKUP confidence."""
    engine = ActionProbeEngine()
    item = GameObject(object_type=ObjectType.ITEM, x=12, y=8, col=3, row=2)
    s_before = create_mock_world_state(col=2, row=2, carried_id=None, objects=[item])
    s_after = create_mock_world_state(col=2, row=2, carried_id="item_3_2", objects=[])

    obs = engine.extract_observation(s_before, GameAction.ACTION5, s_after)
    scores = ActionSemanticClassifier.classify(obs)
    assert scores[ActionSemanticLabel.PICKUP] >= 0.90
    assert scores.get(ActionSemanticLabel.DROP, 0.0) < 0.20


@pytest.mark.unit
def test_drop_hypothesis():
    """13. Test that drop observation triggers high DROP confidence."""
    engine = ActionProbeEngine()
    item = GameObject(object_type=ObjectType.ITEM, x=12, y=8, col=3, row=2)
    s_before = create_mock_world_state(col=2, row=2, carried_id="item_3_2", objects=[])
    s_after = create_mock_world_state(col=2, row=2, carried_id=None, objects=[item])

    obs = engine.extract_observation(s_before, GameAction.ACTION5, s_after)
    scores = ActionSemanticClassifier.classify(obs)
    assert scores[ActionSemanticLabel.DROP] >= 0.90
    assert scores.get(ActionSemanticLabel.PICKUP, 0.0) < 0.20


@pytest.mark.unit
def test_wait_hypothesis():
    """14. Test that zero displacement with stationary world triggers WAIT."""
    engine = ActionProbeEngine()
    s_before = create_mock_world_state(col=2, row=2)
    s_after = create_mock_world_state(col=2, row=2)

    obs = engine.extract_observation(s_before, GameAction.ACTION1, s_after)
    scores = ActionSemanticClassifier.classify(obs)
    assert scores[ActionSemanticLabel.WAIT] >= 0.85
    assert scores.get(ActionSemanticLabel.MOVEMENT, 0.0) < 0.20


@pytest.mark.unit
def test_focus_switch_hypothesis():
    """15. Test that switching active controllable entity triggers FOCUS_SWITCH."""
    engine = ActionProbeEngine()
    s_before = create_mock_world_state(ctrl_id="block_A")
    s_after = create_mock_world_state(ctrl_id="block_B")

    obs = engine.extract_observation(s_before, GameAction.ACTION5, s_after)
    assert obs.focus_switched is True
    scores = ActionSemanticClassifier.classify(obs)
    assert scores[ActionSemanticLabel.FOCUS_SWITCH] >= 0.90


@pytest.mark.unit
def test_entity_creation_hypothesis():
    """16. Test that adding a new actor triggers ENTITY_CREATION."""
    engine = ActionProbeEngine()
    e1 = GameObject(object_type=ObjectType.CONTROLLABLE, x=4, y=4, col=1, row=1)
    e2 = GameObject(object_type=ObjectType.CONTROLLABLE, x=12, y=12, col=3, row=3)
    s_before = create_mock_world_state(objects=[e1])
    s_after = create_mock_world_state(objects=[e1, e2])

    obs = engine.extract_observation(s_before, GameAction.ACTION5, s_after)
    scores = ActionSemanticClassifier.classify(obs)
    assert scores[ActionSemanticLabel.ENTITY_CREATION] >= 0.90


@pytest.mark.unit
def test_multi_label_semantics():
    """17. Test that an action can simultaneously support multiple hypotheses."""
    engine = ActionProbeEngine()
    # ACTION5 in WA30: used for PICKUP and DROP
    # First record PICKUP
    s1 = create_mock_world_state(carried_id=None)
    s2 = create_mock_world_state(carried_id="item_1")
    engine.record_transition(s1, GameAction.ACTION5, s2)

    # Then record DROP
    s3 = create_mock_world_state(carried_id="item_1")
    s4 = create_mock_world_state(carried_id=None)
    engine.record_transition(s3, GameAction.ACTION5, s4)

    profile = engine.get_profile(GameAction.ACTION5)
    # Both PICKUP and DROP have accumulated evidence
    assert profile.get_confidence(ActionSemanticLabel.PICKUP) > 0.30
    assert profile.get_confidence(ActionSemanticLabel.DROP) > 0.30


@pytest.mark.unit
def test_confidence_accumulation():
    """18. Test that repeated consistent observations increase confidence asymptotically."""
    engine = ActionProbeEngine()
    profile = engine.get_profile(GameAction.ACTION4)
    assert profile.get_confidence(ActionSemanticLabel.MOVEMENT) == 0.0

    # First observation
    s1 = create_mock_world_state(col=1, row=1)
    s2 = create_mock_world_state(col=2, row=1)
    engine.record_transition(s1, GameAction.ACTION4, s2)
    c1 = profile.get_confidence(ActionSemanticLabel.MOVEMENT)
    assert c1 > 0.30

    # Second observation
    s3 = create_mock_world_state(col=2, row=1)
    s4 = create_mock_world_state(col=3, row=1)
    engine.record_transition(s3, GameAction.ACTION4, s4)
    c2 = profile.get_confidence(ActionSemanticLabel.MOVEMENT)
    assert c2 > c1

    # Third observation
    s5 = create_mock_world_state(col=3, row=1)
    s6 = create_mock_world_state(col=4, row=1)
    engine.record_transition(s5, GameAction.ACTION4, s6)
    c3 = profile.get_confidence(ActionSemanticLabel.MOVEMENT)
    assert c3 > c2
    assert c3 >= 0.70


@pytest.mark.unit
def test_ambiguous_action_remains_unknown():
    """19. Test that an action with no clear deltas remains UNKNOWN without false positives."""
    engine = ActionProbeEngine()
    obs = ActionObservation(action=GameAction.ACTION5)
    scores = ActionSemanticClassifier.classify(obs)
    # Without any delta, it is recognized as wait or unknown, never pickup or movement
    assert scores.get(ActionSemanticLabel.MOVEMENT, 0.0) == 0.0
    assert scores.get(ActionSemanticLabel.PICKUP, 0.0) == 0.0
    assert scores.get(ActionSemanticLabel.DROP, 0.0) == 0.0


@pytest.mark.unit
def test_unsafe_probe_rejected():
    """20. Test that potentially destructive actions are blocked by safety engine."""
    engine = ActionProbeEngine()
    # Simulate an action that caused death
    s_before = create_mock_world_state(remaining_lives=2)
    s_after = create_mock_world_state(remaining_lives=1)  # life lost!
    engine.record_transition(s_before, GameAction.ACTION2, s_after)

    safety = engine.determine_probe_safety(GameAction.ACTION2, s_after)
    assert safety == ProbeSafety.DO_NOT_PROBE

    # Test step budget urgency
    critical_state = create_mock_world_state(remaining_steps=5)
    assert engine.determine_probe_safety(GameAction.ACTION1, critical_state) == ProbeSafety.DO_NOT_PROBE


@pytest.mark.unit
def test_follow_up_observation():
    """21. Test multi-frame observation window detects delayed transitions."""
    engine = ActionProbeEngine()
    s_before = create_mock_world_state()
    s_after = create_mock_world_state()

    # Follow up frame has animation change and entity creation
    e_clone = GameObject(object_type=ObjectType.CONTROLLABLE, x=8, y=8, col=2, row=2)
    scr_f = np.zeros((32, 32), dtype=np.int8)
    scr_f[2:8, 2:8] = 5
    s_follow = create_mock_world_state(objects=[e_clone], screen=scr_f)

    obs = engine.extract_observation(s_before, GameAction.ACTION5, s_after, follow_up_states=[s_follow])
    assert obs.follow_up_frames_count == 1
    assert obs.environment_pixel_changes > 0


@pytest.mark.unit
def test_game_independent_semantics():
    """22. Test that classifier functions identically across synthetic fixtures without game IDs."""
    c = ActionSemanticClassifier()
    # Pure data-driven observation
    obs = ActionObservation(
        action=GameAction.ACTION5,
        carried_changed=True,
        carried_before="item_A",
        carried_after=None,
    )
    scores = c.classify(obs)
    assert scores[ActionSemanticLabel.DROP] >= 0.90
    assert ActionSemanticLabel.DROP in scores


@pytest.mark.unit
def test_v39_regression():
    """23. Test that V3.9 reachability analyzer and core models remain completely intact."""
    analyzer = ReachabilityAnalyzer()
    divider = {(3, r) for r in range(6)}
    comps = analyzer.compute_connected_components(cols=6, rows=6, walls=divider)
    assert len(comps) == 2
    comp0 = analyzer.get_component((1, 2))
    comp1 = analyzer.get_component((4, 2))
    assert comp0 != comp1
    assert comp0 is not None
    assert comp1 is not None
