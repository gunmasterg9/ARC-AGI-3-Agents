import sys
sys.path.append(r'C:\Users\GAUTAM\arc-agi-3\environment_files\ls20\9607627b')
import ls20
from arcengine import ActionInput
from agents.arcagi_solver import ArcagiSolver

print("Creating ArcagiSolver...")
solver = ArcagiSolver()

print("Creating local Ls20 game...")
game = ls20.Ls20()

frames = []
# Initial frame
first_frame = game.perform_action(ActionInput(id=ls20.GameAction.RESET))
frames.append(first_frame)

action_count = 0
max_actions = 600

while not solver.is_done(frames, frames[-1]) and action_count < max_actions:
    action = solver.choose_action(frames, frames[-1])
    frame = game.perform_action(ActionInput(id=action))
    frames.append(frame)
    action_count += 1
    
    if action_count % 50 == 0 or solver.is_done(frames, frame):
        print(f"Action {action_count}: score={game._score}, state={game._state.name}, action={action.name}")
        
    if game._state.name == "GAME_OVER":
        print(f"FAILED: Game Over at action {action_count}")
        break

print(f"\n==========================================")
print(f"FINAL RESULT: Total Actions = {action_count}")
print(f"Score / Levels completed = {game._score}")
print(f"Game State = {game._state.name}")
print(f"Solver is_done = {solver.is_done(frames, frames[-1])}")
print(f"==========================================")

assert game._score == 7, f"Expected 7 levels completed, got {game._score}"
assert solver.is_done(frames, frames[-1]), "Expected solver.is_done() to be True"
print("ALL TESTS PASSED WITH 100% SUCCESS!")
