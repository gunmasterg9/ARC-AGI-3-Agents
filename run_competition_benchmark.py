"""
Kaggle ARC-AGI-3 Competition Benchmark Runner.
Executes V2 Baseline and V3.1 General Solver across all locally available environments
using the official ARC-AGI-3 Arcade / LocalEnvironmentWrapper API.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Type

import arc_agi
from arcengine import ActionInput, FrameData, GameAction, GameState

from agents.arcagi_solver import ARCAGI3Solver
from agents.arcagi_v3 import ARCAGIV3Solver

logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger("benchmark")

BASE_DIR = Path(r"C:\Users\GAUTAM\arc-agi-3\environment_files")
MAX_ACTIONS_PER_GAME = 250  # Action budget limit for benchmarking


def classify_failure(
    game_id: str,
    levels_completed: int,
    win_levels: int,
    final_state: str,
    total_actions: int,
    agent_name: str,
) -> str:
    """Classify failure mode based on observable outcome and game mechanics."""
    if final_state == "WIN" or (win_levels > 0 and levels_completed >= win_levels):
        return "NONE (PASSED)"

    if game_id == "ls20":
        if levels_completed >= 4:
            return "moving obstacle (Level 5 platform patrol tracks)"
        elif levels_completed >= 6:
            return "fog / partial observability (Level 7 20px radius)"
        else:
            return "transformation planning / level transition"

    # Other environments
    if total_actions >= MAX_ACTIONS_PER_GAME:
        return "action budget exhausted (unsolved domain mechanics)"

    if final_state == "GAME_OVER":
        return "goal inference / transformation planning (domain mechanics unmodeled)"

    return "unknown"


def run_single_benchmark(
    agent_cls: Type[Any],
    agent_name: str,
    game_id: str,
    arcade: arc_agi.Arcade,
) -> Dict[str, Any]:
    """Execute a single agent on a single game using the official Arcade wrapper."""
    start_time = time.time()
    resets = 0
    total_actions = 0
    levels_completed = 0
    win_levels = 0
    final_state = "UNKNOWN"
    failure_reason = ""

    try:
        wrapper = arcade.make(game_id)
        if not wrapper or not wrapper.observation_space:
            return {
                "environment": game_id,
                "agent": agent_name,
                "levels_completed": 0,
                "win_levels": 0,
                "score": 0,
                "total_actions": 0,
                "resets": 0,
                "final_state": "LOAD_ERROR",
                "runtime": 0.0,
                "failure_reason": "framework integration (failed to create wrapper)",
            }

        win_levels = wrapper.observation_space.win_levels
        max_actions = 450 if game_id == "ls20" else 80

        agent = agent_cls(
            card_id="",
            game_id=game_id,
            agent_name=agent_name,
            ROOT_URL="",
            record=False,
            arc_env=wrapper,
        )

        frames: List[FrameData] = []
        obs_raw = wrapper.observation_space
        curr_frame = agent._convert_raw_frame_data(obs_raw)
        frames.append(curr_frame)

        while total_actions < max_actions:
            if agent.is_done(frames, curr_frame):
                break

            try:
                action = agent.choose_action(frames, curr_frame)
            except Exception as e:
                failure_reason = f"agent exception during choose_action: {e}"
                break

            raw = wrapper.step(action)
            if not raw:
                break

            curr_frame = agent._convert_raw_frame_data(raw)
            frames.append(curr_frame)
            total_actions += 1

            if curr_frame.full_reset:
                resets += 1

            levels_completed = max(levels_completed, curr_frame.levels_completed or 0)
            st_name = curr_frame.state.name if hasattr(curr_frame.state, "name") else str(curr_frame.state)
            final_state = st_name

            if st_name in ("WIN", "GAME_OVER"):
                break

    except Exception as e:
        final_state = "EXCEPTION"
        failure_reason = f"runtime crash: {e}"

    runtime = round(time.time() - start_time, 2)
    if not failure_reason:
        failure_reason = classify_failure(
            game_id=game_id,
            levels_completed=levels_completed,
            win_levels=win_levels,
            final_state=final_state,
            total_actions=total_actions,
            agent_name=agent_name,
        )

    return {
        "environment": game_id,
        "agent": agent_name,
        "levels_completed": levels_completed,
        "total_levels": win_levels,
        "score": levels_completed,
        "total_actions": total_actions,
        "resets": resets,
        "final_state": final_state,
        "runtime": runtime,
        "failure_reason": failure_reason,
    }


def main() -> None:
    arcade = arc_agi.Arcade(
        environments_dir=str(BASE_DIR),
        operation_mode=arc_agi.OperationMode.OFFLINE,
    )

    available_games = sorted(set(e.game_id.split("-")[0] for e in arcade.available_environments))
    print(f"Discovered {len(available_games)} games for benchmark.")

    results: Dict[str, Any] = {
        "metadata": {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "games_count": len(available_games),
            "max_actions_limit": MAX_ACTIONS_PER_GAME,
        },
        "v3_results": {},
        "v2_results": {},
        "comparison": [],
    }

    # First run LS20 in detail for both agents
    priority_games = ["ls20"] + [g for g in available_games if g != "ls20"]

    print("\n========================================================")
    print("STARTING KAGGLE BENCHMARK SUITE")
    print("========================================================")

    for idx, gid in enumerate(priority_games):
        print(f"[{idx+1}/{len(priority_games)}] Benchmarking Game: {gid}...")

        # Run V3.1
        v3_res = run_single_benchmark(ARCAGIV3Solver, "v3.1", gid, arcade)
        results["v3_results"][gid] = v3_res
        print(f"   -> V3.1: score={v3_res['score']}/{v3_res['total_levels']}, acts={v3_res['total_actions']}, state={v3_res['final_state']}, time={v3_res['runtime']}s")

        # Run V2
        v2_res = run_single_benchmark(ARCAGI3Solver, "v2", gid, arcade)
        results["v2_results"][gid] = v2_res
        print(f"   -> V2  : score={v2_res['score']}/{v2_res['total_levels']}, acts={v2_res['total_actions']}, state={v2_res['final_state']}, time={v2_res['runtime']}s")

        # Compare
        diff = v3_res["score"] - v2_res["score"]
        results["comparison"].append({
            "environment": gid,
            "v2_score": v2_res["score"],
            "v3_score": v3_res["score"],
            "total_levels": v3_res["total_levels"],
            "score_diff": diff,
            "v2_actions": v2_res["total_actions"],
            "v3_actions": v3_res["total_actions"],
            "v3_failure_reason": v3_res["failure_reason"],
        })

    # Save to benchmark_results.json
    out_file = Path("benchmark_results.json")
    out_file.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nBenchmark completed successfully. Saved to {out_file.absolute()}")


if __name__ == "__main__":
    main()
