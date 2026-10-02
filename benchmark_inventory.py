import json
import logging
from pathlib import Path
from datetime import datetime, timezone
import arc_agi

logging.basicConfig(level=logging.ERROR)

base_dir = Path(r"C:\Users\GAUTAM\arc-agi-3\environment_files")
rec_dir = Path(r"C:\Users\GAUTAM\arc-agi-3\ARC-AGI-3-Agents\recordings")

arcade = arc_agi.Arcade(
    environments_dir=str(base_dir),
    operation_mode=arc_agi.OperationMode.OFFLINE,
)

results = []
for env_info in sorted(arcade.available_environments, key=lambda e: e.game_id):
    gid = env_info.game_id.split("-")[0]
    meta_path = Path(env_info.local_dir) / "metadata.json" if env_info.local_dir else None
    meta = {}
    if meta_path and meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            pass

    rec_matches = list(rec_dir.glob(f"*{gid}*")) if rec_dir.exists() else []

    status = "OK"
    win_levels = None
    err = None
    try:
        wrapper = arcade.make(gid)
        if wrapper and wrapper.observation_space:
            win_levels = wrapper.observation_space.win_levels
        else:
            status = "FAILED_TO_MAKE"
    except Exception as e:
        status = "ERROR"
        err = str(e)

    results.append({
        "game_id": gid,
        "full_id": env_info.game_id,
        "title": meta.get("title", env_info.title),
        "version": env_info.game_id.split("-")[-1] if "-" in env_info.game_id else "",
        "class_name": meta.get("class_name", env_info.class_name),
        "win_levels": win_levels,
        "tags": meta.get("tags", env_info.tags),
        "local_dir": env_info.local_dir,
        "status": status,
        "error": err,
        "has_recording": len(rec_matches) > 0,
        "recordings": [r.name for r in rec_matches],
    })

print(f"Total environments evaluated: {len(results)}")
ok_count = sum(1 for r in results if r["status"] == "OK")
print(f"Successfully loaded: {ok_count}")
for r in results:
    rec_str = f"rec={len(r['recordings'])}" if r["has_recording"] else "no-rec"
    print(f"  {r['game_id']:4s} | {r['full_id']:15s} | levels={r['win_levels']} | status={r['status']:4s} | tags={r['tags']} | {rec_str}")

with open("environment_inventory.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)
print("Saved inventory to environment_inventory.json")
