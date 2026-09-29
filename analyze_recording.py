import json
from collections import Counter

import numpy as np


PATH = (
    r"recordings\ls20-9607627b.arcagi3solver."
    r"af963ffd-6d86-41aa-a715-9b62c514a8d0.recording.jsonl"
)


def main():
    records = [
        json.loads(line)
        for line in open(PATH, encoding="utf-8")
        if line.strip()
    ]

    print(f"Records: {len(records)}")

    for i, record in enumerate(records[:5]):
        data = record.get("data", {})

        print(f"\n=== RECORD {i} ===")
        print("Keys:", list(data.keys()))
        print("Action:", data.get("action_input"))
        print("State:", data.get("state"))
        print("Levels:", data.get("levels_completed"))

        frame = data.get("frame")

        if frame:
            print("Number of grids:", len(frame))

            for gi, grid in enumerate(frame):
                arr = np.asarray(grid)

                print(
                    f"Grid {gi}: "
                    f"shape={arr.shape}, "
                    f"min={arr.min()}, "
                    f"max={arr.max()}, "
                    f"unique={sorted(np.unique(arr).tolist())}"
                )

    print("\n=== COLOR FREQUENCIES ===")

    colors = Counter()

    for record in records:
        frame = record.get("data", {}).get("frame")

        if frame:
            for grid in frame:
                colors.update(np.asarray(grid).flatten().tolist())

    for color, count in sorted(colors.items()):
        print(f"{color:2}: {count}")


if __name__ == "__main__":
    main()