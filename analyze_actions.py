import json
import numpy as np

p = r"recordings\ls20-9607627b.arcagi3solver.af963ffd-6d86-41aa-a715-9b62c514a8d0.recording.jsonl"

rows = [
    json.loads(x)
    for x in open(p, encoding="utf-8")
]

frames = []

for r in rows:
    data = r.get("data", {})
    frame = data.get("frame")

    if isinstance(frame, list) and frame:
        grid = np.array(frame[0])

        if grid.shape == (64, 64):
            frames.append(grid)


# This is the action sequence produced by V1.
action_names = {
    1: "ACTION1",
    2: "ACTION2",
    3: "ACTION3",
    4: "ACTION4",
}

actions = [
    ((i % 4) + 1)
    for i in range(len(frames))
]


def bbox(grid, color):
    ys, xs = np.where(grid == color)

    if len(xs) == 0:
        return None

    return (
        int(xs.min()),
        int(ys.min()),
        int(xs.max()),
        int(ys.max()),
        int(len(xs))
    )


def centroid(grid, color):
    ys, xs = np.where(grid == color)

    if len(xs) == 0:
        return None

    return (
        round(float(xs.mean()), 2),
        round(float(ys.mean()), 2)
    )


print("Frames:", len(frames))

print()
print("=" * 100)
print("ACTION -> FRAME TRANSITIONS")
print("=" * 100)

for i in range(1, len(frames)):

    previous = frames[i - 1]
    current = frames[i]

    action = actions[i - 1]

    diff = previous != current
    changed = int(diff.sum())

    print(
        f"{i-1:02d} -> {i:02d} | "
        f"{action_names[action]:8s} | "
        f"changed={changed:4d} | "
        f"C11={centroid(current, 11)} | "
        f"C12={centroid(current, 12)}"
    )


print()
print("=" * 100)
print("ACTION STATISTICS")
print("=" * 100)

for action in range(1, 5):

    changes = []
    c11_dx = []
    c11_dy = []
    c12_dx = []
    c12_dy = []

    for i in range(1, len(frames)):

        previous_action = actions[i - 1]

        if previous_action != action:
            continue

        diff = int((frames[i - 1] != frames[i]).sum())
        changes.append(diff)

        a = centroid(frames[i - 1], 11)
        b = centroid(frames[i], 11)

        if a and b:
            c11_dx.append(round(b[0] - a[0], 2))
            c11_dy.append(round(b[1] - a[1], 2))

        a = centroid(frames[i - 1], 12)
        b = centroid(frames[i], 12)

        if a and b:
            c12_dx.append(round(b[0] - a[0], 2))
            c12_dy.append(round(b[1] - a[1], 2))

    print()
    print(action_names[action])

    print("  frame changes:", changes)

    print("  color 11 dx:", c11_dx)
    print("  color 11 dy:", c11_dy)

    print("  color 12 dx:", c12_dx)
    print("  color 12 dy:", c12_dy)


print()
print("=" * 100)
print("FULL BOARD TRANSITIONS")
print("=" * 100)

for i in range(1, len(frames)):

    changed = int((frames[i - 1] != frames[i]).sum())

    if changed > 3000:

        print(
            f"Transition at {i-1} -> {i} | "
            f"action={action_names[actions[i-1]]} | "
            f"changed={changed}"
        )

        print("  previous C11:", bbox(frames[i - 1], 11))
        print("  current  C11:", bbox(frames[i], 11))

        print("  previous C12:", bbox(frames[i - 1], 12))
        print("  current  C12:", bbox(frames[i], 12))