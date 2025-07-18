"""Detect bench collisions during spiral motion and output the final state."""

from __future__ import annotations

import numpy as np
import pandas as pd
from shapely.geometry import LineString

import problem1 as p1

# ---------------------------------------------------------------------------
# Parameters and pre-computed trajectory
# ---------------------------------------------------------------------------
# Extend simulation to 600 s so that potential collisions are captured
MAX_TIME = 600
if p1.T_total < MAX_TIME:
    p1.times = np.arange(0, MAX_TIME + p1.DT, p1.DT)
    p1.T_total = MAX_TIME
    OUTPUT, VELOCITY = p1.generate_data()
else:
    OUTPUT, VELOCITY = p1.output, p1.velocity

TIMES = p1.times
DT = p1.DT
N = p1.N

# Width of each bench (m)
BENCH_WIDTH = 0.30

# Bench lengths (head bench then body benches)
lengths = np.array([p1.D_head] + [p1.D_body] * (N - 1))
# Envelope radius used for quick rejection
env_radii = np.sqrt((lengths / 2) ** 2 + (BENCH_WIDTH / 2) ** 2)

# Extract coordinates and velocities
X = OUTPUT.iloc[0::2].to_numpy()
Y = OUTPUT.iloc[1::2].to_numpy()
V = VELOCITY.to_numpy()


def segment_distance(a1: tuple[float, float], a2: tuple[float, float],
                     b1: tuple[float, float], b2: tuple[float, float]) -> float:
    """Return minimal distance between two line segments."""
    seg1 = LineString([a1, a2])
    seg2 = LineString([b1, b2])
    return float(seg1.distance(seg2))


collision_time: float | None = None

# ---------------------------------------------------------------------------
# Main collision detection loop
# ---------------------------------------------------------------------------
for t_idx, t in enumerate(TIMES):
    for i in range(N):
        ax, ay = X[i, t_idx], Y[i, t_idx]
        bx, by = X[i + 1, t_idx], Y[i + 1, t_idx]
        mid_i = ((ax + bx) / 2, (ay + by) / 2)
        radius_i = env_radii[i]

        for j in range(i + 2, N):  # ignore adjacent benches
            cx, cy = X[j, t_idx], Y[j, t_idx]
            dx, dy = X[j + 1, t_idx], Y[j + 1, t_idx]
            mid_j = ((cx + dx) / 2, (cy + dy) / 2)

            # Envelope circle check
            if np.hypot(mid_i[0] - mid_j[0], mid_i[1] - mid_j[1]) > radius_i + env_radii[j]:
                continue

            # Precise distance check
            dist = segment_distance((ax, ay), (bx, by), (cx, cy), (dx, dy))
            if dist < BENCH_WIDTH:
                collision_time = t
                break
        if collision_time is not None:
            break
    if collision_time is not None:
        break

# ---------------------------------------------------------------------------
# Determine the safe stopping time t* and export the required data
# ---------------------------------------------------------------------------
if collision_time is None:
    t_star = TIMES[-1]
else:
    t_star = collision_time - DT

idx = np.where(TIMES == t_star)[0]
if idx.size == 0:
    raise ValueError("Requested time not available in data")
col_idx = int(idx[0])

rows = ["龙头", "第1节龙身", "第51节龙身", "第101节龙身",
        "第151节龙身", "第201节龙身", "龙尾（后）"]
positions = [0, 1, 51, 101, 151, 201, 223]

result_df = pd.DataFrame({
    "row": rows,
    "x (m)": X[positions, col_idx],
    "y (m)": Y[positions, col_idx],
    "v (m/s)": V[positions, col_idx],
})

result_df.to_excel("result2.xlsx", index=False)

if collision_time is None:
    print("链条在模拟时段内未发生碰撞。")
else:
    print(
        f"检测到碰撞，碰撞时刻 {collision_time:.1f} s，终止时刻 t* = {t_star:.1f} s"
    )
