"""Detect spiral chain collision and output state at the stopping time."""

from __future__ import annotations

import numpy as np
import pandas as pd

import problem1 as p1

# Extend simulation time to capture potential collisions
MAX_TIME = 600
DT = 1.0
if p1.T_total < MAX_TIME:
    p1.times = np.arange(0, MAX_TIME + DT, DT)
    p1.T_total = MAX_TIME
    OUTPUT, VELOCITY = p1.generate_data()
else:
    OUTPUT, VELOCITY = p1.output, p1.velocity

TIMES = p1.times
N = p1.N
D_HEAD = p1.D_head
D_BODY = p1.D_body


def point_to_segment_dist(px: float, py: float, ax: float, ay: float, bx: float,
                          by: float) -> float:
    """Return distance from point (px, py) to segment AB."""
    if ax == bx and ay == by:
        return float(np.hypot(px - ax, py - ay))
    t = ((px - ax) * (bx - ax) + (py - ay) * (by - ay)) / ((bx - ax) ** 2 + (by - ay) ** 2)
    t = float(np.clip(t, 0.0, 1.0))
    projx = ax + t * (bx - ax)
    projy = ay + t * (by - ay)
    return float(np.hypot(px - projx, py - projy))


def segment_distance(a1: tuple[float, float], a2: tuple[float, float],
                     b1: tuple[float, float], b2: tuple[float, float]) -> float:
    """Return minimal distance between segments a1a2 and b1b2."""
    Ax, Ay = a1
    Bx, By = a2
    Cx, Cy = b1
    Dx, Dy = b2
    d1 = point_to_segment_dist(Ax, Ay, Cx, Cy, Dx, Dy)
    d2 = point_to_segment_dist(Bx, By, Cx, Cy, Dx, Dy)
    d3 = point_to_segment_dist(Cx, Cy, Ax, Ay, Bx, By)
    d4 = point_to_segment_dist(Dx, Dy, Ax, Ay, Bx, By)
    return min(d1, d2, d3, d4)


# Extract coordinates and velocities as arrays for faster access
X = OUTPUT.iloc[0::2].to_numpy()
Y = OUTPUT.iloc[1::2].to_numpy()
V = VELOCITY.to_numpy()

collision = False
min_clearance = float("inf")
t_star: float | None = None

for t_index, t in enumerate(TIMES):
    handles = list(zip(X[:, t_index], Y[:, t_index]))
    n_handles = len(handles)
    min_dist = float("inf")

    # check non-adjacent bench pairs
    for i in range(n_handles - 2):
        for j in range(i + 2, n_handles):
            xi, yi = handles[i]
            xj, yj = handles[j]
            if abs(xi - xj) > D_BODY * 2 or abs(yi - yj) > D_BODY * 2:
                continue
            dist = segment_distance(
                handles[i], handles[i + 1] if i + 1 < n_handles else handles[i],
                handles[j], handles[j + 1] if j + 1 < n_handles else handles[j],
            )
            if dist < min_dist:
                min_dist = dist

    if min_dist < min_clearance:
        min_clearance = min_dist

    if min_dist <= 0.30:
        collision = True
        t_star = float(t)
        break

result_time = (t_star - DT) if collision else TIMES[-1]
col = np.where(TIMES == result_time)[0]
if col.size == 0:
    raise ValueError("Requested time not available in data")
col_idx = int(col[0])

rows = ["龙头", "第1节龙身", "第51节龙身", "第101节龙身",
        "第151节龙身", "第201节龙身", "龙尾（后）"]
idx = [0, 1, 51, 101, 151, 201, 223]

out_df = pd.DataFrame({
    "row": rows,
    "x (m)": X[idx, col_idx],
    "y (m)": Y[idx, col_idx],
    "v (m/s)": V[idx, col_idx],
})

out_df.to_excel("result2.xlsx", index=False)

if collision:
    print(f"检测到碰撞，链条无法继续盘入。终止时刻 t* = {t_star:.1f} s")
else:
    print("链条在模拟时段内未发生碰撞。")
