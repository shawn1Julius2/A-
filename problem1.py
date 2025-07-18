"""Compute dragon dance chain positions and speeds along an Archimedean spiral.

This module implements the rigid chain model described in the paper. It
simulates the dragon with 223 benches moving on a spiral at a constant
head speed and outputs the position and velocity tables required for
``result1.xlsx``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Basic parameters of the dragon chain and spiral
# ---------------------------------------------------------------------------
N = 223  # total number of benches (1 head, 221 body, 1 tail)

# distance between the head bench front handle and the next bench
D_head = 2.860  # metres

# distance between subsequent body benches
D_body = 1.650  # metres

# Archimedean spiral pitch (m)
p = 0.55

# spiral coefficient a = p / (2*pi)
a = p / (2 * np.pi)

# head bench translation speed along the spiral (m/s)
v_head = 1.0

# total simulation duration (s)
T_total = 300

# time sequence used throughout the simulation
times = np.arange(T_total + 1)

# expose upper-case aliases for backward compatibility
D_HEAD = D_head
D_BODY = D_body
PITCH = p
A = a
V_HEAD = v_head
T_TOTAL = T_total


def spiral_length(theta: float) -> float:
    """Return arc length of ``r = a * theta`` from 0 to ``theta``."""
    return 0.5 * A * (theta * np.sqrt(theta ** 2 + 1) + np.arcsinh(theta))


def invert_length(s_target: float, theta_guess: float) -> float:
    """Invert :func:`spiral_length` using Newton iteration."""
    theta = theta_guess
    for _ in range(20):
        f = spiral_length(theta) - s_target
        if abs(f) < 1e-10:
            break
        theta -= f / (A * np.sqrt(theta ** 2 + 1))
    return theta


# initial angle of the head (16 full turns)
theta_head0 = 16 * 2 * np.pi

# corresponding arc length position of the head at t=0
s_head0 = spiral_length(theta_head0)


def generate_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate position and velocity tables for ``times`` seconds."""
    cols = [f"{t} s" for t in times]

    pos_index = ["龙头x (m)", "龙头y (m)"]
    vel_index = ["龙头 (m/s)"]
    for i in range(1, N - 1):
        pos_index.extend([f"第{i}节龙身x (m)", f"第{i}节龙身y (m)"])
        vel_index.append(f"第{i}节龙身  (m/s)")
    pos_index.extend(["龙尾x (m)", "龙尾y (m)", "龙尾（后）x (m)", "龙尾（后）y (m)"])
    vel_index.extend(["龙尾  (m/s)", "龙尾（后） (m/s)"])

    pos_df = pd.DataFrame(index=pos_index, columns=cols, dtype=float)
    vel_df = pd.DataFrame(index=vel_index, columns=cols, dtype=float)

    # Theta values for head, body benches and tail rear handle
    thetas = np.zeros(N + 1)

    # initialise all segments at t=0
    thetas[0] = theta_head0
    for seg in range(1, N):
        s_i = s_head0 - (D_HEAD + (seg - 1) * D_BODY)
        thetas[seg] = invert_length(s_i, thetas[seg - 1])
    s_tail_rear = s_head0 - (D_HEAD + (N - 1) * D_BODY)
    thetas[N] = invert_length(s_tail_rear, thetas[N - 1])

    x = np.zeros((N + 1, len(times)))
    y = np.zeros((N + 1, len(times)))

    for t_idx, t in enumerate(times):
        s_head = s_head0 - v_head * t
        thetas[0] = invert_length(s_head, thetas[0])
        x[0, t_idx] = A * thetas[0] * np.cos(thetas[0])
        y[0, t_idx] = A * thetas[0] * np.sin(thetas[0])

        for seg in range(1, N):
            s_i = s_head - (D_HEAD + (seg - 1) * D_BODY)
            thetas[seg] = invert_length(s_i, thetas[seg])
            x[seg, t_idx] = A * thetas[seg] * np.cos(thetas[seg])
            y[seg, t_idx] = A * thetas[seg] * np.sin(thetas[seg])

        s_tail_rear = s_head - (D_HEAD + (N - 1) * D_BODY)
        thetas[N] = invert_length(s_tail_rear, thetas[N])
        x[N, t_idx] = A * thetas[N] * np.cos(thetas[N])
        y[N, t_idx] = A * thetas[N] * np.sin(thetas[N])

    # Fill position DataFrame
    for seg in range(len(pos_index) // 2):
        pos_df.loc[pos_index[2 * seg], :] = x[seg, :]
        pos_df.loc[pos_index[2 * seg + 1], :] = y[seg, :]

    # Compute velocities
    vel = np.zeros((N + 1, len(times)))
    vel[:, 1:] = np.sqrt(np.diff(x, axis=1) ** 2 + np.diff(y, axis=1) ** 2)

    for idx, name in enumerate(vel_index):
        vel_df.loc[name, :] = vel[idx, :]

    return pos_df, vel_df


# Pre-compute tables so that other modules can reuse them directly
output, velocity = generate_data()


if __name__ == "__main__":
    with pd.ExcelWriter("result1.xlsx") as writer:
        output.to_excel(writer, sheet_name="位置", float_format="%.6f")
        velocity.to_excel(writer, sheet_name="速度", float_format="%.6f")
