"""模拟链条在螺旋线中的运动并生成位置与速度数据。

本模块实现问题一中的刚性链螺旋盘入模型。它提供
``spiral_length`` 和 ``invert_length`` 两个函数，用于弧长与
极角的互相转换，并在 :func:`generate_data` 中按照文档中的
建模思路生成 0--300 秒内关键板凳的位置和速度表。
其他脚本可直接导入这些函数及生成的结果。
"""

import numpy as np
import pandas as pd

# 常量定义
N = 223  # 板凳节数（含龙头、龙尾）
D_head = 2.860  # 龙头与第1节龙身距离(m)
D_body = 1.650  # 龙身各节间距(m)
p = 0.55  # 螺线螺距(m)
a = p / (2 * np.pi)
v_head = 1.0  # 龙头速度(m/s)
T_total = 300  # 模拟总时长(s)


# 螺线弧长与极角关系：数值积分近似
def spiral_length(theta):
    # 积分0到theta的弧长: ∫0^θ sqrt(r^2 + (dr/dθ)^2) dθ, r=a*θ
    # = ∫0^θ sqrt(a^2 θ^2 + a^2) dθ = 0.5*a[θ*sqrt(θ^2+1) + asinh(θ)]
    return 0.5 * a * (theta * np.sqrt(theta**2 + 1) + np.arcsinh(theta))


def invert_length(s_target: float, theta_guess: float) -> float:
    """根据给定弧长反求螺线极角。

    参数 ``theta_guess`` 用作牛顿迭代初值。该值应当接近目标解，
    在仿真中可直接使用上一时刻的极角。为了稳健起见，如迭代
    出现发散将回退到二分搜索。"""

    theta = theta_guess
    for _ in range(20):
        s = spiral_length(theta)
        f = s - s_target
        if abs(f) < 1e-10:
            return theta
        theta -= f / (a * np.sqrt(theta**2 + 1))

    # 若牛顿迭代未收敛，改用二分法
    low, high = (min(theta, theta_guess), max(theta, theta_guess))
    for _ in range(40):
        mid = 0.5 * (low + high)
        if spiral_length(mid) < s_target:
            low = mid
        else:
            high = mid
    return 0.5 * (low + high)


def generate_data():
    """计算链条盘入过程中的位置和速度数据。"""

    times = np.arange(0, T_total + 1)
    pos_cols = (
        ["龙头x (m)", "龙头y (m)"]
        + [f"第{i}节龙身x (m)" for i in [1, 51, 101, 151, 201]]
        + [f"第{i}节龙身y (m)" for i in [1, 51, 101, 151, 201]]
        + ["龙尾x (m)", "龙尾y (m)"]
    )
    vel_cols = (
        ["龙头速度 (m/s)"]
        + [f"第{i}节龙身速度 (m/s)" for i in [1, 51, 101, 151, 201]]
        + ["龙尾速度 (m/s)"]
    )

    pos_df = pd.DataFrame(index=times, columns=pos_cols, dtype=float)
    vel_df = pd.DataFrame(index=times, columns=vel_cols, dtype=float)

    theta_head0 = 16 * 2 * np.pi
    s_head0 = spiral_length(theta_head0)

    theta_head = theta_head0
    s_head = s_head0
    prev_coords = None

    for t in times:
        theta_head = invert_length(s_head, theta_head)
        head_x = a * theta_head * np.cos(theta_head)
        head_y = a * theta_head * np.sin(theta_head)
        pos_df.at[t, "龙头x (m)"] = head_x
        pos_df.at[t, "龙头y (m)"] = head_y
        coords = [(head_x, head_y)]

        for seg in [2, 52, 102, 152, 202]:
            s_i = s_head - (D_head + (seg - 2) * D_body)
            if s_i <= 0:
                theta_i = 0.0
            else:
                theta_i = invert_length(s_i, theta_head)
            xi = a * theta_i * np.cos(theta_i)
            yi = a * theta_i * np.sin(theta_i)
            idx = seg - 1
            pos_df.at[t, f"第{idx}节龙身x (m)"] = xi
            pos_df.at[t, f"第{idx}节龙身y (m)"] = yi
            coords.append((xi, yi))

        # 龙尾后把手（比第223节前孔再落后一个 D_body）
        s_tail = s_head - (D_head + 221 * D_body)
        if s_tail <= 0:
            theta_tail = 0.0
        else:
            theta_tail = invert_length(s_tail, theta_head)
        tail_x = a * theta_tail * np.cos(theta_tail)
        tail_y = a * theta_tail * np.sin(theta_tail)
        pos_df.at[t, "龙尾x (m)"] = tail_x
        pos_df.at[t, "龙尾y (m)"] = tail_y
        coords.append((tail_x, tail_y))

        if prev_coords is not None:
            step_vel = [
                np.hypot(c[0] - pc[0], c[1] - pc[1])
                for c, pc in zip(coords, prev_coords)
            ]
            vel_df.loc[t] = step_vel
        prev_coords = coords

        # 下一时刻头节沿螺线向内推进 1 m
        s_head -= v_head

    return times, theta_head0, s_head0, pos_df, vel_df


times, theta_head0, s_head0, output, velocity = generate_data()


if __name__ == "__main__":
    with pd.ExcelWriter("result1.xlsx") as writer:
        output.to_excel(writer, sheet_name="位置", index_label="时间(s)", float_format="%.6f")
        velocity.to_excel(writer, sheet_name="速度", index_label="时间(s)", float_format="%.6f")
