"""模拟链条在螺旋线中的运动并生成位置数据。

该模块提供计算螺旋线弧长與極角轉換的函數，以及一個生成
位置數據的 ``generate_data`` 函數。其他腳本可以直接導入
這些函數和常量，而不會在導入時產生副作用。
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


def invert_length(s_target, theta_guess):
    """由螺线弧长反求极角θ，使弧长≈s_target（θ_guess为初值）"""
    # 用二分法求解θ
    low, high = 0.0, max(theta_guess, theta_guess + 10)
    # 确保high对应弧长超过目标
    while spiral_length(high) < s_target:
        low = high
        high *= 2
    for _ in range(50):
        mid = 0.5 * (low + high)
        if spiral_length(mid) < s_target:
            low = mid
        else:
            high = mid
    return 0.5 * (low + high)


def generate_data():
    """生成300秒内指定板凳節點的坐標資料表。"""
    times = np.arange(0, T_total + 1)
    cols = (
        ["龙头x (m)", "龙头y (m)"]
        + [f"第{i}节龙身x (m)" for i in [1, 51, 101, 151, 201]]
        + [f"第{i}节龙身y (m)" for i in [1, 51, 101, 151, 201]]
        + ["龙尾x (m)", "龙尾y (m)"]
    )
    output = pd.DataFrame(np.zeros((len(times), len(cols))), index=times, columns=cols)

    theta_head0 = 16 * 2 * np.pi
    s_head0 = spiral_length(theta_head0)

    for t in times:
        # 头部沿螺旋线向内收缩，弧长随时间递减
        s_head = max(0.0, s_head0 - v_head * t)

        theta_head = invert_length(s_head, theta_head0)
        x_head = a * theta_head * np.cos(theta_head)
        y_head = a * theta_head * np.sin(theta_head)
        output.at[t, "龙头x (m)"] = x_head
        output.at[t, "龙头y (m)"] = y_head

        for i in [2, 52, 102, 152, 202]:
            # 各节龙身或龙尾所处弧长，相对龙头后退相应间距
            s_i = max(0.0, s_head - (D_head + (i - 2) * D_body))
            theta_i = invert_length(s_i, theta_head if i < 50 else 0)
            xi = a * theta_i * np.cos(theta_i)
            yi = a * theta_i * np.sin(theta_i)
            colx = f"第{i - 1}节龙身x (m)" if i < 223 else "龙尾x (m)"
            coly = f"第{i - 1}节龙身y (m)" if i < 223 else "龙尾y (m)"
            output.at[t, colx] = xi
            output.at[t, coly] = yi

    return times, theta_head0, s_head0, output


times, theta_head0, s_head0, output = generate_data()


if __name__ == "__main__":
    # 打印几个时刻的龙头半径, 验证其随时间减小
    for t_sample in [0, 100, 200, 300]:
        r_head = np.hypot(output.at[t_sample, "龙头x (m)"], output.at[t_sample, "龙头y (m)"])
        print(f"t={t_sample}s, r_head={r_head:.3f} m")

    output.to_excel("result1.xlsx", index_label="时间(s)", float_format="%.6f")
