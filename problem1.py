import numpy as np
import pandas as pd

# 常量定义
N = 223  # 板凳节数（含龙头、龙尾）
D_head = 2.860  # 龙头与第1节龙身距离(m)
D_body = 1.650  # 龙身各节间距(m)
p = 0.55       # 螺线螺距(m)
a = p/(2*np.pi)
v_head = 1.0   # 龙头速度(m/s)
T_total = 300  # 模拟总时长(s)

# 螺线弧长与极角关系：数值积分近似
def spiral_length(theta):
    # 积分0到theta的弧长: ∫0^θ sqrt(r^2 + (dr/dθ)^2) dθ, r=a*θ
    # = ∫0^θ sqrt(a^2 θ^2 + a^2) dθ = 0.5*a[θ*sqrt(θ^2+1) + asinh(θ)]
    return 0.5*a*(theta*np.sqrt(theta**2+1) + np.arcsinh(theta))

def invert_length(s_target, theta_guess):
    """由螺线弧长反求极角θ，使弧长≈s_target（θ_guess为初值）"""
    # 用二分法求解θ
    low, high = 0.0, max(theta_guess, theta_guess+10)
    # 确保high对应弧长超过目标
    while spiral_length(high) < s_target:
        low = high
        high *= 2
    for _ in range(50):
        mid = 0.5*(low+high)
        if spiral_length(mid) < s_target:
            low = mid
        else:
            high = mid
    return 0.5*(low+high)

# 初始化输出数据结构
times = np.arange(0, T_total+1)
cols = ["龙头x (m)", "龙头y (m)"] + [f"第{i}节龙身x (m)" for i in [1,51,101,151,201]] \
       + [f"第{i}节龙身y (m)" for i in [1,51,101,151,201]] + ["龙尾x (m)", "龙尾y (m)"]
output = pd.DataFrame(np.zeros((len(times), len(cols))), index=times, columns=cols)

# 初始龙头弧长位置（t=0处）: 设龙头初始在θ0处
theta_head0 = 16*2*np.pi  # 初始极角
s_head0 = spiral_length(theta_head0)  # 对应弧长
# 仿真逐秒推进
for t in times:
    # 龙头当前弧长位置
    s_head = s_head0 + v_head * t  # 以1 m/s前进
    # 计算各指定节的坐标
    # 龙头:
    theta_head = invert_length(s_head, theta_head0)
    x_head = a*theta_head*np.cos(theta_head)
    y_head = a*theta_head*np.sin(theta_head)
    output.at[t, "龙头x (m)"] = x_head
    output.at[t, "龙头y (m)"] = y_head
    # 第1、51、101、151、201节龙身前把手:
    for i in [2, 52, 102, 152, 202]:  # 对应上述节号+1（因为龙头为1节）
        # 计算该节前把手应在的弧长参数
        s_i = s_head - (D_head + (i-2)*D_body)
        if s_i < 0: 
            s_i = 0.0  # 若尚未盘入螺线（链条尾部在外围直线），则置0
        theta_i = invert_length(s_i, theta_head if i<50 else 0)
        xi = a*theta_i*np.cos(theta_i)
        yi = a*theta_i*np.sin(theta_i)
        # 对应输出列（第1节龙身为i=2）
        colx = f"第{i-1}节龙身x (m)" if i<223 else "龙尾x (m)"
        coly = f"第{i-1}节龙身y (m)" if i<223 else "龙尾y (m)"
        output.at[t, colx] = xi
        output.at[t, coly] = yi
    # （可选）速度计算: 用相邻时刻位置差近似
    # 此处省略，稍后问题5将集中计算速度
# 保存result1.xlsx结果
output.to_excel("result1.xlsx", index_label="时间(s)", float_format="%.6f")
