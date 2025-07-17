import numpy as np
from math import cos, sin, pi, acos
import problem1 as p1
import problem4 as p4

# 从问题1和问题4获取所需参数
N = p1.N
D_head = p1.D_head
D_body = p1.D_body
a = p1.a
invert_length = p1.invert_length
spiral_length = p1.spiral_length
R1_opt = p4.R1_opt
R2_opt = p4.R2_opt
Bx_opt = p4.Bx_opt
By_opt = p4.By_opt
R_turn = p4.R_turn
A = p4.A
C = p4.C

# 计算转向弧角
O1 = np.array([0.0, R_turn - R1_opt])
O2 = np.array([0.0, -R_turn + R2_opt])
theta1_opt = acos(np.clip(np.dot(A - O1, np.array([Bx_opt, By_opt]) - O1) / (np.linalg.norm(A - O1) * np.linalg.norm(np.array([Bx_opt, By_opt]) - O1)), -1, 1))
theta2_opt = acos(np.clip(np.dot(C - O2, np.array([Bx_opt, By_opt]) - O2) / (np.linalg.norm(C - O2) * np.linalg.norm(np.array([Bx_opt, By_opt]) - O2)), -1, 1))
theta_A = R_turn / a
theta_C = theta_A
L_arc1 = R1_opt * theta1_opt
L_arc2 = R2_opt * theta2_opt

# 假设已获得问题4中优化路径参数 R1_opt, R2_opt, 以及对应曲线ABC坐标
# 这里直接沿用问题1的链条运动模型，但需要组合螺旋线和圆弧路径

# 定义全局路径参数化长度
# （theta1_opt, theta2_opt可由优化解计算得到，即AO1B和CO2B角）
total_time = 200  # 模拟从调头前100s到调头后100s
dt = 1.0
v_head = 1.0  # 基准龙头速度1 m/s

# 准备存储各节速度最大值
max_speed = np.zeros(N)
max_speed_time = np.zeros(N)

# 准备初始状态：t=-100s时龙头尚未进入调头圈，在螺旋线上
# 设t=0对应龙头位于A点（调头开始），则t=-100时龙头在盘入螺线上距A弧长100m处
s_head_start = -100.0  # 以A处为0参考
# 记录前一时刻各节位置用于计算速度
prev_positions = None

for step in range(int(total_time/dt)+1):
    t = -100 + step*dt
    # 计算当前龙头沿路径的弧长位移 s_head (A点为0)
    if t < 0:
        s_head = t * v_head  # 负值，表示盘入螺线尚未到A
    elif t <= L_arc1/v_head:
        s_head = t * v_head  # 从0增加到L_arc1
    else:
        # 龙头经过两段圆弧后继续盘出螺线
        extra = t - L_arc1/v_head
        s_head = L_arc1 + L_arc2 + extra * v_head
    # 计算各节板凳沿全局路径的当前位置
    positions = []
    for i in range(1, N+1):
        # 计算该节前把手距离A点的路径弧长参数
        if i == 1:
            s_i = s_head
        else:
            offset = D_head + (i-2)*D_body if i>2 else D_head
            s_i = s_head - offset
        # 判断s_i所属路径段并计算坐标
        if s_i < 0:
            # 在盘入螺线 (A之前)
            theta = invert_length(spiral_length(theta_A) + s_i, theta_A)
            x = a*theta * np.cos(theta); y = a*theta * np.sin(theta)
        elif s_i <= L_arc1:
            # 在第一段圆弧
            theta = pi/2 - (s_i/R1_opt)
            O1 = np.array([0.0, R_turn - R1_opt])
            x = O1[0] + R1_opt * cos(theta); y = O1[1] + R1_opt * sin(theta)
        elif s_i <= L_arc1 + L_arc2:
            # 在第二段圆弧
            s2 = s_i - L_arc1
            phi = -pi/2 + (s2/R2_opt)
            O2 = np.array([0.0, -R_turn + R2_opt])
            x = O2[0] + R2_opt * cos(phi); y = O2[1] + R2_opt * sin(phi)
        else:
            # 在盘出螺线 (C之后)
            s3 = s_i - (L_arc1 + L_arc2)
            # 出螺线极坐标：设C点极角为θ_C，半径4.5
            theta = theta_C + invert_length(s3, 0)  # 从C点继续盘出（θ增加）
            x = a*theta * np.cos(theta); y = a*theta * np.sin(theta)
        positions.append((x,y))
    # 若有前一步结果则计算速度
    if prev_positions is not None:
        for i in range(N):
            dx = positions[i][0] - prev_positions[i][0]
            dy = positions[i][1] - prev_positions[i][1]
            vi = np.hypot(dx, dy) / dt
            if vi > max_speed[i]:
                max_speed[i] = vi
                max_speed_time[i] = t
    prev_positions = positions

# 计算ρ和α_max
rho = max_speed.max() / v_head
alpha_max = 2.0 / rho
v_head_max = alpha_max * v_head
i_max = np.argmax(max_speed)
print(f"最大速度发生在第{i_max+1}节板凳，v_max ≈ {max_speed.max():.3f} m/s 于 t≈{max_speed_time[i_max]:.1f}s")
print(f"ρ = {rho:.3f}, 龙头速度提升上限 α_max = {alpha_max:.4f}, 即 v_head,max ≈ {v_head_max:.6f} m/s")
