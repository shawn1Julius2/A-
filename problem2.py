# 延续上题模拟，加入碰撞检测
import numpy as np
import problem1 as p1

# 引入问题1中的参数和结果
times = p1.times
invert_length = p1.invert_length
N = p1.N
D_head = p1.D_head
D_body = p1.D_body
s_head0 = p1.s_head0
theta_head0 = p1.theta_head0
v_head = p1.v_head
a = p1.a
output = p1.output
T_total = p1.T_total


collision = False
t_star = None
min_clearance = float('inf')
for t in times:
    # 计算所有板凳前/后把手坐标（用于碰撞检测）
    handles = []  # 存储每节板凳前把手坐标，以及龙尾后把手
    theta_head = invert_length(s_head0 + v_head*t, theta_head0)
    handles.append((a*theta_head*np.cos(theta_head), a*theta_head*np.sin(theta_head)))
    for i in range(2, N+1):
        s_i = s_head0 + v_head*t - (D_head + (i-2)*D_body)
        if s_i < 0: 
            s_i = 0.0
        theta_i = invert_length(s_i, theta_head)
        handles.append((a*theta_i*np.cos(theta_i), a*theta_i*np.sin(theta_i)))
    # 加入龙尾后把手位置（尾节后孔距再减D_body）
    theta_tail_rear = invert_length(s_i - D_body, theta_i)
    handles.append((a*theta_tail_rear*np.cos(theta_tail_rear), a*theta_tail_rear*np.sin(theta_tail_rear)))
    
    # 快速筛选：按x、y投影距离过滤明显不可能碰撞的板凳对（例如相差两倍板凳长度以上直接跳过）
    # 精细检测：计算剩余候选对的线段最小距离
    n_handles = len(handles)
    min_dist = float('inf')
    for i in range(n_handles-2):        # 相邻板凳不需检测
        for j in range(i+2, n_handles):
            # 快速排除：以曼哈顿距离为粗略条件
            xi, yi = handles[i]; xj, yj = handles[j]
            if abs(xi-xj) > D_body*2 or abs(yi-yj) > D_body*2:
                continue
            # 计算板凳i和j线段最小距离
            # 板凳i线段端点：handles[i]前把手->handles[i+1]前把手（或尾后把手）
            if j == i+1: 
                continue  # 跳过相邻
            Ax, Ay = handles[i]; Bx, By = handles[i+1] if i+1 < n_handles else handles[i]  # i的后把手
            Cx, Cy = handles[j]; Dx, Dy = handles[j+1] if j+1 < n_handles else handles[j]
            # 计算线段AB与CD最短距离
            def point_line_dist(px, py, ax, ay, bx, by):
                # 投影参数t
                if ax == bx and ay == by:
                    return np.hypot(px-ax, py-ay)
                t = max(0, min(1, ((px-ax)*(bx-ax)+(py-ay)*(by-ay)) / ((bx-ax)**2+(by-ay)**2)))
                projx, projy = ax + t*(bx-ax), ay + t*(by-ay)
                return np.hypot(px-projx, py-projy)
            # 4种情况取最小
            d1 = point_line_dist(Ax,Ay,Cx,Cy,Dx,Dy)
            d2 = point_line_dist(Bx,By,Cx,Cy,Dx,Dy)
            d3 = point_line_dist(Cx,Cy,Ax,Ay,Bx,By)
            d4 = point_line_dist(Dx,Dy,Ax,Ay,Bx,By)
            dist = min(d1,d2,d3,d4)
            if dist < min_dist:
                min_dist = dist
    # 更新最小间隙
    if min_dist < min_clearance:
        min_clearance = min_dist
    # 判断碰撞
    if min_dist <= 0.30:  # 达到板宽阈值
        collision = True
        t_star = t
        break

if collision:
    print(f"检测到碰撞，链条无法继续盘入。终止时刻 t* = {t_star} s")
else:
    print("链条在模拟时段内未发生碰撞。")
# 碰撞发生时刻前一秒的数据作为输出
result_time = t_star - 1 if collision else T_total
result_data = output.loc[[result_time]]
result_data.to_excel("result2.xlsx", index=False)
