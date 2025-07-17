import numpy as np
import problem1 as p1

# 从问题1导入参数
N = p1.N
D_head = p1.D_head
D_body = p1.D_body
v_head = p1.v_head
p_reference = p1.p  # pitch used in problem1 to compute initial radius

def spiral_length(theta, a):
    """Return arc length of Archimedean spiral r=a*theta."""
    return 0.5 * a * (theta * np.sqrt(theta**2 + 1) + np.arcsinh(theta))


def invert_length(s_target, theta_guess, a):
    """Invert spiral length for given ``a`` using Newton iteration."""
    theta = max(theta_guess, 0.0)
    for _ in range(8):  # Newton iteration converges very fast
        f = 0.5 * a * (theta * np.sqrt(theta**2 + 1) + np.arcsinh(theta)) - s_target
        if abs(f) < 1e-9:
            break
        theta -= f / (a * np.sqrt(theta**2 + 1))
    return theta

def point_line_dist(px, py, ax, ay, bx, by):
    """Return distance from point ``P`` to segment ``AB``."""
    if ax == bx and ay == by:
        return np.hypot(px - ax, py - ay)
    t = max(0.0, min(1.0, ((px - ax) * (bx - ax) + (py - ay) * (by - ay)) / ((bx - ax) ** 2 + (by - ay) ** 2)))
    projx = ax + t * (bx - ax)
    projy = ay + t * (by - ay)
    return np.hypot(px - projx, py - projy)


def segment_distance(a1, a2, b1, b2):
    """Return minimal distance between segments ``a1a2`` and ``b1b2``."""
    Ax, Ay = a1
    Bx, By = a2
    Cx, Cy = b1
    Dx, Dy = b2
    d1 = point_line_dist(Ax, Ay, Cx, Cy, Dx, Dy)
    d2 = point_line_dist(Bx, By, Cx, Cy, Dx, Dy)
    d3 = point_line_dist(Cx, Cy, Ax, Ay, Bx, By)
    d4 = point_line_dist(Dx, Dy, Ax, Ay, Bx, By)
    return min(d1, d2, d3, d4)

# 判定函数：给定螺距p，返回龙头能否无碰撞盘入至4.5m
def can_reach_no_collision(p):
    a = p / (2 * np.pi)
    # initial radius corresponds to the same 16 turns for the given pitch
    r_start = 16 * p
    theta_head0 = r_start / a
    s_head0 = spiral_length(theta_head0, a)
    # 直接检查最终配置：龙头位于半径4.5 m处
    theta_head = 4.5 / a
    s_head = spiral_length(theta_head, a)
    handles = []
    seg_coords = []
    x_head = a * theta_head * np.cos(theta_head)
    y_head = a * theta_head * np.sin(theta_head)
    handles.append((x_head, y_head))
    theta_prev = theta_head
    last_theta = theta_head
    last_s = s_head
    for i in range(2, N + 1):
        s_i = s_head - (D_head + (i - 2) * D_body)
        if s_i < 0:
            break
        theta_i = invert_length(s_i, theta_prev, a)
        xi = a * theta_i * np.cos(theta_i)
        yi = a * theta_i * np.sin(theta_i)
        handles.append((xi, yi))
        seg_coords.append(((handles[-2][0], handles[-2][1]), (xi, yi)))
        theta_prev = theta_i
        last_theta = theta_i
        last_s = s_i
    # 尾部后把手
    s_tail = last_s - D_body
    if s_tail >= 0:
        theta_tail = invert_length(s_tail, last_theta, a)
        xt = a * theta_tail * np.cos(theta_tail)
        yt = a * theta_tail * np.sin(theta_tail)
        handles.append((xt, yt))
        seg_coords.append(((handles[-2][0], handles[-2][1]), (xt, yt)))

    # 碰撞检测：计算非邻接线段间最小距离
    for idx_a in range(len(seg_coords)-2):
        A1, A2 = seg_coords[idx_a]
        for idx_b in range(idx_a+2, len(seg_coords)):
            B1, B2 = seg_coords[idx_b]
            if segment_distance(A1, A2, B1, B2) < 0.30:
                return False
    return True

if __name__ == "__main__":
    # 二分搜索最小螺距
    p_low, p_high = 0.30, 0.55
    p_min = None
    for _ in range(30):  # 二分迭代
        p_mid = 0.5 * (p_low + p_high)
        if can_reach_no_collision(p_mid):
            p_min = p_mid
            p_high = p_mid  # 螺距可以，尝试减小上界
        else:
            p_low = p_mid   # 螺距太小碰撞，增大下界

    # 若找到合适的螺距，进行四舍五入并输出结果
    if p_min is not None:
        p_min = round(p_min, 6)
        print(f"最小可行螺距 p_min = {p_min} m")
        # 验证：用p_min再模拟全程，输出龙头前把手轨迹末端坐标
        # ...（此处可调用问题1、2合并的模拟过程直到龙头r=4.5，输出最终位置等数据）...
    else:
        print("在给定范围内未找到满足条件的螺距。")
