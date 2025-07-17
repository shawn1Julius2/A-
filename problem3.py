import numpy as np
import problem1 as p1

# 从问题1导入参数
N = p1.N
D_head = p1.D_head
D_body = p1.D_body
v_head = p1.v_head

def spiral_length(theta, a):
    """Return arc length of Archimedean spiral r=a*theta."""
    return 0.5 * a * (theta * np.sqrt(theta**2 + 1) + np.arcsinh(theta))


def invert_length(s_target, theta_guess, a):
    """Invert spiral length for given ``a`` using binary search."""
    low, high = 0.0, max(theta_guess, theta_guess + 10)
    while spiral_length(high, a) < s_target:
        low = high
        high *= 2
    for _ in range(50):
        mid = 0.5 * (low + high)
        if spiral_length(mid, a) < s_target:
            low = mid
        else:
            high = mid
    return 0.5 * (low + high)

# 判定函数：给定螺距p，返回龙头能否无碰撞盘入至4.5m
def can_reach_no_collision(p):
    a = p/(2*np.pi)
    # 初始配置
    theta_head0 = 16*2*np.pi
    s_head0 = 0.5*a*(theta_head0*np.sqrt(theta_head0**2+1) + np.arcsinh(theta_head0))
    # 模拟直到龙头半径=4.5或碰撞
    r_head = None
    max_t = 300  # 最多模拟300s或提前结束
    for t in range(max_t + 1):
        s_head = s_head0 - v_head * t
        # 当前龙头半径
        theta_head = invert_length(s_head, theta_head0, a)
        r_head = a*theta_head
        if r_head <= 4.5:
            return True  # 已到调头边界且无碰撞
        # 碰撞检测（与问题2类似，但为简化仅检测板凳前把手距离）
        coords = []
        coords.append((a*theta_head*np.cos(theta_head), a*theta_head*np.sin(theta_head)))
        theta_prev = theta_head
        for i in range(2, N + 1):
            s_i = s_head - (D_head + (i - 2) * D_body)
            if s_i < 0:
                break  # 该节及之后的板凳尚未进入盘头，忽略
            theta_i = invert_length(s_i, theta_prev, a)
            coords.append((a * theta_i * np.cos(theta_i), a * theta_i * np.sin(theta_i)))
            theta_prev = theta_i
        # 快速碰撞判定：检测所有非邻接板凳前把手之间的距离
        m = len(coords)
        min_dist = float('inf')
        for i in range(m-2):
            for j in range(i+2, m):
                dx = coords[i][0] - coords[j][0]
                dy = coords[i][1] - coords[j][1]
                dist = dx*dx + dy*dy
                if dist < min_dist:
                    min_dist = dist
        if min_dist <= (0.30)**2:  # 距离平方小于板宽平方，碰撞
            return False
    # 300 s内仍未到4.5（应不会发生）
    return False

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
