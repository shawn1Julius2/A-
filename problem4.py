import numpy as np
from math import acos, cos, sin, pi
from scipy.optimize import minimize

# 固定已知点（调头圆半径4.5m，A和C取圆心对称上下）
R_turn = 4.5
A = np.array([0.0, R_turn])
C = np.array([0.0, -R_turn])

# 目标函数：输入变量x=[R1,R2,Bx,By]，返回路径总长度L
def path_length(x):
    R1, R2, Bx, By = x
    # 圆心位置
    O1 = np.array([0.0, R_turn - R1])   # O1在A正下方
    O2 = np.array([0.0, -R_turn + R2])  # O2在C正上方
    # 计算A-O1-B和C-O2-B的夹角
    A_vec = A - O1  # 向量O1->A
    B1_vec = np.array([Bx,By]) - O1  # 向量O1->B
    C_vec = C - O2  # 向量O2->C
    B2_vec = np.array([Bx,By]) - O2  # 向量O2->B
    # 安全性：避免除零
    if np.linalg.norm(A_vec)<1e-6 or np.linalg.norm(B1_vec)<1e-6 \
       or np.linalg.norm(C_vec)<1e-6 or np.linalg.norm(B2_vec)<1e-6:
        return 1e6
    # 计算夹角（采用点积余弦定理）
    cos1 = np.dot(A_vec, B1_vec)/(np.linalg.norm(A_vec)*np.linalg.norm(B1_vec))
    cos2 = np.dot(C_vec, B2_vec)/(np.linalg.norm(C_vec)*np.linalg.norm(B2_vec))
    cos1 = np.clip(cos1, -1, 1)
    cos2 = np.clip(cos2, -1, 1)
    theta1 = acos(cos1)
    theta2 = acos(cos2)
    return R1*theta1 + R2*theta2

# 约束条件函数组
def constraints(x):
    R1, R2, Bx, By = x
    O1 = np.array([0.0, R_turn - R1])
    O2 = np.array([0.0, -R_turn + R2])
    # (1) B在圆弧1上: 距O1距离=R1
    f1 = (Bx - O1[0])**2 + (By - O1[1])**2 - R1**2
    # (2) B在圆弧2上: 距O2距离=R2
    f2 = (Bx - O2[0])**2 + (By - O2[1])**2 - R2**2
    # (3) 两圆外切: O1O2 = R1+R2
    f3 = (O1[1] - O2[1]) - (R1 + R2)
    return [f1, f2, f3]

# 初始猜测值 (经验值R1=6,R2=3，B取两圆心连线中点之一)
x0 = np.array([6.0, 3.0, 0.0, 0.0])  # 取B初值在原方案上，位于两圆心连线中点
cons = ({'type':'eq', 'fun': lambda x: constraints(x)[0]},
        {'type':'eq', 'fun': lambda x: constraints(x)[1]},
        {'type':'eq', 'fun': lambda x: constraints(x)[2]})
res = minimize(path_length, x0, method='SLSQP', constraints=cons, options={'disp': True})
R1_opt, R2_opt, Bx_opt, By_opt = res.x
L_opt = res.fun
print(f"优化结果: R1*={R1_opt:.3f} m, R2*={R2_opt:.3f} m, 路径总长 L*={L_opt:.3f} m")
