import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation
import control
from scipy.linalg import sqrtm
from scipy.linalg import pinv, svd

pi = np.pi
cos = np.cos
sin = np.sin
exp = np.exp
tan = np.tan
inv = np.linalg.inv
norm = np.linalg.norm
atan = np.arctan
sign = np.sign
sqrt = np.sqrt
asin = np.arcsin
acos = np.arccos
mat = np.matmul


class Estimator(object):
    def __init__(self, x_hat, u, y, P, Q, R, f_sys, h, Ts):
        self.Ts = Ts
        self.Q = Q
        self.R = R
        self.n = np.shape(Q)[0]
        self.k = np.shape(R)[0]
        self.x_hat = x_hat
        self.u = u
        self.P = P
        self.y = y
        self.f_sys = lambda x, u: f_sys(x, u)
        self.f = lambda x: f_sys(x, self.u)
        self.h = lambda x: h(x)
        self.P_history = [self.P.copy()]

    def update(self, y, u):
        self.y = y
        self.u = u

    def symmetrize(self, P):
        """
        对称化矩阵 P。

        参数：
            P (np.ndarray): 待对称化的矩阵。

        返回：
            np.ndarray: 对称化后的矩阵。
        """
        return (P + P.T) / 2

    def computeJacobian(self, fs, X_op):
        N = len(X_op)
        M = len(fs(X_op))
        J = np.zeros((N, M))
        I = np.identity(N)
        epsilon = 1e-6

        for i in range(N):
            J[i, :] = (fs(X_op + epsilon * I[i, :]) - fs(X_op - epsilon * I[i, :])) / 2 / epsilon

        return np.transpose(J)

    def EKF(self):
        F = self.computeJacobian(self.f, self.x_hat)
        H = self.computeJacobian(self.h, self.x_hat)
        x_hat_prev = self.f_sys(self.x_hat, self.u)
        P_prev = np.matmul(F, np.matmul(self.P, F.T)) + self.Q
        Kg = np.matmul(P_prev, np.matmul(H.T, pinv(self.R + np.matmul(H, np.matmul(P_prev, H.T)))))
        self.x_hat = x_hat_prev + np.matmul(Kg, (self.y - self.h(x_hat_prev)))
        self.P = np.matmul((np.eye(self.n) - np.matmul(Kg, H)), P_prev)

    def meanVarFH(self, X, W):
        """
        计算通过系统和观测函数后的均值和协方差。

        参数：
            X (np.ndarray): 通过系统动态函数传播的 sigma 点，形状为 (2n+1, n)。
            W (np.ndarray): 权重，形状为 (2n+1,)。

        返回：
            tuple: 包含预测均值 x_post, 预测观测 y_pred, 预测协方差 P_post,
                   测量协方差 S, 和交叉协方差 T。
        """
        n = self.n
        k = self.k
        l = 2 * n + 1
        Z = np.zeros((l, n + k))
        for i in range(l):
            Z[i] = np.concatenate((self.f(X[i]), self.h(X[i])))
        meanZ = np.average(Z.T, axis=1, weights=W)
        covZ = np.cov(Z.T, ddof=0, aweights=W)
        return meanZ[:n], meanZ[n:], covZ[:n, :n] + self.Q, covZ[n:, n:] + self.R, covZ[:n, n:]

    def UKF(self):
        """
        执行 Unscented Kalman Filter 的预测和更新步骤。
        """
        # 生成 sigma 点和权重
        X_apri, W_apri, W_c = self.sigmaPoints_SVD()

        # 通过系统动态函数传播 sigma 点
        X_pred = np.array([self.f_sys(x, self.u) for x in X_apri])

        # 计算预测均值
        x_pred = np.sum(W_apri[:, np.newaxis] * X_pred, axis=0)

        # 计算预测协方差
        P_pred = self.Q.copy()
        for i in range(2 * self.n + 1):
            diff = X_pred[i] - x_pred
            P_pred += W_c[i] * np.outer(diff, diff)

        # 更新估计值和协方差
        self.x_hat = x_pred
        self.P = self.symmetrize(P_pred)
        self.P_history.append(self.P.copy())

        # 生成新的 sigma 点和权重
        X_post, W_post, W_c_post = self.sigmaPoints_SVD()

        # 通过观测函数传播 sigma 点
        Y_pred = np.array([self.h(x) for x in X_post])

        # 计算预测观测均值
        y_pred = np.sum(W_post[:, np.newaxis] * Y_pred, axis=0)

        # 计算观测协方差 S 和交叉协方差 T
        S = self.R.copy()
        T = np.zeros((self.n, self.k))
        for i in range(2 * self.n + 1):
            diff_y = Y_pred[i] - y_pred
            diff_x = X_post[i] - self.x_hat
            S += W_c_post[i] * np.outer(diff_y, diff_y)
            T += W_c_post[i] * np.outer(diff_x, diff_y)

        # 计算 Kalman 增益
        K = T @ pinv(S)

        # 更新状态估计和协方差矩阵
        self.x_hat = self.x_hat + K @ (self.y - y_pred)
        self.P = self.P - K @ S @ K.T
        self.P = self.symmetrize(self.P)
        self.P_history.append(self.P.copy())

    def sigmaPoints_SVD(self):
        """
        使用奇异值分解（SVD）生成 UKF 的 sigma 点。

        返回：
            X (np.ndarray): Sigma 点矩阵，形状为 (2n+1, n)。
            W_m (np.ndarray): 均值权重，形状为 (2n+1,)。
            W_c (np.ndarray): 协方差权重，形状为 (2n+1,)。
        """
        k = 2  # 缩放参数
        n = self.n

        # 初始化 sigma 点矩阵
        X = np.zeros((2 * n + 1, n))

        # 计算权重
        W_m = np.full(2 * n + 1, 1 / (2 * (n + k)))
        W_c = np.full(2 * n + 1, 1 / (2 * (n + k)))
        W_m[0] = k / (n + k)
        W_c[0] = k / (n + k) + (1 - 0.01 ** 2 + 2)  # 假设 alpha=0.01, beta=2

        # 确保协方差矩阵是对称的
        P_sym = self.symmetrize(self.P)

        # 执行 SVD 分解
        U, Sigma, VT = svd(P_sym)

        # 构建协方差矩阵的平方根
        sqrt_P = U @ np.diag(np.sqrt(Sigma))

        # 计算缩放因子
        scaling = np.sqrt(n + k)

        # 生成 sigma 点
        X[0] = self.x_hat
        for i in range(n):
            X[i + 1] = self.x_hat + scaling * sqrt_P[:, i]
            X[n + i + 1] = self.x_hat - scaling * sqrt_P[:, i]

        return X, W_m, W_c


"""
Dynamic Inversion - Inner Loop PID Controller
"""


class PID(object):
    def __init__(self, x0):
        self.x = x0
        self.attSet = np.zeros(3)
        self.ref = np.zeros(10)
        self.u = np.zeros(4)
        self.e_z = 0
        self.e_z_prev = 0
        self.e_z_acc = 0
        self.e_phi = 0
        self.e_phi_prev = 0
        self.e_phi_acc = 0
        self.e_theta = 0
        self.e_theta_prev = 0
        self.e_theta_acc = 0
        self.e_psi = 0
        self.e_psi_prev = 0
        self.e_psi_acc = 0
        self.Kp_z = 4
        self.Ki_z = 2
        self.Kd_z = 3
        self.Kp_phi = 2.5
        self.Ki_phi = 0
        self.Kd_phi = 5 * 1.5
        self.Kp_theta = 2.5
        self.Ki_theta = 0
        self.Kd_theta = 5 * 1.5
        self.Kp_psi = 0.25
        self.Ki_psi = 0
        self.Kd_psi = 0.75
        self.Kp_x = 5
        self.Kv_x = 10
        self.Kp_y = 5
        self.Kv_y = 10

    def update(self, x, ref):
        self.x = x
        self.ref = ref

    def attitudePID(self):
        xddot = self.ref[6:][0] + self.Kv_x * (self.ref[3:][0] - self.x[1]) + self.Kp_x * (self.ref[0] - self.x[0])
        yddot = self.ref[6:][1] + self.Kv_y * (self.ref[3:][1] - self.x[3]) + self.Kp_y * (self.ref[1] - self.x[2])
        zddot = self.u[0] / m + g
        psi_ref = self.ref[-1]
        phi_ref = asin(
            np.clip((xddot * sin(psi_ref) - yddot * cos(psi_ref) + self.x[12]) / zddot, -sin(pi / 2), sin(pi / 2)))
        theta_ref = asin(
            np.clip((xddot * cos(psi_ref) + yddot * sin(psi_ref) + self.x[13]) / zddot / cos(phi_ref), -sin(pi / 2),
                    sin(pi / 2)))
        self.attSet = np.array([
            phi_ref,
            theta_ref,
            psi_ref
        ])

        self.e_phi = (self.attSet[0] - self.x[6])
        phi_out = self.Kp_phi * self.e_phi + self.Ki_phi * self.e_phi_acc + self.Kd_phi * (
                self.e_phi - self.e_phi_prev) / Ts
        self.e_phi_acc += self.e_phi * Ts
        self.e_phi_prev = self.e_phi

        self.e_theta = (self.attSet[1] - self.x[8])
        theta_out = self.Kp_theta * self.e_theta + self.Ki_theta * self.e_theta_acc + self.Kd_theta * (
                self.e_theta - self.e_theta_prev) / Ts
        self.e_theta_acc += self.e_theta * Ts
        self.e_theta_prev = self.e_theta

        self.e_psi = (self.attSet[2] - self.x[10])
        psi_out = self.Kp_psi * self.e_psi + self.Ki_psi * self.e_psi_acc + self.Kd_psi * (
                self.e_psi - self.e_psi_prev) / Ts
        self.e_psi_acc += self.e_psi * Ts
        self.e_psi_prev = self.e_psi

        self.u[1] = phi_out
        self.u[2] = theta_out
        self.u[3] = psi_out

    def altitudePID(self):
        zSet = self.ref[2]
        self.e_z = zSet - self.x[4]
        z_out = self.Kp_z * self.e_z + self.Ki_z * self.e_z_acc + self.Kd_z * (self.e_z - self.e_z_prev) / Ts
        self.e_z_prev = self.e_z
        self.e_z_acc += self.e_z * Ts
        self.u[0] = z_out / (cos(self.x[6]) * cos(self.x[8]))



# Time step size

Ts = 1e-2
freq = 5.0
w = 2 * pi * freq / 100
tau = 10

xyA = 10

zH = 30
trajectory = lambda t: np.array([xyA * cos(w * t - pi / 2) / (1 + sin(w * t - pi / 2) ** 2) * (1 - exp(-t / tau)),
                                 xyA * sin(w * t - pi / 2) * cos(w * t - pi / 2) / (1 + sin(w * t - pi / 2) ** 2) * (
                                         1 - exp(-t / tau)),
                                 zH * (1 - exp(-t / tau))])

trajectory_dot = lambda t: np.array([xyA * (-2 * w * (-5 + cos(2 * w * t)) * cos(w * t) / (3 + cos(2 * w * t)) ** 2) * (
        1 - exp(-t / tau)) + xyA * cos(w * t - pi / 2) / (1 + sin(w * t - pi / 2) ** 2) * exp(-t / tau) / tau,
                                     xyA * (-2 * w * (1 + 3 * cos(2 * w * t)) / (3 + cos(2 * w * t)) ** 2) * (
                                             1 - exp(-t / tau)) + xyA * sin(w * t - pi / 2) * cos(
                                         w * t - pi / 2) / (1 + sin(w * t - pi / 2) ** 2) * exp(-t / tau) / tau,
                                     zH * exp(-t / tau) / tau])

trajectory_ddot = lambda t: np.array([xyA * (
        -w * w * (2 * sin(w * t) - 45 * sin(3 * w * t) + sin(5 * w * t)) / (3 + cos(2 * w * t)) ** 3 / 2) + xyA * (
                                              -2 * w * (-5 + cos(2 * w * t)) * cos(w * t) / (
                                              3 + cos(2 * w * t)) ** 2) * exp(-t / tau) / tau - xyA * cos(
    w * t - pi / 2) / (1 + sin(w * t - pi / 2) ** 2) * exp(-t / tau) / tau / tau,
                                      xyA * (4 * w * w * (7 - 3 * cos(2 * w * t)) * sin(2 * w * t)) / (
                                              3 + cos(2 * w * t)) ** 3 + xyA * (
                                              -2 * w * (1 + 3 * cos(2 * w * t)) / (3 + cos(2 * w * t)) ** 2) * exp(
                                          -t / tau) / tau - xyA * sin(w * t - pi / 2) * cos(w * t - pi / 2) / (
                                              1 + sin(w * t - pi / 2) ** 2) * exp(-t / tau) / tau / tau,
                                      -zH * exp(-t / tau) / tau / tau])
# Iniital Condition
X0 = np.array([trajectory(0)[0], trajectory_dot(0)[0], trajectory(0)[1], trajectory_dot(0)[1], trajectory(0)[2],
               trajectory_dot(0)[2], 0, 0, 0, 0, 0, 0])

# Initial estimation state

X0_hat = np.array([trajectory(0)[0], trajectory_dot(0)[0], trajectory(0)[1], trajectory_dot(0)[1], trajectory(0)[2],
               trajectory_dot(0)[2], 0, 0, 0, 0, 0, 0,0,0,0])

# Kalman Filter Tuning Parameters

P = np.diag([0, 0, 0, 0, 0, 0, 1e-1, 1e-1, 1e-1, 1e-1, 1e-1, 1e-1, 1e-4, 1e-4, 1e-4])

Q = np.diag([1e-4, 1e-4, 1e-4, 1e-4, 1e-4, 1e-4, 1e-4, 1e-4, 1e-4, 1e-4, 1e-4, 1e-4, 0.05, 0.05, 0.05])
dr = 0.03
R = np.eye(6) * dr

# Drone Parameters
m = 0.468
g = 9.8
Kf = 2.980 * (10 ** -3)
Km = 2.980 * (10 ** -3)
k = Km / Kf
l = 0.225
b = 0.010
Ixx = 4.856e-3
Iyy = 4.856e-3
Izz = 8.801e-3



def computeJacobian(fs, X_op):
    N = len(X_op)
    J = np.zeros((N, N))
    I = np.identity(N)
    epsilon = 1e-6

    for i in range(N):
        J[i, :] = (fs(X_op + epsilon * I[i, :]) - fs(X_op - epsilon * I[i, :])) / 2 / epsilon

    return np.transpose(J)


pid = PID(X0_hat)


T_transform = np.array([[Kf, Kf, Kf, Kf],
                        [0, -l * Kf, 0, l * Kf],
                        [-l * Kf, 0, l * Kf, 0],
                        [-Km, Km, -Km, Km]
                        ])
# Inertia Tensor
I = np.array([[Ixx, 0, 0],
              [0, Iyy, 0],
              [0, 0, Izz]
              ])
# Drag Coeffiecient
A_drag = np.array([[0.10, 0, 0],
                   [0, 0.10, 0],
                   [0, 0, 0.10]
                   ])
# Yaw rotation matrix
R_psi = lambda X: np.array([[cos(X[10]), sin(X[10]), 0],
                            [-sin(X[10]), cos(X[10]), 0],
                            [0, 0, 1]
                            ])
# Pitch rotation matrix
R_theta = lambda X: np.array([[cos(X[8]), 0, -sin(X[8])],
                              [0, 1, 0],
                              [sin(X[8]), 0, cos(X[8])]
                              ])
# Roll rotation matrix
R_phi = lambda X: np.array([[1, 0, 0],
                            [0, cos(X[6]), sin(X[6])],
                            [0, -sin(X[6]), cos(X[6])]
                            ])
# Euler rates to Angular Velocity
W_n = lambda X: np.array([[1, 0, -sin(X[8])],
                          [0, cos(X[6]), cos(X[8]) * sin(X[6])],
                          [0, -sin(X[6]), cos(X[8]) * cos(X[6])]
                          ])
# Inertial to Body Rotation
R_ib = lambda X: np.matmul(R_phi(X), np.matmul(R_theta(X), R_psi(X)))
# Body to Inertial Rotation
R_bi = lambda X: np.transpose(R_ib(X))
# Exogenous Force on Drone
F_ext = lambda X, w: np.matmul(R_bi(X), np.array([0, 0, np.matmul(T_transform, (w))[0]]))
# Exogenous Moment on Drone
T_ext = lambda X, w: np.matmul(R_bi(X), np.matmul(T_transform, (w))[1:])
# Rotational matrix
J = lambda X: np.matmul(np.transpose(W_n(X)), np.matmul(I, W_n(X)))
# Coriolis matrix
J_dot = lambda X: np.array([
    [0, 0, -Ixx * cos(X[8]) * X[9]],
    [0, (Izz - Iyy) * sin(2 * X[6]) * X[7],
     (Iyy - Izz) * (-cos(X[6]) * sin(X[8]) * sin(X[6]) * X[9] + cos(X[8]) * cos(2 * X[6]) * X[7])],
    [-Ixx * cos(X[8]) * X[9],
     (Iyy - Izz) * (-cos(X[6]) * sin(X[8]) * sin(X[6]) * X[9] + cos(X[8]) * cos(2 * X[6]) * X[7]), -2 * cos(X[8]) * (
             sin(X[8]) * (-Ixx + Izz * cos(X[6]) ** 2 + Iyy * sin(X[6]) ** 2) * X[9] + (Izz - Iyy) * cos(
         X[8]) * cos(X[6]) * sin(X[6]) * X[7])]
])
# Generalized translational force
Q_trans = lambda X: np.array([0,
                              0,
                              -m * g
                              ])
# Generalized rotational force
Q_rot = lambda X: np.array([-(Iyy - Izz) * (
        sin(2 * X[6]) * X[9] ** 2 - X[9] * X[11] * cos(2 * X[6]) * cos(X[8]) - cos(X[6]) * sin(X[6]) * cos(
    X[8]) ** 2 * X[11] ** 2),
                            X[11] * (-2 * Ixx * X[7] * cos(X[8]) + sin(X[8]) * (
                                    (Izz - Iyy) * sin(2 * X[6]) * X[9] + 2 * cos(X[8]) * (
                                    Ixx - Izz * cos(X[6]) ** 2 - Iyy * sin(X[6]) ** 2) * X[11])),
                            0
                            ])
# xi_ddot
F_x = lambda X, w: (1 / m) * (-np.matmul(A_drag, X[1::2][:3]) + Q_trans(X) + F_ext(X, w))
# eta_ddot
F_eta = lambda X, w: np.linalg.solve(J(X), (-np.matmul(J_dot(X), X[1::2][3:6]) + Q_rot(X) + T_ext(X, w)))

F_wind_x = lambda t: sin(24* pi * t / 50) * 2

F_wind_y = lambda t: sin(36*pi * t / 50) * 2


# System function
def fs(t, X):
    X_dot = np.zeros(len(X))
    X_dot[::2] = X[1::2]
    X_dot[1::2][:3] = (1 / m) * (-np.matmul(A_drag, X[1::2][:3]) + Q_trans(X)) + np.array(
        [F_wind_x(t), F_wind_y(t), 0]) / m
    X_dot[1::2][3:6] = np.linalg.solve(J(X), (-np.matmul(J_dot(X), X[1::2][3:6]) + Q_rot(X)))
    return X_dot


# Control function
def gs(X):
    G = np.zeros([len(X), 4])
    G[1::2, :][:3, :] = np.matmul(R_bi(X) / m, np.array([[0, 0, 0, 0], [0, 0, 0, 0], [Kf, Kf, Kf, Kf]]))
    G[1::2, :][3:6, :] = np.matmul(np.linalg.solve(J(X), R_bi(X)),
                                   np.array([[0, -l * Kf, 0, l * Kf], [-l * Kf, 0, l * Kf, 0], [-Km, Km, -Km, Km]]))
    return G


# Dynamic Inversion Controller
def dynamicInversion(t, x):
    ref = np.zeros(10)
    ref[:3] = trajectory(t)
    ref[3:6] = trajectory_dot(t)
    ref[6:9] = trajectory_ddot(t)
    ref[-1] = 0
    v = np.zeros(12)
    pid.update(x, ref)
    pid.altitudePID()
    pid.attitudePID()
    v[5::2] = pid.u
    return mat(pinv(gs(x[:12])), v - fs(t, x[:12]))

def dynamic_function(t,X,u):
    X_dot=fs(t, X)
    X_dot[1::2][:3]+=np.matmul(np.matmul(R_bi(X) / m, np.array([[0, 0, 0, 0], [0, 0, 0, 0], [Kf, Kf, Kf, Kf]])),u)
    X_dot[1::2][3:6]+=np.matmul(np.matmul(np.linalg.solve(J(X), R_bi(X)),
                                   np.array([[0, -l * Kf, 0, l * Kf], [-l * Kf, 0, l * Kf, 0], [-Km, Km, -Km, Km]])),u)
    return X_dot
def rk4_dynamics(t,x,u,h=Ts):
    k1=dynamic_function(t,x,u)
    k2=dynamic_function(t,x+0.5*h*k1,u)
    k3=dynamic_function(t,x+0.5*h*k2,u)
    k4=dynamic_function(t,x+h*k3,u)
    return x+(h/6.0)*(k1+2*k2+2*k3+k4)



def f_est(x, u):
    return x + np.hstack((Ts * (fs(0, x[:-3]) + np.matmul(gs(x[:-3]), u) + np.array(
        [0, x[12], 0, x[13], 0, 0, 0, 0, 0, 0, 0, 0]) / m), np.zeros(3)))


def plotOutput():
    plt.figure(figsize=(16, 9))
    plt.subplot(211)
    plt.plot(t, x, label='$x$')
    plt.plot(t, y, label='$y$')
    plt.plot(t, z, c='#e62e00', label='$z$')
    plt.plot(t, xSet, label='$x_{Traj}$', lw=2.5, linestyle='--', c='tab:blue')
    plt.plot(t, ySet, label='$y_{Traj}$', lw=2.5, linestyle='--', c='tab:orange')
    plt.plot(t, zSet, label='$z_{Traj}$', linestyle='--', c='tab:red', lw=2.5)
    plt.xlabel('time in seconds', fontsize=18)
    plt.ylabel('distance in $m$', fontsize=18)
    plt.title('Translational coordinates', fontsize=18)
    plt.legend(fontsize=18)
    plt.grid()

    plt.subplot(212)
    plt.plot(t, 180 * phi / pi, label='$\phi$', lw=2.5)
    plt.plot(t, 180 * theta / pi, label='$\Theta$', lw=2.5)
    plt.plot(t, 180 * psi / pi, c='#e62e00', label='$\psi$', lw=2.5)
    plt.xlabel('time in seconds', fontsize=18)
    plt.ylabel('angle in $^\circ$', fontsize=18)
    plt.title('Rotational coordinates', fontsize=18)
    plt.legend(fontsize=18)
    plt.grid()
    plt.subplots_adjust(hspace=0.4)
    plt.savefig("state1.png")
    plt.show()

    plt.figure(figsize=(16, 9))
    plt.subplot(211)
    plt.plot(t, xdot, label='$\dot{x}$', lw=2.5)
    plt.plot(t, ydot, label='$\dot{y}$', lw=2.5)
    plt.plot(t, zdot, c='#e62e00', label='$\dot{z}$', lw=2.5)
    plt.plot(t, trajectory_dot(t)[0], label='$vx_{Traj}$', lw=2.5, linestyle='--', c='tab:blue')
    plt.plot(t, trajectory_dot(t)[1], label='$vy_{Traj}$', lw=2.5, linestyle='--', c='tab:orange')
    plt.plot(t, trajectory_dot(t)[2], label='$vz_{Traj}$', linestyle='--', c='tab:red', lw=2.5)
    plt.xlabel('time in seconds', fontsize=18)
    plt.ylabel('distance in $m/s$', fontsize=18)
    plt.title('Translational velocities', fontsize=18)
    plt.legend(fontsize=18)
    plt.grid()

    plt.subplot(212)
    plt.plot(t, 180 * phidot / pi, label='$\dot{\phi}$', lw=2.5)
    plt.plot(t, 180 * thetadot / pi, label='$\dot{\Theta}$', lw=2.5)
    plt.plot(t, 180 * psidot / pi, c='#e62e00', label='$\dot{\psi}$', lw=2.5)
    plt.xlabel('time in seconds', fontsize=18)
    plt.ylabel('angular rate in $^\circ/s$', fontsize=18)
    plt.title('Rotational velocities', fontsize=18)
    plt.legend(fontsize=18)
    plt.grid()
    plt.subplots_adjust(hspace=0.4)
    plt.savefig("state2.png")
    plt.show()


def plotestimate():
    plt.figure(figsize=(16, 9))
    plt.subplot(211)
    plt.plot(t, x_est, label='$\hat{x}$')
    plt.plot(t, y_est, label='$\hat{y}$')
    plt.plot(t, z_est, c='#e62e00', label='$\hat{z}$')
    plt.plot(t, x, label='$x_{True}$', lw=2.5, linestyle='--', c='tab:blue')
    plt.plot(t, y, label='$y_{True}$', lw=2.5, linestyle='--', c='tab:orange')
    plt.plot(t, z, label='$z_{True}$', linestyle='--', c='tab:red', lw=2.5)
    plt.xlabel('time in seconds', fontsize=18)
    plt.ylabel('distance in $m$', fontsize=18)
    plt.title('Translational coordinates', fontsize=18)
    plt.legend(fontsize=18)
    plt.grid()

    plt.subplot(212)
    plt.plot(t, xdot_est, label='$\dot{\hat{x}}$', lw=2.5)
    plt.plot(t, ydot_est, label='$\dot{\hat{y}}$', lw=2.5)
    plt.plot(t, zdot_est, c='#e62e00', label='$\dot{\hat{z}}$', lw=2.5)
    plt.plot(t, xdot, label='$vx_{True}$', lw=2.5, linestyle='--', c='tab:blue')
    plt.plot(t, ydot, label='$vy_{True}$', lw=2.5, linestyle='--', c='tab:orange')
    plt.plot(t, zdot, label='$vz_{True}$', linestyle='--', c='tab:red', lw=2.5)
    plt.xlabel('time in seconds', fontsize=18)
    plt.ylabel('distance in $m/s$', fontsize=18)
    plt.title('Translational velocities', fontsize=18)
    plt.legend(fontsize=18)
    plt.grid()
    plt.subplots_adjust(hspace=0.4)
    plt.savefig("estimates.png")
    plt.show()

def plotControlEffort():
    plt.figure(figsize=(16, 9))
    plt.subplot(221)
    plt.plot(t, U[:, 0], lw=2.5)
    plt.xlabel('time in seconds', fontsize=18)
    plt.ylabel('Rotor RPM', fontsize=18)
    plt.legend(fontsize=18)
    plt.title('Rotor - 1 RPM (Saturation RPM = 1320 rpm)', fontsize=18)
    plt.grid()
    plt.subplot(222)
    plt.plot(t, U[:, 1], lw=2.5)
    plt.xlabel('time in seconds', fontsize=18)
    plt.ylabel('Rotor RPM', fontsize=18)
    plt.legend(fontsize=18)
    plt.title('Rotor - 2 RPM (Saturation RPM = 1320 rpm)', fontsize=18)
    plt.grid()
    plt.subplot(223)
    plt.plot(t, U[:, 2], lw=2.5)
    plt.xlabel('time in seconds', fontsize=18)
    plt.ylabel('Rotor RPM', fontsize=18)
    plt.legend(fontsize=18)
    plt.title('Rotor - 3 RPM (Saturation RPM = 1320 rpm)', fontsize=18)
    plt.grid()
    plt.subplot(224)
    plt.plot(t, U[:, 3], lw=2.5)
    plt.xlabel('time in seconds', fontsize=18)
    plt.ylabel('Rotor RPM', fontsize=18)
    plt.legend(fontsize=18)
    plt.title('Rotor - 4 RPM (Saturation RPM = 1320 rpm)', fontsize=18)
    plt.grid()
    plt.subplots_adjust(hspace=0.4)
    plt.savefig("control.png")
    plt.show()


def plotwind():
    plt.figure(figsize=(16, 9))
    plt.subplot(311)
    plt.plot(t, F_wind_x(t), label='Wind Force in x-axis', lw=2.5)
    plt.plot(t, X_hat[:, 12], label='Estimated Wind Force in x-axis', lw=2.5)
    plt.xlabel('time in seconds', fontsize=18)
    plt.ylabel('F_wind_x', fontsize=18)
    plt.legend(fontsize=18)
    plt.title('Wind Disturbance Estimation and Rejection', fontsize=18)
    plt.grid()
    plt.subplot(312)
    plt.plot(t, F_wind_y(t), label='Wind Force in y-axis', lw=2.5)
    plt.plot(t, X_hat[:, 13], label='Estimated Wind Force in y-axis', lw=2.5)
    plt.xlabel('time in seconds', fontsize=18)
    plt.ylabel('F_wind_y', fontsize=18)
    plt.legend(fontsize=18)
    plt.grid()

    plt.subplots_adjust(hspace=0.4)
    plt.savefig("wind_est.png")
    plt.show()


"""
Plot 3D Trajectory
"""


def plot3DTrajectory():
    fig = plt.figure()
    ax = plt.axes(projection='3d')

    # Data for a three-dimensional line
    zline = z
    xline = x
    yline = y
    ax.plot3D(xline, yline, zline, 'tab:blue', lw=2.5, label='Actual Drone Trajectory')
    zdata = zSet[::100]
    xdata = xSet[::100]
    ydata = ySet[::100]
    ax.scatter3D(xdata, ydata, zdata, c='tab:red', s=50, label='Waypoints')
    plt.title('Drone 3D Cascade Controller Trajectory Tracking', fontsize=18)
    ax.set_xlabel('x', fontsize=18)
    ax.set_ylabel('y', fontsize=18)
    ax.set_zlabel('z', fontsize=18)
    plt.legend(fontsize=18)
    plt.show()

def F(t, x):
    u_di = dynamicInversion(t, estimator.x_hat)
    y_true = h(x)
    noise_scale = 0.1
    # 生成与 y_true 形状相同的标准正态分布噪声
    multiplicative_noise = noise_scale * y_true * np.random.normal(0, dr, size=y_true.shape)
    # 添加噪声
    y_noisy = y_true + multiplicative_noise
    estimator.update(y_noisy, u_di)
    estimator.EKF()
    return rk4_dynamics(t, x, u_di), estimator.x_hat, u_di, pid.attSet
h = lambda X: X[1::2]    #速度

h_est = lambda X: X[:-3][1::2]

estimator = Estimator(X0_hat, dynamicInversion(0, X0_hat), h_est(X0_hat), P, Q, R, f_est, h_est, Ts)

N = 5000

t = np.arange(N) * Ts

X = np.zeros([N, 12])

X_hat = np.zeros([N, 15])

ref = np.zeros([N, 3])

U = np.zeros([N, 4])

X[0] = X0

X_hat[0] = X0_hat
xSet = trajectory(t)[0]
ySet = trajectory(t)[1]
zSet = trajectory(t)[2]
for k in range(N - 1):
    X[k + 1], X_hat[k + 1], U[k + 1], ref[k + 1] = F(k * Ts, X[k])


U[0] = U[1]

x = X[:, 0]
y = X[:, 2]
z = X[:, 4]

x_est = X_hat[:, 0]
y_est = X_hat[:, 2]
z_est = X_hat[:, 4]

phi = X[:, 6]
theta = X[:, 8]
psi = X[:, 10]

wind_x_hat=X_hat[:, 12]
wind_y_hat=X_hat[:, 13]

xdot = X[:, 1]
ydot = X[:, 3]
zdot = X[:, 5]

xdot_est = X_hat[:, 1]
ydot_est = X_hat[:, 3]
zdot_est = X_hat[:, 5]

phidot = X[:, 7]
thetadot = X[:, 9]
psidot = X[:, 11]

xSet = trajectory(t)[0]
ySet = trajectory(t)[1]
zSet = trajectory(t)[2]

"""
Tracking Errors
"""

x_mse = np.sqrt(np.mean(np.square(x_est - x)))
y_mse = np.sqrt(np.mean(np.square(y_est - y)))
z_mse = np.sqrt(np.mean(np.square(z_est - z)))
print(x_mse,y_mse,z_mse)
x_mse = np.sqrt(np.mean(np.square(xSet - x)))
y_mse = np.sqrt(np.mean(np.square(ySet - y)))
z_mse = np.sqrt(np.mean(np.square(zSet - z)))
print(x_mse,y_mse,z_mse)
wind_x_mse = np.sqrt(np.mean(np.square(wind_x_hat - F_wind_x(t))))
wind_y_mse = np.sqrt(np.mean(np.square(wind_y_hat - F_wind_y(t))))
print(wind_x_mse,wind_y_mse)
plotOutput()
plotestimate()
plotwind()
plotControlEffort()
plot3DTrajectory()
K = 10

T = 80
x_c = x[::K]
y_c = y[::K]
z_c = z[::K]

a = phi[::K]
b = theta[::K]
c = psi[::K]

l = 4.0

x_x1 = x_c + l * cos(b) * cos(c)
y_x1 = y_c + l * cos(b) * sin(c)
z_x1 = z_c + (-l) * sin(b)

x_x2 = x_c - l * cos(b) * cos(c)
y_x2 = y_c - l * cos(b) * sin(c)
z_x2 = z_c - (-l) * sin(b)

x_y1 = x_c + l * cos(c) * sin(b) * sin(a) - l * cos(a) * sin(c)
y_y1 = y_c + l * cos(a) * cos(c) + l * sin(b) * sin(a) * sin(c)
z_y1 = z_c + l * cos(b) * sin(a)

x_y2 = x_c - l * cos(c) * sin(b) * sin(a) + l * cos(a) * sin(c)
y_y2 = y_c - l * cos(a) * cos(c) - l * sin(b) * sin(a) * sin(c)
z_y2 = z_c - l * cos(b) * sin(a)

fig = plt.figure(figsize=(12, 12))
ax = fig.add_subplot(projection='3d')

i = 0

line1, = ax.plot(np.array([x_x2[i], x_x1[i]]), np.array([y_x2[i], y_x1[i]]), np.array([z_x2[i], z_x1[i]]), c='k', lw=4,
                 marker='o')
line2, = ax.plot(np.array([x_y2[i], x_y1[i]]), np.array([y_y2[i], y_y1[i]]), np.array([z_y2[i], z_y1[i]]), c='k', lw=4,
                 marker='o')
line3, = ax.plot(x_c[:i], y_c[:i], z_c[:i], label='Drone Trajectory', lw=2.5)
ax.plot3D(xSet, ySet, zSet, c='tab:orange', linestyle='--', label='Waypoints', lw=0.5);

plt.legend()


def makeFrame(i, line1, line2, line3):
    line1.set_data(np.array([x_x2[i], x_x1[i]]), np.array([y_x2[i], y_x1[i]]))
    line1.set_3d_properties(np.array([z_x2[i], z_x1[i]]))
    line2.set_data(np.array([x_y2[i], x_y1[i]]), np.array([y_y2[i], y_y1[i]]))
    line2.set_3d_properties(np.array([z_y2[i], z_y1[i]]))
    line3.set_data(x_c[:i], y_c[:i])
    line3.set_3d_properties(z_c[:i])


# Setting the axes properties
ax.set_xlim3d([-xyA - 5, xyA + 5])
ax.set_xlabel('X')

ax.set_ylim3d([-xyA - 5, xyA + 5])
ax.set_ylabel('Y')

ax.set_zlim3d([-5, zH + 5])
ax.set_zlabel('Z')

ani = animation.FuncAnimation(fig, makeFrame, N // K, fargs=(line1, line2, line3), interval=1000 / 120, blit=False)

ani.save('EKF.gif', writer='pillow')