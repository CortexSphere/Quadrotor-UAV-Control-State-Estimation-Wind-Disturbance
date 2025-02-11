import casadi as ca
import numpy as np
import osqp
import matplotlib.pyplot as plt
from matplotlib import animation
from scipy.linalg import  solve_discrete_are
from scipy.sparse import hstack, csr_matrix, eye,vstack
import scipy.sparse as sparse
pi = np.pi
cos = np.cos
sin = np.sin
exp=np.exp



Ts = 1e-2
freq = 5
#reference trajectory
w = 2*pi*freq/100
tau = 10
xyA = 10
zH = 30
def generate_reference_trajectory():
    t_sys=ca.SX.sym("t")
    ref_trajectory=ca.vertcat(xyA*cos(w*t_sys - pi/2)/(1 + sin(w*t_sys - pi/2)**2)*(1 - exp(-t_sys/tau)),
                                     xyA*sin(w*t_sys - pi/2)*cos(w*t_sys - pi/2)/(1 + sin(w*t_sys - pi/2)**2)*(1 - exp(-t_sys/tau)),
                                     zH*(1 - exp(-t_sys/tau))  )
    ref_trajectory_dot=ca.jacobian(ref_trajectory,t_sys)
    ref_trajectory_ddot=ca.jacobian(ref_trajectory_dot,t_sys)
    ref_trajectory_func=ca.Function("ref_trajectory",[t_sys],[ref_trajectory])
    ref_trajectory_dot_func=ca.Function("ref_trajectory_dot", [t_sys],[ref_trajectory_dot])
    ref_trajectory_ddot_func=ca.Function("ref_trajectory_ddot", [t_sys],[ref_trajectory_ddot])
    return ref_trajectory_func, ref_trajectory_dot_func, ref_trajectory_ddot_func

trajectory, trajectory_dot, trajectory_ddot=generate_reference_trajectory()
# Iniital Condition
X0 = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
X0[::2][:3] = trajectory(0).full().flatten()
X0[1::2][:3] = trajectory_dot(0).full().flatten()
Nx=np.size(X0)
Nu=4
# Drone Parameters
m = 0.468
g = 9.8
Kf = 2.980*(10**-3)
Km = 2.980*(10**-3)
k = Km/Kf
l = 0.225

Ixx = 4.856e-3*1.5
Iyy = 4.856e-3*1.5
Izz = 8.801e-3*1.5


def dynamic_function(X,u):
    # Transform matrix from control inputs to system forces
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
         (Iyy - Izz) * (-cos(X[6]) * sin(X[8]) * sin(X[6]) * X[9] + cos(X[8]) * cos(2 * X[6]) * X[7]),
         -2 * cos(X[8]) * (sin(X[8]) * (-Ixx + Izz * cos(X[6]) ** 2 + Iyy * sin(X[6]) ** 2) * X[9] + (Izz - Iyy) * cos(
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

    X_dot = np.zeros(len(X))
    X_dot[::2] = X[1::2]
    X_dot[1::2][:3] = (1/m) * (-np.matmul(A_drag, X[1::2][:3]) + Q_trans(X))+np.matmul(np.matmul(R_bi(X)/m, np.array([[0, 0, 0, 0], [0, 0, 0, 0], [Kf, Kf, Kf, Kf]])),u)
    X_dot[1::2][3:6] = (np.linalg.solve(J(X), (-np.matmul(J_dot(X), X[1::2][3:6]) + Q_rot(X)))
                        +np.matmul(np.matmul(np.linalg.solve(J(X), R_bi(X)), np.array([[0, -l*Kf, 0, l*Kf], [-l*Kf, 0, l*Kf, 0], [-Km, Km, -Km, Km]])),u))
    return X_dot

def rk4_dynamics(x,u,h=Ts):
    k1=dynamic_function(x,u)
    k2=dynamic_function(x+0.5*h*k1,u)
    k3=dynamic_function(x+0.5*h*k2,u)
    k4=dynamic_function(x+h*k3,u)
    return x+(h/6.0)*(k1+2*k2+2*k3+k4)

def linearize_dynamics(x, u,eps=Ts):
    A=np.zeros((Nx,Nx))
    for i in range(Nx):
        x_new=np.copy(x)
        x_new[i]+=eps
        A[:,i]=(rk4_dynamics(x_new,u)-rk4_dynamics(x,u))/eps
    B=np.zeros((Nx,Nu))
    for i in range(Nu):
        u_new=np.copy(u)
        u_new[i]+=eps
        B[:,i]=(rk4_dynamics(x,u_new)-rk4_dynamics(x,u))/eps
    return A,B




Q = 150*np.diag([200, 5, 200, 5, 120, 4, 15, 100, 15, 100, 15, 100])

R = 1e-5*np.eye(4)
Nh = 300
N = 5000
x_hover=np.zeros(Nx)
u_hover=np.ones(Nu)*(m*g/4.0)/Kf

A,B=linearize_dynamics(x_hover,u_hover)
P=solve_discrete_are(A, B, Q, R)
Qf=Q


inf=np.inf
vz_min=-inf
vz_max=inf
u_min=0.75*u_hover
u_max=1.75*u_hover

x_min=np.array([
    -inf,-inf,-inf,-inf,-inf,vz_min,-2*pi,-inf,-2*pi,-inf,-2*pi,-inf
])
x_max = np.array([
    inf,inf,inf,inf,inf,vz_max,2*pi,inf,2*pi,inf,2*pi,inf
])
# Build QP matrices for OSQP

# Construct H matrix (quadratic cost)
#H = sparse.block_diag([Q] * Nh + [R] * Nh).tocsc()
H=sparse.block_diag([sparse.kron(sparse.eye(Nh-1),sparse.block_diag([R]*1+[Q]*1))]+[sparse.block_diag([R]*1+[Qf]*1)]  ).tocsc()
# Initialize b and other matrices
b = np.zeros(Nh * (Nx + Nu))

# Construct C matrix (dynamics constraints)
# First, state constraints
C_top = hstack((B,-eye(Nx),csr_matrix(np.zeros((Nx,(Nh-1)*(Nx+Nu))))))
# Then, the rest of the dynamics
C_rest1=csr_matrix(np.zeros((Nx*(Nh-1),Nu)))
C_rest2=hstack((sparse.kron(sparse.eye(Nh-1),np.hstack((A,B))),csr_matrix(np.zeros((Nx*(Nh-1),Nx)))))
C_rest3=hstack(  (csr_matrix(np.zeros((Nx*(Nh-1),Nx))),sparse.kron(sparse.eye(Nh-1), np.hstack((np.zeros((Nx,Nu)),-np.eye(Nx)  )  ))))
C = vstack((C_top,hstack((C_rest1,C_rest2+C_rest3))))


# Thrust constraints
U_matrix = sparse.kron(sparse.eye(Nh), np.hstack([np.eye(Nu), np.zeros((Nu, Nx))]))
X_matrix = sparse.kron(sparse.eye(Nh), np.hstack([np.zeros((Nx, Nu)),np.eye(Nx)]))

# Dynamics + thrust limit + theta constraints
D = vstack([
    C,
    U_matrix,
    X_matrix
]).tocsc()

# Define lower and upper bounds
lb = np.concatenate([
    np.zeros(Nx * Nh),  # Dynamics equality constraints
    np.tile(u_min-u_hover , Nh),  # Control lower limits
    np.tile(x_min, Nh)  # Theta lower limits
])

ub = np.concatenate([
    np.zeros(Nx * Nh),  # Dynamics equality constraints
    np.tile(u_max-u_hover, Nh),  # Control upper limits
    np.tile(x_max, Nh)  # Theta upper limits
])

# Setup OSQP problem
prob = osqp.OSQP()
prob.setup(P=H, q=b, A=D, l=lb, u=ub, verbose=True, eps_abs=1e-2, eps_rel=1e-2, polish=False)

def mpc_controller(t,x):
    """
    MPC controller using OSQP.
    """
    lb[:Nx] = -A@x
    ub[:Nx] = -A@x
    x_ref = np.zeros(Nx)
    x_ref[::2][:3] = trajectory((t+1) * Ts).full().flatten()
    x_ref[1::2][:3] = trajectory_dot((t+1) * Ts).full().flatten()
    # the linear guys in function J due to the nonzero x_ref
    for j in range(Nh - 1):
        #x_ref[::2][:3] = trajectory(t*Ts).full().flatten()
        #x_ref[1::2][:3] = trajectory_dot(j*Ts).full().flatten()
        b[Nu + j * (Nx + Nu):Nu + j * (Nx + Nu) + Nx] = -Q @ x_ref
    #x_ref[::2][:3] = trajectory(Ts*t).full().flatten()
    #x_ref[1::2][:3] = trajectory_dot((Nh-1)*Ts).full().flatten()
    b[Nu + (Nh - 1) * (Nx + Nu):Nu + (Nh - 1) * (Nx + Nu) + Nx] = -Qf @ x_ref

    # Update the first Nx constraints to be the current state
    prob.update(q=b, l=lb, u=ub)

    # Solve QP
    results = prob.solve()
    # Extract the first control input
    delta_u = results.x[:Nu]


    return delta_u+u_hover


def closed_loop(x0,  Nt):
    """
    Simulate closed-loop system.
    """
    xhist = np.zeros((Nx, Nt))
    uhist = np.zeros((Nu, Nt - 1))
    u0 = mpc_controller(0,x0)

    xhist[:, 0] = x0
    uhist[:, 0] = u0
    for k in range(Nt - 1):
        u= mpc_controller(k,xhist[:, k])
        # Enforce control limits
        #u = np.clip(u, u_min+u_hover, u_max+u_hover)
        uhist[:, k] = u
        # Integrate dynamics
        x_next = rk4_dynamics(xhist[:, k], u)
        xhist[:, k + 1] = x_next

    return xhist.T, uhist.T


X,U=closed_loop(X0,N)
t = np.arange(N) * Ts



x = X[:, 0]
y = X[:, 2]
z = X[:, 4]

phi = X[:, 6]
theta = X[:, 8]
psi = X[:, 10]

xdot = X[:, 1]
ydot = X[:, 3]
zdot = X[:, 5]

phidot = X[:, 7]
thetadot = X[:, 9]
psidot = X[:, 11]

xSet = np.zeros(N)
ySet = np.zeros(N)
zSet = np.zeros(N)
xSet_dot = np.zeros(N)
ySet_dot = np.zeros(N)
zSet_dot = np.zeros(N)
for i in range(N):
    xSet[i], ySet[i], zSet[i] = trajectory(t[i]).full()
    xSet_dot[i], ySet_dot[i], zSet_dot[i] = trajectory_dot(t[i]).full()

"""
Tracking Errors
"""

x_mse = np.sqrt(np.mean(np.square(xSet - x)))
y_mse = np.sqrt(np.mean(np.square(ySet - y)))
z_mse = np.sqrt(np.mean(np.square(zSet - z)))
print("x_mse = ", x_mse)
print("y_mse = ", y_mse)
print("z_mse = ", z_mse)

"""
Plot Output Variables
"""


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
    plt.legend(fontsize=18)
    plt.title('Translational coordinates', fontsize=18)

    plt.grid()

    plt.subplot(212)
    plt.plot(t, 180 * phi / pi, label='$\phi$', lw=2.5)
    plt.plot(t, 180 * theta / pi, label='$\Theta$', lw=2.5)
    plt.plot(t, 180 * psi / pi, c='#e62e00', label='$\psi$', lw=2.5)
    plt.xlabel('time in seconds', fontsize=18)
    plt.ylabel('angle in $^\circ$', fontsize=18)
    plt.legend(fontsize=18)
    plt.title('Rotational coordinates', fontsize=18)

    plt.grid()
    plt.subplots_adjust(hspace=0.4)
    plt.savefig("state1_with_no_state_limits.png")
    plt.show()

    plt.figure(figsize=(16, 9))
    plt.subplot(211)
    plt.plot(t, xdot, label='$\dot{x}$', lw=2.5)
    plt.plot(t, ydot, label='$\dot{y}$', lw=2.5)
    plt.plot(t, zdot, c='#e62e00', label='$\dot{z}$', lw=2.5)
    plt.plot(t, xSet_dot, label='$vx_{Traj}$', lw=2.5, linestyle='--', c='tab:blue')
    plt.plot(t, ySet_dot, label='$vy_{Traj}$', lw=2.5, linestyle='--', c='tab:orange')
    plt.plot(t, zSet_dot, label='$vz_{Traj}$', linestyle='--', c='tab:red', lw=2.5)
    plt.xlabel('time in seconds', fontsize=18)
    plt.ylabel('distance in $m/s$', fontsize=18)
    plt.legend(fontsize=18)
    plt.title('Translational velocities', fontsize=18)

    plt.grid()

    plt.subplot(212)
    plt.plot(t, 180 * phidot / pi, label='$\dot{\phi}$', lw=2.5)
    plt.plot(t, 180 * thetadot / pi, label='$\dot{\Theta}$', lw=2.5)
    plt.plot(t, 180 * psidot / pi, c='#e62e00', label='$\dot{\psi}$', lw=2.5)
    plt.xlabel('time in seconds', fontsize=18)
    plt.ylabel('angular rate in $^\circ/s$', fontsize=18)
    plt.legend(fontsize=18)
    plt.title('Rotational velocities', fontsize=18)

    plt.grid()
    plt.subplots_adjust(hspace=0.4)
    plt.savefig("state2_with_no_state_limits.png")
    plt.show()


def plotControlEffort():
    plt.figure(figsize=(16, 9))
    plt.subplot(221)
    plt.plot(t[:-1], U[:, 0], lw = 2.5)
    plt.xlabel('time in seconds', fontsize = 18)
    plt.ylabel('Rotor RPM', fontsize = 18)


    plt.title('Rotor - 1 RPM (Saturation RPM = 1320 rpm)', fontsize = 18)
    plt.grid()
    plt.subplot(222)
    plt.plot(t[:-1], U[:, 1], lw = 2.5)
    plt.xlabel('time in seconds', fontsize = 18)
    plt.ylabel('Rotor RPM', fontsize = 18)


    plt.title('Rotor - 2 RPM (Saturation RPM = 1320 rpm)', fontsize = 18)
    plt.grid()
    plt.subplot(223)
    plt.plot(t[:-1], U[:, 2], lw = 2.5)
    plt.xlabel('time in seconds', fontsize = 18)
    plt.ylabel('Rotor RPM', fontsize = 18)


    plt.title('Rotor - 3 RPM (Saturation RPM = 1320 rpm)', fontsize = 18)
    plt.grid()
    plt.subplot(224)
    plt.plot(t[:-1], U[:, 3], lw = 2.5)
    plt.xlabel('time in seconds', fontsize = 18)
    plt.ylabel('Rotor RPM', fontsize = 18)


    plt.title('Rotor - 4 RPM (Saturation RPM = 1320 rpm)', fontsize = 18)
    plt.grid()
    plt.subplots_adjust(hspace=0.4)
    plt.savefig("control_with_no_state_limits.png")
    plt.show()


plotOutput()
plotControlEffort()

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


ani.save('MPC_tracking_with_no_state_limits.gif', writer='pillow')
