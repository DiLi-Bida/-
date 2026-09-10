#显式方法解1D扩散方程 u_t = a*u_xx, u(x,0) = I(x), u(0,t) = u(L,t) = 0
import numpy as np
import matplotlib.pyplot as plt

def solver(I, a, L, T, F):
    dx = 0.1  # 空间步长
    dt = F*dx**2/a  # 时间步长
    Nt = int(round(T/dt))  # 时间步数
    Nx = int(round(L/dx))  # 空间步数
    t = np.linspace(0, Nt*dt, Nt+1)  # 时间数组
    x = np.linspace(0, Nx*dx, Nx+1)  # 空间数组
    u = np.zeros((Nt+1, Nx+1))  # 初始化解数组
    F = a*dt/dx**2  # 稳定性条件参数
    
    u[:, 0] = 0  # 边界条件 u(0,t) = 0
    u[:, -1] = 0  # 边界条件 u(L,t) = 0
    u[0, :] = I(x)  # 初始条件 u(x,0) = I(x)

    # 显式方法迭代计算
    for n in range(0, Nt):
        u[n+1, 1:-1] = u[n, 1:-1] +\
            F * (u[n, 2:] - 2*u[n, 1:-1] + u[n, :-2])
       
    return u, t, x 

def u_exact(t, x, a, L):
    # 精确解函数
    return np.exp(-np.pi**2*a*t[:, None]/L**2) * np.sin(np.pi*x[None, :]/L)
    # return 5*t*x*(L - x)

def visualize(u, t, x, a, L):
    plt.figure(figsize=(8, 5))
    u_ex = u_exact(t, x, a, L)
    for n in range(len(t)):
        plt.clf()
        plt.plot(x, u[n,:], 'r-', label='Numerical Solution')
        plt.plot(x, u_ex[n,:], 'b--', label='Exact Solution')
        plt.xlabel('Position (m)')
        plt.ylabel('Temperature (T)')
        plt.title('Explicit Method vs Exact Solution —— t=%.2f s' % t[n])
        plt.ylim(0, 1.1)
        plt.legend(['numerical', 'exact'], loc='best')
        plt.grid()
        plt.pause(0.001)
        # plt.show()

F = 0.5  # 稳定性条件参数(显示方法要求 F <= 0.5)
a = 1.0  # 扩散系数
L = 1.0  # 空间长度
T = 0.5  # 总时间
I = lambda x: np.sin(np.pi*x/L)  # 初始条件函数
u, t, x = solver(I, a, L, T, F)
visualize(u, t, x, a, L)