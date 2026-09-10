import numpy as np
import matplotlib.pyplot as plt
import argparse

#偏微分方程为：u'' + w^2*u = 0, u(0) = I, u'(0) = 0
def solver(I, w, T, dt):
    dt = float(dt)
    Nt = int(round(T/dt))       #取整
    t = np.linspace(0, Nt*dt, Nt+1)
    u = np.zeros(Nt+1)

    u[0] = I
    u[1] = u[0] - 0.5*dt**2*w**2*u[0]
    for n in range(1, Nt):        #“标量化”中心差分
        u[n+1] = 2*u[n] - u[n-1] - dt**2*w**2*u[n]
    #该差分格式不能这样一次性计算，因为 u_{n+1} 依赖刚刚得到的 u_n。NumPy 计算右侧时，u[2:] 中的大部分元素仍然是零，所以后续结果错误。
    #“矢量化”中心差分目前看来只适用于计算速度等微分量，它们的计算不依赖依赖于前两个时间步的值，而依赖已经得到的位移(微分前)的值。
    # u[2:] = 2*u[1:-1] - u[:-2] - dt**2*w**2*u[1:-1]
    return u, t

def u_exact(t, I, w):       #精确解 
    return I*np.cos(w*t)

def visualize(u, t, I, w):
    plt.figure(figsize=(8, 5))
    t_fine = np.linspace(0, t[-1], 1001)        #fine mesh for u_e
    u_ex = u_exact(t_fine, I, w)
    plt.plot(t, u, 'r-', label='Numerical Solution')        #'-'实线
    plt.plot(t_fine, u_ex, 'b--', label='Exact Solution')       #'--'虚线
    plt.xlabel('Time (s)')
    plt.ylabel('Displacement (m)')
    plt.title('Finite Difference Method vs Exact Solution —— dt=%g' % (t[1]-t[0]))
    plt.legend(['numerical', 'exact'], loc='best')
    plt.grid()
    plt.savefig('./finite_difference.png', dpi=300)
    plt.show()

parser = argparse.ArgumentParser(description='Finite Difference Method for Simple Harmonic Oscillator')
parser.add_argument('--I', type=float, default=1.0, help='Initial displacement')
parser.add_argument('--w', type=float, default=2*np.pi, help='Angular frequency')
parser.add_argument('--dt', type=float, default=0.05, help='Time step')
parser.add_argument('--num_periods', type=int, default=5, help='Number of periods to simulate')
a =  parser.parse_args()

I = a.I
w = a.w
dt = a.dt
num_periods = a.num_periods
T = num_periods * (2*np.pi/w)       #总时间
u, t = solver(I, w, T, dt)
visualize(u, t, I, w)