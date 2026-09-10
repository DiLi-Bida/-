#微分方程为：u'' + w^2*u = f, 设其解为u(t) = t^2 + 2t + 1, 则其中f = w^2*(t^2 + 2t + 1) + 2, u(0) = I = 1, u'(0) = V = 2
import numpy as np
import matplotlib.pyplot as plt
import argparse

def solver(I, V, w, T, dt):
    dt = float(dt)
    Nt = int(round(T/dt))       #取整
    t = np.linspace(0, Nt*dt, Nt+1)
    u = np.zeros(Nt+1)

    u[0] = I
    u[1] = u[0] + V*dt + 0.5*(w**2 + 2 - w**2*I)*dt**2
    for n in range(1, Nt):        #“标量化”中心差分
        u[n+1] = (2 - w**2*dt**2)*u[n] - u[n-1] + (w**2*(t[n]**2 + 2*t[n] + 1) + 2)*dt**2       #注意此时引入了t，要用t[n]来计算f(t)
    return u, t

def u_exact(t, I):       #精确解
    return t**2 + 2*t + 1

def visualize(u, t, I):
    plt.figure(figsize=(8, 5))
    t_fine = np.linspace(0, t[-1], 1001)        #fine mesh for u_e
    u_ex = u_exact(t_fine, I)
    plt.plot(t, u, 'r-', label='Numerical Solution')        #'-'实线
    plt.plot(t_fine, u_ex, 'b--', label='Exact Solution')       #'--'虚线
    plt.xlabel('Time (s)')
    plt.ylabel('Displacement (m)')
    plt.title('Finite Difference Method vs Exact Solution —— dt=%g' % (t[1]-t[0]))
    plt.legend(['numerical', 'exact'], loc='best')
    plt.grid()
    plt.savefig('./linear_quadratic_solution.png', dpi=300)
    plt.show()

parser = argparse.ArgumentParser(description='Finite Difference Method for Simple Harmonic Oscillator')
parser.add_argument('--I', type=float, default=1.0, help='Initial displacement')
parser.add_argument('--V', type=float, default=2.0, help='Initial velocity')
parser.add_argument('--w', type=float, default=2*np.pi, help='Angular frequency')
parser.add_argument('--dt', type=float, default=0.05, help='Time step')
parser.add_argument('--num_periods', type=int, default=5, help='Number of periods to simulate')
a =  parser.parse_args()

I = a.I
V = a.V
w = a.w
dt = a.dt
num_periods = a.num_periods
T = num_periods * (2*np.pi/w)       #总时间
u, t = solver(I, V, w, T, dt)
visualize(u, t, I)