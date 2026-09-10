import numpy as np
import matplotlib.pyplot as plt
import argparse

#一维波动方程求解器 DtDtu = c**2 * DxDxu u(x,0) = I(x), u_t(x,0) = 0
def scolver(I, c, L, T, dt, dx):
    dt = float(dt)
    dx = float(dx)
    Nt = int(round(T/dt))       #取整
    Nx = int(round(L/dx))
    t = np.linspace(0, Nt*dt, Nt+1)
    x = np.linspace(0, Nx*dx, Nx+1)
    u = np.zeros((Nt+1, Nx+1))

    u[:, 0] = u[:, -1]       #周期性边界条件 u(0,t) = u(L,t)
    u[0,:] = I(x)        #初始位移

    #-------------下面通过roll函数来滚动数组的方式计算差分，实际上对应的是周期性边界条件
    #u0的左临点会变成u100，u100的右临点会变成u0，形成一个环形的结构：u(0,t)↔u(L,t).

    #roll(u[0,:], -1)作用是取出第一行来，并向左平移一位，对第 i 个位置来说，它相当于提供：u[0, i+1]
    #-1 单独作为索引时，表示“最后一个元素”，但Python 的切片规则是：a[start:stop]其中 start 包含，stop 不包含。
    u[1, 1:-1] = u[0, 1:-1] + 0.5*(c*dt/dx)**2 * (np.roll(u[0, 1:-1], -1) - 2*u[0, 1:-1] + np.roll(u[0, 1:-1], 1))        #初始速度为零，使用中心差分计算第一步

    #对于时间方向的计算，是“标量化”的；而对于空间方向，则是“矢量化”的
    #因为空间方向不依赖正在计算的新数据，这些邻点都是旧时间层上的已知数据
    #而时间方向上要依赖于前两个时间层的值，所以不能一次性计算所有时间层，只能逐步迭代
    for n in range(1, Nt):        #“标量化”中心差分
        u[n+1, 1:-1] = 2*u[n, 1:-1] - u[n-1, 1:-1] + (c*dt/dx)**2 * (np.roll(u[n, 1:-1], -1) - 2*u[n, 1:-1] + np.roll(u[n, 1:-1], 1))
    
    return u, t, x

def u_exact(t, x, c, L):
    #将t和x扩展为二维数组，以便进行广播运算：t[:, None]表示将t扩展为列向量，x[None, :]表示将x扩展为行向量
    #对于周期性边条件，解析解应该为u(x,t) = A*cos(2*pi*c*t/L)*sin(2*pi*x/L)
    return A*np.cos(2*np.pi*c*t[:, None]/L) * np.sin(2*np.pi*x[None, :]/L)        #精确解，A为振幅，L为长度

def visualize(u, t, x, c, L):
    plt.figure(figsize=(8, 5))
    u_ex = u_exact(t, x, c, L)
    #绘制数值解和精确解的动画
    for n in range(len(t)):
        plt.clf()       #清除当前图形
        plt.plot(x, u[n,:], 'r-', label='Numerical Solution')        #'-'实线
        plt.plot(x, u_ex[n,:], 'b--', label='Exact Solution')       #'--'虚线
        plt.xlabel('Position (m)')
        plt.ylabel('Displacement (m)')
        plt.title('Finite Difference Method vs Exact Solution —— t=%.2f s' % t[n])
        plt.legend(['numerical', 'exact'], loc='best')
        plt.grid()
        plt.pause(0.01)     #在动画或连续绘图中创建暂停效果

A = 1.0        #振幅
I = lambda x: A*np.sin(2*np.pi*x/L)        #初始位移函数，A为振幅，L为长度
c = 1.0        #波速
L = 1.0        #长度
dt = 0.01        #时间步长
dx = 0.01        #空间步长
num_periods = 5        #模拟的周期数
# 下面计算总时间 T，使用波动方程的频率公式 w = πc/L，周期为 2π/w
T = num_periods * (2*np.pi/(np.pi*c/L))       #总时间
u, t, x = scolver(I, c, L, T, dt, dx)
visualize(u, t, x, c, L)