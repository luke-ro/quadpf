import numpy as np
import random
import matplotlib.pyplot as plt
import scipy.integrate as integrate

g = 9.81

ARM_LEN = 0.15 #m
M_ARM = 0.05 #kg
M_MOTOR = 0.05 #kg
M = M_ARM + M_MOTOR

#    MOI of motors           MOI of structure     
Iy = 2*M_ARM*ARM_LEN**2 + (M_ARM*(2*ARM_LEN)**2)/12

def rotate(ang,vec):
    R = np.array([[np.cos(ang), np.sin(ang)],
         [-np.sin(ang), np.cos(ang)]])
    return R*vec

def gen_surface(n=3):
    sines = []
    f_cos = lambda x,a,b,c : a*np.cos(b*(x+c))
    # for i in range(n):
    #     # a = (np.random.rand()*10)+5
    #     a=1
    #     # b = np.random.rand()*0.2
    #     b = ((i+1)*2*3.14)/20
    #     # c = np.random.rand()*2*3.14
    #     c = 0
    #     f = lambda x : f_cos(x,a,b,c)
    #     sines.append(f)
    #     print(a,b,c,f(0))
    #     # print(sines)


    sines.append(lambda x : 5*np.cos(2*3.14/400*x))
    sines.append(lambda x : 10*np.cos(2*3.14/150*(x+10)))
    sines.append(lambda x : 10*np.cos(2*3.14/25*(x+5)))
    
    # f_final = lambda x : np.sum([func(x) for func in sines])
    def f_final(x):
        f_all = [func(x) for func in sines]
        res = sum(f_all)
        return res
    # print([func(0) for func in sines])
    print(f_final(0))
    return f_final

def dynamics(y):
    # x = x0[0] # horizontal position inertial
    # z = x0[1] # vertical position inertial
    print(y)
    th = y[2] # pitch 
    u = y[3] # horz velocity body
    w = y[4] # vert velocity body 
    q = y[5] # pitch rate
    Fx = y[6] # force in x body
    Fz = y[7] # force in z body
    pitch_moment = y[8] # moment about y 

    # derivative states
    dx = np.cos(th)*u + np.sin(th)*w 
    dz = -np.sin(th)*u + np.cos(th)*w
    dth = np.cos(th)*q
    du = -q*w + Fx/M  + g*np.sin(th)/M# Account for aerodynamics?
    dw = q*u + Fz/M + g*np.cos(th)/M# Account for aerodynamics?
    dq = pitch_moment/Iy

    return np.array([dx,dz,dth,du,dw,dq,0,0,0])

def rk4(fun,y0,dt):
    k1 = fun(y0)
    k2 = fun(y0+dt*k1/2)
    k3 = fun(y0+dt*k2/2)
    k4 = fun(y0+dt*k3)

    y1 = y0 + (dt/6) * (k1 + (2*k2) + (2*k3) + k4)
    return y1

def propogate_step(x, motor_forces, Dt):
    fx = 0
    fz = motor_forces[0] + motor_forces[1]
    moment = ARM_LEN*(motor_forces[1]-motor_forces[0])
    y0 = np.append(x,[fx,fz,moment])
    y = rk4(dynamics,y0,Dt)
    return y[0:6]

def plotquad(ax,traj):
    for p in traj:
        motor = rotate(p[5],[0.5,0]) 
        x_pts = [p[0]-motor[0], p[0]+motor[0]]
        y_pts = [p[1]-motor[1], p[1]+motor[1]]
        ax.plot(x_pts,y_pts)

def plot_trajectory(ax,traj):
    ax.plot(traj[:,0],traj[:,1])
    plotquad(ax,traj)

if __name__==   "__main__":
    # np.random.seed(1)

    # Surface test
    func = gen_surface()
    print(func(0))
    x = np.linspace(0,100,1000)
    y = [func(val) for val in x]

    fig,ax = plt.subplots()
    ax.plot(x,y)


    # propogate test
    x0 = [0,0,0,0,0,0]
    motor_forces = [.1, 0.09]
    x = np.zeros([100,6])
    x[0,:] = x0
    for i in range(1,100):
        x[i,:] = propogate_step(x[i-1,:],motor_forces,0.01)

    fig,ax = plt.subplots()
    plot_trajectory(ax,x)
    ax.set_aspect('equal', 'box')

    plt.show()

