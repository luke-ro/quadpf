import numpy as np
import matplotlib.pyplot as plt

g = 9.81

ARM_LEN = 0.15 #m
M_ARM = 0.05 #kg
M_MOTOR = 0.05 #kg
M = M_ARM + M_MOTOR

#    MOI of motors           MOI of structure     
Iy = 2*M_ARM*ARM_LEN**2 + (M_ARM*(2*ARM_LEN)**2)/12
def rotate(ang,vec):
    R = np.array([[np.cos(ang), -np.sin(ang)],
         [np.sin(ang), np.cos(ang)]])
    return R@vec

def dynamics(y):
    # x = x0[0] # horizontal position inertial
    # z = x0[1] # vertical position inertial
    # print(y)
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
    du = -q*w + Fx/M  - g*np.sin(th)# Account for aerodynamics?
    dw = q*u + Fz/M + g*np.cos(th)# Account for aerodynamics?
    dq = pitch_moment/Iy

    dy = np.array([dx,dz,dth,du,dw,dq,0,0,0])
    # print(dy)
    print(th)
    return dy

def rk4(fun,y0,dt):
    k1 = fun(y0)
    k2 = fun(y0+dt*k1/2)
    k3 = fun(y0+dt*k2/2)
    k4 = fun(y0+dt*k3)

    y1 = y0 + (dt/6) * (k1 + (2*k2) + (2*k3) + k4)
    return y1

def propogate_step(x, motor_forces, Dt):
    fx = 0
    fz = -motor_forces[0] - motor_forces[1]
    moment = ARM_LEN*(motor_forces[1]-motor_forces[0])
    y0 = np.append(x,[fx,fz,moment])
    y = rk4(dynamics,y0,Dt)
    return y[0:6]

def plotquad(ax,traj):
    for p in traj:
        motor = rotate(-p[2],[0.5,0]) 
        x_pts = [p[0]-motor[0], p[0]+motor[0]]
        y_pts = np.array([p[1]-motor[1], p[1]+motor[1]])
        ax.plot(x_pts,-y_pts,color="steelblue")

        vel_e = rotate(-p[2],p[3:5])
        if(np.linalg.norm(vel_e)>0):
            plt.arrow(p[0],-p[1],vel_e[0]*0.1,-vel_e[1]*0.1)

def plot_trajectory(ax,traj):
    # ax.plot(traj[:,0],traj[:,1])
    plotquad(ax,traj)