import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation 

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
    # print(th)
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

def plotquad(ax,state):
    motor = rotate(-state[2],[0.5,0]) 
    x_pts = [state[0]-motor[0], state[0]+motor[0]]
    y_pts = np.array([state[1]-motor[1], state[1]+motor[1]])
    return ax.plot(x_pts,-y_pts,color="steelblue")

def plotquad_data(state):
    motor = rotate(-state[2],[0.5,0]) 
    x_pts = [state[0]-motor[0], state[0]+motor[0]]
    y_pts = np.array([state[1]-motor[1], state[1]+motor[1]])
    return x_pts,y_pts

def plotquad_traj(ax,traj):
    for p in traj:
        plotquad(ax,p)

        vel_e = rotate(-p[2],p[3:5])
        if(np.linalg.norm(vel_e)>0):
            plt.arrow(p[0],-p[1],vel_e[0]*0.1,-vel_e[1]*0.1)

def plot_trajectory(ax,traj):
    # ax.plot(traj[:,0],traj[:,1])
    plotquad_traj(ax,traj)

def animate_traj(traj,particles,surface=None,buffer=10,frame_time=30):
    # https://www.geeksforgeeks.org/using-matplotlib-for-animations/
    fig,ax = plt.subplots()
    x_min = np.min(traj[:,0])-buffer
    x_max = np.max(traj[:,0])+buffer
    z_min = np.min(traj[:,1])-buffer
    z_max = np.max(traj[:,1])+buffer

    scat = ax.scatter(particles[0,:,0],particles[0,:,1],label="Particles",s=1)
    x_quad,y_quad = plotquad_data(traj[0,:])
    line2 = ax.plot(x_quad,y_quad,label="Quadcopter")[0]

    if surface is not None:
        x_sur = np.linspace(x_min,x_max,len(traj))
        y_sur = [surface(val) for val in x_sur]
        ax.plot(x_sur,y_sur,label="Ground",color="g")
        if np.max(y_sur)+buffer>z_max:
            z_max=np.max(y_sur) + buffer
        ax.fill_between(x_sur,y_sur,np.ones(len(traj))*z_max,color="g",alpha = 0.5)

    ax.set(xlim=[x_min,x_max],ylim=[z_min,z_max],xlabel="x [m]",ylabel="y [m]")
    ax.invert_yaxis()
    ax.legend()

    def update(i):
        par_x = np.squeeze(particles[i,:,0])
        par_z = np.squeeze(particles[i,:,1])
        data = np.stack([par_x,par_z]).T
        scat.set_offsets(data)

        x_quad,y_quad = plotquad_data(traj[i,:])
        line2.set_xdata(x_quad)
        line2.set_ydata(y_quad)

        return(scat,line2)
    
    ani = FuncAnimation(fig=fig, func=update, frames = len(traj),interval=frame_time)
    return ani
    