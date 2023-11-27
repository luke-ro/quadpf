import numpy as np
import random
import matplotlib.pyplot as plt
import scipy.integrate as integrate
import quad



# def rotate(ang,vec):
#     R = np.array([[np.cos(ang), -np.sin(ang)],
#          [np.sin(ang), np.cos(ang)]])
#     return R@vec

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

# def dynamics(y):
#     # x = x0[0] # horizontal position inertial
#     # z = x0[1] # vertical position inertial
#     # print(y)
#     th = y[2] # pitch 
#     u = y[3] # horz velocity body
#     w = y[4] # vert velocity body 
#     q = y[5] # pitch rate
#     Fx = y[6] # force in x body
#     Fz = y[7] # force in z body
#     pitch_moment = y[8] # moment about y 

#     # derivative states
#     dx = np.cos(th)*u + np.sin(th)*w 
#     dz = -np.sin(th)*u + np.cos(th)*w
#     dth = np.cos(th)*q
#     du = -q*w + Fx/M  - g*np.sin(th)# Account for aerodynamics?
#     dw = q*u + Fz/M + g*np.cos(th)# Account for aerodynamics?
#     dq = pitch_moment/Iy

#     dy = np.array([dx,dz,dth,du,dw,dq,0,0,0])
#     # print(dy)
#     print(th)
#     return dy

# def rk4(fun,y0,dt):
#     k1 = fun(y0)
#     k2 = fun(y0+dt*k1/2)
#     k3 = fun(y0+dt*k2/2)
#     k4 = fun(y0+dt*k3)

#     y1 = y0 + (dt/6) * (k1 + (2*k2) + (2*k3) + k4)
#     return y1

# def propogate_step(x, motor_forces, Dt):
#     fx = 0
#     fz = -motor_forces[0] - motor_forces[1]
#     moment = ARM_LEN*(motor_forces[1]-motor_forces[0])
#     y0 = np.append(x,[fx,fz,moment])
#     y = rk4(dynamics,y0,Dt)
#     return y[0:6]

# def plotquad(ax,traj):
#     for p in traj:
#         motor = rotate(-p[2],[0.5,0]) 
#         x_pts = [p[0]-motor[0], p[0]+motor[0]]
#         y_pts = np.array([p[1]-motor[1], p[1]+motor[1]])
#         ax.plot(x_pts,-y_pts,color="steelblue")

#         vel_e = rotate(-p[2],p[3:5])
#         if(np.linalg.norm(vel_e)>0):
#             plt.arrow(p[0],-p[1],vel_e[0]*0.1,-vel_e[1]*0.1)

# def plot_trajectory(ax,traj):
#     # ax.plot(traj[:,0],traj[:,1])
#     plotquad(ax,traj)

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
    x0 = [0,0,-.1,0,0,0]
    motor_forces = [0.5, 0.5]
    x = np.zeros([100,6])
    x[0,:] = x0
    for i in range(1,100):
        x[i,:] = quad.propogate_step(x[i-1,:],motor_forces,0.1)

    fig,ax = plt.subplots()
    quad.plot_trajectory(ax,x)
    # ax.set_aspect('equal', 'box')

    plt.show()

