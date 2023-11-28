import numpy as np
import random
import matplotlib.pyplot as plt
import scipy.integrate as integrate
import quad
import pf
import copy

Dt=0.3 # timestep for sim

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


    sines.append(lambda x : 5*np.cos(2*3.14/400*x)+10)
    sines.append(lambda x : 10*np.cos(2*3.14/150*(x+10))+10)
    sines.append(lambda x : 10*np.cos(2*3.14/25*(x+5))+10)
    
    # f_final = lambda x : np.sum([func(x) for func in sines])
    def f_final(x):
        if x>20 and x< 40:
            return 60
        if x>60 and x<70:
            return 5
        f_all = [func(x) for func in sines]
        res = sum(f_all)
        return res
    # print([func(0) for func in sines])
    print(f_final(0))
    return f_final

if __name__==   "__main__":
    np.random.seed(1)
    n_steps = 40
    num_particles = 1000

    surf_func = gen_surface()
    # propogate test
    x0 = [17,30,-.1,.2,-1,0]
    motor_forces = np.ones([n_steps,2])*.55
    x = np.zeros([n_steps,6])
    x[0,:] = x0

    #initialize particles

    X = pf.initializeParticles(x_lim=[x0[0]-10,x0[0]+80], y_lim=[x0[1]-40,x0[1]+10],surface_func=surf_func,n=num_particles)
    particle_history = np.zeros([n_steps,num_particles,2])
    particle_history[0,:,:] = X

    # loop through the motion
    for i in range(1,n_steps):
        print(f"Loop {i}")

        #get true state
        x[i,:] = quad.propogate_step(x[i-1,:],motor_forces[i-1,:],Dt)

        #run particles filter
        X = pf.runMCL_step(X, copy.copy(x[i,:]),copy.copy(motor_forces[i-1,:]),surface_func=surf_func,Dt=Dt)
        particle_history[i,:,:] = X

    

    # partics = pf.runParticleFilter(x_lim=[0,100],y_lim=[0,100],state_vec=copy.copy(x),motion_commands=motor_forces,surface_fun=surf_func,Dt=Dt)

    # partics = np.zeros([n,10,10])
    anim = quad.animate_traj(traj=x, particles=particle_history, surface=surf_func,frame_time=200)
    # quad.plot_trajectory(ax,x)
    # ax.set_aspect('equal', 'box')

    plt.show()

