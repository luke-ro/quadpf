import numpy as np
import random
import matplotlib.pyplot as plt
import scipy.integrate as integrate
import quad
import pf
import copy

Dt=0.6 # timestep for sim

def getSSE(x1, x2):
    if len(x1) != len(x2):
        Exception("getSEE(): Vectors are not the same length")

    sum = 0
    for i in range(len(x1)):
        sum += np.linalg.norm(x1[i,:]-x2[i,:])**2

    return sum

def plot1DTraj(ax,t,X):
    means = np.zeros(len(X))
    sig = np.zeros(len(X))
    for i in range(len(X)):
        means[i] = np.mean(X[i,:])
        sig[i] = np.std(X[i,:])

    ax.plot(t,means,label="Estimate")
    ax.plot(t,means+sig,'--',color="orange",label="2$\\sigma$")
    ax.plot(t,means-sig,'--',color="orange")


def motorToControls(motor_controls, l_arm):
    u = np.zeros([2,len(motor_controls)])
    for i,mc in enumerate(motor_controls):
        u[0,i] = -mc[0] - mc[1]
        u[1,i] = l_arm*(mc[1]-mc[0])
    return u

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
        if x>50 and x<70:
            return 5
        f_all = [func(x) for func in sines]
        res = sum(f_all)
        return res
    # print([func(0) for func in sines])
    print(f_final(0))
    return f_final

if __name__==   "__main__":
    np.random.seed(2)
    n_steps = 10
    num_particles = 2000

    t = np.arange(0,Dt*n_steps,Dt)

    surf_func = gen_surface()
    # propogate test
    x0 = [37,0,-.1,2,-1,0.5]
    motor_forces = np.ones([n_steps,2]) *.6
    controls = motorToControls(motor_forces,quad.ARM_LEN)
    # motor_forces[:,0] = 0.45 
    # motor_forces[:,1] = 0.55 
    # motor_forces[int(n_steps/2):,:] = .1
    x = np.zeros([n_steps,6])
    pos_est = np.zeros([n_steps,2])
    x[0,:] = x0

    #initialize particles

    X = pf.initializeParticles(x_lim=[x0[0]-20,x0[0]+80], y_lim=[x0[1]-60,x0[1]+60],surface_func=surf_func,n=num_particles)
    particle_history = np.zeros([n_steps,num_particles,2])
    particle_history[0,:,:] = X

    # loop through the motion
    for i in range(1,n_steps):
        print(f"Loop {i}")
        # fig,ax=plt.subplots()
        # ax = quad.plotInstant(ax,x[i-1,:],X,surf_func,xlim=(20,90),ylim=(-40,70))
        # ax.invert_yaxis()

        #get true state
        x[i,:] = quad.propogate_step(x[i-1,:],controls[:,i-1],Dt)
        print(f"state: {x[i,:]}")

        #run particles filter
        alt = abs(pf.getNoisyMeas(x[i,0],x[i,1],surface_fun=copy.copy(surf_func)))

        X, curr_pos_est = pf.runMCL_step(X, copy.copy(x[i-1,:]), copy.copy(controls[:,i-1]), alt, surface_func=surf_func,Dt=Dt)
        pos_est[i,:] = curr_pos_est
        # pos_est[i,1] = pf.get_estimate(X[:,1],0.25)
        particle_history[i,:,:] = X


        print(f"PF est x pos: {pos_est[i,0]}, y pos: {pos_est[i,1]}")


    print(f"SSE {getSSE(x[3:,0:2],pos_est[3:,:])}")

    fig,ax = plt.subplots(2,1)
    plot1DTraj(ax[0],t,np.squeeze(particle_history[:,:,0]))
    ax[0].plot(t,x[:,0],color='r',label="Truth")
    ax[0].legend()
    ax[0].set_ylabel("x [m]")
    ax[0].set_xlabel("t [s]")


    plot1DTraj(ax[1],t,np.squeeze(particle_history[:,:,1]))
    ax[1].plot(t,x[:,1],color='r',label="Truth")
    ax[1].invert_yaxis()
    ax[1].set_ylabel("y [m]")
    ax[1].set_xlabel("t [s]")
    # ax[1].legend()

    # ax[0].set_xlim([3,8])
    # ax[1].set_xlim([3,8])

    fig.tight_layout()

    # partics = pf.runParticleFilter(x_lim=[0,100],y_lim=[0,100],state_vec=copy.copy(x),motion_commands=motor_forces,surface_fun=surf_func,Dt=Dt)

    # partics = np.zeros([n,10,10])
    anim = quad.animate_traj(traj=x, particles=particle_history, surface=surf_func,frame_time=200)


    anim.save('pf_animation.gif',  
          writer = 'ffmpeg', fps = 4) 
    # quad.plot_trajectory(ax,x)
    # ax.set_aspect('equal', 'box')

    plt.show()

