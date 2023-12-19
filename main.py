import numpy as np
import random
import matplotlib.pyplot as plt
import scipy.integrate as integrate
import copy
import time
import math

import quad
import PF
import CM
import h

Dt=0.6 # timestep for sim


if __name__==   "__main__":
    np.random.seed(3)
    n_steps = 80

    # time vector
    t = np.arange(0,Dt*n_steps,Dt)

    #ground function
    surf_func = h.gen_surface(mode=1)

    ## Motor forces stuff and intial state
    # x0 = [-116, -10, -.25, -5, 0, 0]
    x0 = [0, -100, -.1, 5, -1, 0]
    motor_forces = np.ones([n_steps,2]) *.492
    controls = h.motorToControls(motor_forces,quad.ARM_LEN)
    # motor_forces[:,0] = 0.45 
    # motor_forces[:,1] = 0.55 
    # motor_forces[int(n_steps/2):,:] = .1

    # initial guess for filters:
    x_guess_bounds = (-100,100)

    #initialize PF 
    num_particles = 200
    pFilt = PF.ParticleFilter(surface_func=surf_func,
                      x_lim=x_guess_bounds, 
                      n_partics=num_particles,
                      MPF=False)
    particle_history = np.zeros([n_steps,num_particles,2])
    particle_history[0,:,:] = pFilt.getParticles()

    # initialize contour matching
    x0_guess = np.zeros([6])
    x0_guess[0] = np.mean(x_guess_bounds)
    cm = CM.CM(surface_fun=surf_func,
               x_est_bounds=x_guess_bounds,
               match_interval=15)
    
    # preallocate vectors for storing state estimates
    pf_pos_est = np.zeros([n_steps,2])
    cm_pos_est = np.zeros([n_steps,2])
    cm_pos_est[0:2] = x0_guess[0:2]
    #preallocate vector for keeping track of the true state
    x = np.zeros([n_steps,6])
    x[0,:] = x0

    countour = [] # list for plotting the contour 

    # loop through the motion
    for i in range(1,n_steps):
        print(f"Loop {i}")

        # get true state
        x[i,:] = quad.propogate_step(x[i-1,:],controls[:,i-1],Dt)
        print(f"state: {x[i,:]}")

        # get noisy measurement of agl
        alt_meas = abs(PF.getNoisyMeas(x[i,0],x[i,1],surface_fun=copy.copy(surf_func)))

        # run cm
        cm_start = time.perf_counter()
        cm_state_est, x_cm, countour = cm.runCM(i,Dt,copy.copy(x[i-1,:]),copy.copy(controls[:,i-1]),alt_meas)
        cm_stop = time.perf_counter()
        cm_pos_est[i,:] = cm_state_est[0:2]

        # run particles filter
        pf_start = time.perf_counter()
        X, curr_pos_est = pFilt.runMCL_step(copy.copy(x[i-1,:]), copy.copy(controls[:,i-1]), alt_meas, Dt=Dt)
        pf_stop = time.perf_counter()
        pf_pos_est[i,:] = curr_pos_est
        particle_history[i,:,:] = X


        print(f"({pf_stop-pf_start}) PF est x pos: {pf_pos_est[i,0]}, y pos: {pf_pos_est[i,1]}")
        print(f"({cm_stop-cm_start}) CM est x pos: {cm_pos_est[0]}, y pos: {cm_pos_est[1]}")


    print(f"SSE {h.getSSE(x[3:,0:2],pf_pos_est[3:,:])}")


    ## contour from CM plot
    fig,ax = plt.subplots()
    ax.plot(x_cm[1:,0], countour)
    ax.set_title("countour")
    ax.invert_yaxis()

    fig,ax = plt.subplots(2,1)
    h.plot1DTraj(ax[0],t,np.squeeze(particle_history[:,:,0]))
    ax[0].plot(t,cm_pos_est[:,0],color='r',label="CM")
    ax[0].plot(t,x[:,0],color='g',label="Truth")
    ax[0].legend()
    ax[0].set_ylabel("x [m]")
    ax[0].set_xlabel("t [s]")


    h.plot1DTraj(ax[1],t,np.squeeze(particle_history[:,:,1]))
    ax[1].plot(t,cm_pos_est[:,1],color='r',label="CM")
    ax[1].plot(t,x[:,1],color='g',label="Truth")
    ax[1].invert_yaxis()
    ax[1].set_ylabel("y [m]")
    ax[1].set_xlabel("t [s]")
    # ax[1].legend()

    # ax[0].set_xlim([3,8])
    # ax[1].set_xlim([3,8])

    fig.tight_layout()

    # partics = pf.runParticleFilter(x_lim=[0,100],y_lim=[0,100],state_vec=copy.copy(x),motion_commands=motor_forces,surface_fun=surf_func,Dt=Dt)

    # partics = np.zeros([n,10,10])
    anim = quad.animate_traj(traj=x, 
                             particles=particle_history, 
                             cm_est=cm_pos_est, 
                             surface=surf_func,
                             frame_time=200,
                             xlim = (-200,700))


    anim.save('pf_animation.gif',  
          writer = 'ffmpeg', fps = 4) 
    # quad.plot_trajectory(ax,x)
    # ax.set_aspect('equal', 'box')

    plt.show()

