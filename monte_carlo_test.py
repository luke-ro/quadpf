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

Dt = 0.6

if __name__==   "__main__":
    np.random.seed(3)
    n_steps = 80
    n_runs = 20

    # time vector
    t = np.arange(0,Dt*n_steps,Dt)

    #ground function
    surf_func = h.gen_surface(mode=1)

    ## Motor forces stuff and intial state
    # x0 = [0, -100, -.1, 5, -1, 0]
    motor_forces = np.ones([n_steps,2]) *.492
    controls = h.motorToControls(motor_forces,quad.ARM_LEN)


    #initialize PF 
    num_particles = 200
    # particle_history = np.zeros([n_steps,num_particles,2])
 



    comp_times_pf =  np.zeros([n_runs,n_steps])
    comp_times_cm =  np.zeros([n_runs,n_steps])

    error_pf = np.zeros([n_runs,n_steps])
    error_cm = np.zeros([n_runs,n_steps])

    # Monte carlo outer loop
    for k in range(0,n_runs):
        print(f"MC Iteration {k}")

        # initial guess for filters:
        mu_x = np.random.uniform(0,500)
        x_rand_bounds = (-50,50)
        x_guess_bounds = (mu_x-100,mu_x+100)

        #initial start state:
        x0 = np.zeros([6])
        x0[0] = np.random.uniform(mu_x-x_rand_bounds[0],mu_x+x_rand_bounds[1])
        x0[1] =  -70
        x0[2] =  -.1
        x0[3] =    5
        x0[4] =   -1
        x0[5] =    0

        print(f"Random state: {x0}")

        # x0 = [0, -100, -.1, 5, -1, 0]
        # motor_forces = np.ones([n_steps,2]) *.492
        # controls = h.motorToControls(motor_forces,quad.ARM_LEN)

        #preallocate vector for keeping track of the true state
        x = np.zeros([n_steps,6])
        x[0,:] = x0

        ## setup new PF instance
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
                   match_interval=10,
                   contour_len=10)
        
        # preallocate vectors for storing state estimates
        pf_x_est = np.zeros([n_steps,2])
        cm_x_est = np.zeros([n_steps,2])
    

        #propogate state and filters
        for i in range(1,n_steps):
            # get true state
            x[i,:] = quad.propogate_step(x[i-1,:],controls[:,i-1],Dt)
            # print(f"state: {x[i,:]}")

            # get noisy measurement of agl
            alt_meas = abs(PF.getNoisyMeas(x[i,0],x[i,1],surface_fun=copy.copy(surf_func)))

            # run particles filter
            pf_start = time.perf_counter()
            X, pf_pos = pFilt.runMCL_step(copy.copy(x[i-1,:]), copy.copy(controls[:,i-1]), alt_meas, Dt=Dt)
            pf_stop = time.perf_counter()
            pf_x_est[i,0:2] = pf_pos[0:2]
            particle_history[i,:,:] = X

            # run cm
            cm_start = time.perf_counter()
            cm_pos,_,_ = cm.runCM(i,Dt,copy.copy(x[i-1,:]),copy.copy(controls[:,i-1]),alt_meas)
            cm_stop = time.perf_counter()
            cm_x_est[i,:] = cm_pos[0:2]

            comp_times_pf[k,i] = pf_stop-pf_start
            comp_times_cm[k,i] = cm_stop-cm_start

            error_pf[k,i] = pf_pos[0] - x[i,0]
            error_cm[k,i] = cm_pos[0] - x[i,0]

        # anim = quad.animate_traj(traj=x, 
        #                     particles=particle_history, 
        #                     cm_est=cm_x_est, 
        #                     surface=surf_func,
        #                     frame_time=50,
        #                     xlim = (-200,700))
        # plt.show()

    np.save("/home/user/repos/quadpf/mc_runs/comp_times_pf.npy",comp_times_pf)
    np.save("/home/user/repos/quadpf/mc_runs/comp_times_cm.npy",comp_times_cm)
    np.save("/home/user/repos/quadpf/mc_runs/error_pf.npy",error_pf)
    np.save("/home/user/repos/quadpf/mc_runs/error_cm.npy",error_cm)

    fig,ax = plt.subplots()
    ax.violinplot(np.stack([comp_times_pf.flatten(),comp_times_cm.flatten()]).T)
    ax.set_xticks([1,2],["PF","CM"])
    ax.set_ylabel("Time [s]")
    ax.set_title("Computation Time of Each Iteration (PF vs CM)")


    fig,ax = plt.subplots()
    ax.plot(t,np.mean(error_pf,0),label="PF",linestyle="solid")
    ax.plot(t,np.mean(error_cm,0),label="CM",linestyle="solid")
    ax.set_title("Mean error from true")
    ax.legend()

    plt.show()

        # print(f"({pf_stop-pf_start}) PF est x pos: {pf_pos_est[i,0]}, y pos: {pf_pos_est[i,1]}")
        # print(f"({cm_stop-cm_start}) CM est x pos: {cm_pos_est[0]}, y pos: {cm_pos_est[1]}")


    # print(f"SSE {h.getSSE(x[3:,0:2],pf_pos_est[3:,:])}")


    