import numpy as np
import scipy.stats as stats
import quad

ALT_NOISE_STD = 3 #std dev of altimeter measurements

def getNoisyMeas(z,surface_fun):
    return z-surface_fun(z) + np.random.normal(0,ALT_NOISE_STD)

def initializeParticles(x_lim,y_lim,surface_func,n=100):
    #initialize particles
    partics = np.zeros([n,2])
    partics[:,0] = np.random.uniform(low=x_lim[0],high=x_lim[1],size=n)
    partics[:,1] = np.random.uniform(low=y_lim[0],high=y_lim[1],size=n)
    for i in range(len(partics)):
        j = 0
        while(partics[i,1] < surface_func(partics[i,0])):
            partics[i,0] = np.random.uniform(low=x_lim[0],high=x_lim[1],size=1)
            partics[i,1] = np.random.uniform(low=y_lim[0],high=y_lim[1],size=1)
            if(j>30):
                print("unable to find valid sample in particle filter,this is bad")
                break
            j+=1
    return partics

def runMCL_step(particles,y,motor_command,surface_func,Dt):
    n = len(particles)
    alt_meas = getNoisyMeas(y[1],surface_fun=surface_func)
    for i in range(len(particles)):
        #given at location specified by particle, what are the odds you'd be there?

        #assuming that everything except position is given,
        temp_state = y
        temp_state[0:2] = particles[i,:] # <- fill vector with given states (except for pos)

        # propogate motion using particle's pos, and given state for vel and angular position
        particles[i,:] = quad.propogate_step(temp_state, motor_command, Dt)[0:2] 
        # add noise?

        diff = abs(alt_meas-surface_func(particles[i,0]))
        probs[i] = 2*stats.norm.cdf(-diff,scale=ALT_NOISE_STD) # two tailed distrobution probabilty of getting a worse or same measurement

    # normalize probabilities to one
    probs = probs/np.sum(probs)
    
    ## Resample

    #iterate through particles and select them probabilistically according to their prob
    i = 0
    new_particles = np.zeros([n,2])
    while i < n:
        idx = np.random.choice(range(n)) #randomly select a particle
        if(np.random.rand() < probs[idx]): # if np.random.rand() is less than the probability of that particle, add it
            new_particles[i,:] = particles[idx,:]
            i += 1

    return new_particles

def runParticleFilter(x_lim,y_lim,state_vec,motion_commands,surface_fun,Dt,n=100):
    all_particles = np.zeros([len(state_vec),n,2])

    #initialize particles
    partics = np.zeros([n,2])
    partics[:,0] = np.random.uniform(low=x_lim[0],high=x_lim[1],size=n)
    partics[:,1] = np.random.uniform(low=y_lim[0],high=y_lim[1],size=n)
    for i in range(len(partics)):
        j = 0
        while(partics[i,1] < surface_fun(partics[i,0])):
            partics[i,0] = np.random.uniform(low=x_lim[0],high=x_lim[1],size=1)
            partics[i,1] = np.random.uniform(low=y_lim[0],high=y_lim[1],size=1)
            if(j>30):
                print("unable to find valid sample in particle filter,this is bad")
                break
            j+=1
    
    probs = np.ones([n])/n

    for j in range(1,len(state_vec)):
        y = state_vec[j,:]

        alt_meas = getNoisyMeas(y[1],surface_fun=surface_fun)
        for i in range(len(partics)):
            #given at location specified by particle, what are the odds you'd be there?

            #assuming that everything except position is given,
            temp_state = y
            temp_state[0:2] = partics[i,:] # <- fill vector with given states (except for pos)

            # propogate motion using particle's pos, and given state for vel and angular position
            partics[i,:] = quad.propogate_step(temp_state, motion_commands[j,:], Dt)[0:2] 
            # add noise?

            diff = abs(alt_meas-surface_fun(partics[i,0]))
            probs[i] = 2*stats.norm.cdf(-diff,scale=ALT_NOISE_STD) # two tailed distrobution probabilty of getting a worse or same measurement

        # normalize probabilities to one
        probs = probs/np.sum(probs)
        
        ## Resample

        #iterate through particles and select them probabilistically according to their prob
        i = 0
        new_particles = np.zeros([n,2])
        while i < n:
            idx = np.random.choice(range(n)) #randomly select a particle
            if(np.random.rand() < probs[idx]): # if np.random.rand() is less than the probability of that particle, add it
                new_particles[i,:] = partics[idx,:]
                i += 1

        all_particles[j,:,:] = partics
        partics = new_particles

    return all_particles


