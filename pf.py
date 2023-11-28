import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import quad
import copy

ALT_NOISE_STD = 1 #std dev of altimeter measurements

def getNoisyMeas(x,z,surface_fun):
    return abs(z-surface_fun(x) + np.random.normal(0,ALT_NOISE_STD))
    # return abs(z-surface_fun(x))

def initializeParticles(x_lim,y_lim,surface_func,n=100):
    #initialize particles
    partics = np.zeros([n,2])
    partics[:,0] = np.random.uniform(low=x_lim[0], high=x_lim[1], size=n)
    partics[:,1] = np.random.uniform(low=y_lim[0], high=y_lim[1], size=n)
    for i in range(len(partics)):
        j = 0
        while(partics[i,1] > surface_func(partics[i,0])):
            partics[i,0] = np.random.uniform(low=x_lim[0],high=x_lim[1],size=1)
            partics[i,1] = np.random.uniform(low=y_lim[0],high=y_lim[1],size=1)
            if(j>30):
                print("unable to find valid sample in particle filter,this is bad")
                break
            j+=1
    return partics

def runMCL_step(particles,meas,y,motor_command,surface_func,Dt):
    n = len(particles)
    alt_meas = meas
    probs = np.zeros([n])
    for i in range(len(particles)):
        #given at location specified by particle, what are the odds you'd be there?

        #assuming that everything except position is given,
        temp_state = copy.copy(y)
        temp_state[0:2] = particles[i,:] # <- fill vector with given states (except for pos)

        # propogate motion using particle's pos, and given state for vel and angular position
        particles[i,:] = quad.propogate_step(temp_state, motor_command, Dt)[0:2] 
        particles[i,:] += np.random.normal(0,0.1,size=[2])
        # add noise?

        diff = abs(alt_meas-abs(particles[i,1]-surface_func(particles[i,0])))
        probs[i] = stats.norm.cdf(-diff,scale=ALT_NOISE_STD) # two tailed distrobution probabilty of getting a worse or same measurement

    # normalize probabilities to one
    probs = probs/np.sum(probs)

    #get a cdf of current particles
    probs_cdf=np.zeros(n)
    probs_cdf[0] = probs[0]
    for i in range(1,n):
        probs_cdf[i] = probs_cdf[i-1]+probs[i]
    
    # Resample using a uniform dist and the cdf
    rands = np.random.rand(n)
    new_particles = np.array([particles[np.argmax(probs_cdf>rands[i])] for i in range(n)])

    #iterate through particles and select them probabilistically according to their prob
    # i = 0
    # new_particles = np.zeros([n,2])
    # while i < n:
    #     idx = np.random.choice(range(n)) #randomly select a particle
    #     if(np.random.rand() < probs[idx]): # if np.random.rand() is less than the probability of that particle, add it
    #         new_particles[i,:] = particles[idx,:]
    #         i += 1



    return new_particles

def get_estimate(x,step):
    x_min = np.min(x)
    x_max = np.max(x)
    # y_min = np.min(X[:,1])
    # y_max = np.max(X[:,1])


    bins = np.arange(x_min,x_max+step,step)
    counts = np.histogram(x,bins)[0]
    idx = np.argmax(counts)

    return bins[idx] + (step/2) 



