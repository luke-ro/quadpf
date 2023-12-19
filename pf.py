import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import quad
import copy

ALT_NOISE_STD = 1 #std dev of altimeter measurements

def get_estimate(x,step):
    x_min = np.min(x)
    x_max = np.max(x)
    # y_min = np.min(X[:,1])
    # y_max = np.max(X[:,1])

    if x_max-x_min<step:
        return (x_min+x_max)/2


    bins = np.arange(x_min,x_max+step,step)
    counts = np.histogram(x,bins)[0]
    idx = np.argmax(counts)

    return bins[idx] + (step/2) 

def getNoisyMeas(x,z,surface_fun):
    return abs(z-surface_fun(x) + np.random.normal(0,ALT_NOISE_STD))
    # return abs(z-surface_fun(x))

class ParticleFilter:
    def __init__(self, surface_func, x_lim, y_lim=None, n_partics=100, MPF=False, oneD=True):
        self.surface_func = surface_func
        self.x_lim = x_lim
        self.y_lim = y_lim
        self.n = n_partics
        self.oneD = oneD
        X = self.initializeParticles()

    def getParticles(self):
        return copy.copy(self.X)


    def initializeParticles(self):
        #initialize particles
        if self.y_lim is not None:
        
            partics = np.zeros([self.n,2])
            partics[:,0] = np.random.uniform(low=self.x_lim[0], high=self.x_lim[1], size=n)
            partics[:,1] = np.random.uniform(low=self.y_lim[0], high=self.y_lim[1], size=n)
            for i in range(len(partics)):
                j = 0
                while(partics[i,1] > self.surface_func(partics[i,0])):
                    partics[i,0] = np.random.uniform(low=self.x_lim[0],high=self.x_lim[1],size=1)
                    partics[i,1] = np.random.uniform(low=self.y_lim[0],high=self.y_lim[1],size=1)
                    if(j>30):
                        print("unable to find valid sample in particle filter,this is bad")
                        break
                    j+=1
        else:
            partics = np.zeros([self.n,2])
            partics[:,0] = np.random.uniform(low=self.x_lim[0], high=self.x_lim[1], size=self.n)
        
        self.X = partics
        return partics


    def runMCL_step(self, x_prev, control, meas, Dt):
        n = len(self.X)
        alt_meas = meas
        probs = np.zeros([n])
        for i in range(len(self.X)):
            #given at location specified by particle, what are the odds you'd be there?

            #assuming that everything except position is given,
            temp_state = copy.copy(x_prev)

            if not self.oneD:
                temp_state[0:2] = self.X[i,:] # <- fill vector with given states (except for pos)
            else:
                temp_state[0:1] = self.X[i,0] # <- fill vector with given states (except for pos)


            # propogate motion using particle's pos, and given state for vel and angular position
            self.X[i,:] = quad.propogate_step(temp_state, control, Dt)[0:2] 
            self.X[i,:] += np.random.normal(0,0.1,size=[2])
            # add noise?

            diff = abs(alt_meas-abs(self.X[i,1]-self.surface_func(self.X[i,0])))
            probs[i] = stats.norm.cdf(-diff,scale=ALT_NOISE_STD) # two tailed distrobution probabilty of getting a worse or same measurement

        # normalize probabilities to one
        probs = probs/np.sum(probs)

        #get a cdf of current X
        probs_cdf=np.zeros(n)
        probs_cdf[0] = probs[0]
        for i in range(1,n):
            probs_cdf[i] = probs_cdf[i-1]+probs[i]
        
        # Resample using a uniform dist and the cdf
        rands = np.random.rand(n)
        new_particles = np.array([self.X[np.argmax(probs_cdf>rands[i])] for i in range(n)])

        x_est = get_estimate(new_particles[:,0],0.25)
        y_est = get_estimate(new_particles[:,1],0.25)
        
        self.X = new_particles

        return new_particles, (x_est, y_est)




