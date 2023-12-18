import quad 
import numpy as np
from copy import copy

class CM:
    def __init__(self, surface_fun, x0, search_lim, match_interval=10, match_res=0.25):
        self.surface_fun = surface_fun # ground function
        self.match_interval = match_interval #number of timesteps between countour matches
        self.state_est = np.array([x0])
        self.res = match_res
        self.search_lim = search_lim

    
    contour = [] #countour of ground (dynamics accounted for)
    
    ## checks a contour (dynamics accoutne for)
    def checkMatch(self, agl):
        d = self.state_est[-self.match_interval+1:,0]
        d = d-np.min(d) # zero the x locs
        MAD = np.zeros([int(2*self.search_lim/self.res)])
        xx = np.arange(self.state_est[-1,0]-self.search_lim, self.state_est[-1,0]+self.search_lim, self.res)
        for i in range(len(xx)):
            temp_contour = self.contour[-self.match_interval+1:] - self.surface_fun(d[0])
            for j in range(len(d)):
                MAD[i] += abs(xx[i]+temp_contour[j]-self.surface_fun(xx[i]+d[j]))
        idx_min = np.argmin(MAD)

        #get best guess of location (must add the x location at last idx)
        xx_min = xx[idx_min] + (d[-1]-d[0])
        state = self.state_est[-1,:]
        state[0] = xx_min
        state[1] = self.surface_fun(xx_min) - agl
        # self.state_est[:,0] = self.state_est[:,0] + xx_min
        return state

    
    def runCM(self,k,Dt,x_prev,u_prev,meas):
        # self.state_est.append[[0,0,x[2],x[3],x[4],x[5]]]
        mu_x_prev = x_prev
        mu_x_prev[0:2] = self.state_est[-1,0:2]

        mu_x = quad.propogate_step(mu_x_prev, u_prev, Dt)
        self.contour.append(mu_x[1]+meas)

        if k%self.match_interval==0:
            mu_x = self.checkMatch(meas)


        self.state_est = np.vstack([self.state_est, mu_x])
        return np.array(mu_x), copy(self.state_est), copy(self.contour)

