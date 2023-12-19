import quad 
import numpy as np
from copy import copy

class CM:
    def __init__(self, surface_fun, x_est_bounds, match_interval=10, match_res=0.25):
        self.surface_fun = surface_fun # ground function
        self.match_interval = match_interval #number of timesteps between countour matches
        self.x_est = [np.mean(x_est_bounds)]
        self.res = match_res
        self.search_lim = abs(x_est_bounds[1]-x_est_bounds[0])/2

    
    contour = [] #countour of ground (dynamics accounted for)
    
    ## checks a contour (dynamics accoutne for)
    def checkMatch(self, agl):
        d = self.x_est[-self.match_interval+1:]
        offset = np.min(d)
        d = d - offset# zero the x locs
        xx = np.arange(self.x_est[-1]-self.search_lim, self.x_est[-1]+self.search_lim, self.res)
        MAD = np.zeros([len(xx)])
        for i in range(len(xx)):
            temp_contour = self.contour[-self.match_interval+1:] - self.contour[-self.match_interval+1] + self.surface_fun(xx[i]+d[0])
            # temp_contour = self.contour[-self.match_interval+1:] - self.surface_fun(d[0])
            for j in range(len(d)):
                MAD[i] += abs(temp_contour[j]-self.surface_fun(xx[i]+d[j]))
        idx_min = np.argmin(MAD)

        #get best guess of location (must add the x location at last idx)
        xx_min = xx[idx_min] + d[-1]
        # state = (self.x_est[-1], self.surface_fun(xx_min) - agl)
        return xx_min

    
    def runCM(self,k,Dt,x_prev,u_prev,meas):
        # self.state_est.append[[0,0,x[2],x[3],x[4],x[5]]]
        mu_x_prev = x_prev
        mu_x_prev[0] = self.x_est[-1]

        mu_x = quad.propogate_step(mu_x_prev, u_prev, Dt)
        self.contour.append(mu_x[1]+meas)

        if k%self.match_interval==0:
            mu_x[0] = self.checkMatch(meas)


        self.x_est = np.vstack([self.x_est, mu_x[0]])
        return np.array(mu_x), copy(self.x_est), copy(self.contour)

