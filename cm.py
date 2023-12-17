
class CM:
    def __init__(self, surface_fun,match_interval=10):
        self.surface_fun = surface_fun # ground function
        self.match_interval = match_interval #number of timesteps between countour matches

    
    contour = [] #countour of ground (dynamics accounted for)

    
    def runCM(self,x,u,meas):
        pass
