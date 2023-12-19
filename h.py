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

def gen_surface(n=3, mode=0):
    if mode==0:
        sines = []
        f_cos = lambda x,a,b,c : a*np.cos(b*(x+c))
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
        # print(f_final(0))
        return f_final
    elif mode==1:
        yf = np.load("/home/user/repos/quadpf/data/lunar_contour.npy")
        yf = -yf/2
        xf = np.linspace(0,500,len(yf))

        def f_final(x):
            if x>500:
                return np.interp(math.fmod(x,500),xf,yf)
            if x<0:
                x = math.fmod(x,500)
                x = 500 + x
                return np.interp(x,xf,yf)
            return np.interp(x,xf,yf)

        return f_final


    return f_final
