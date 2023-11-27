import numpy as np
import random
import matplotlib.pyplot as plt
import scipy.integrate as integrate
import quad

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


    sines.append(lambda x : 5*np.cos(2*3.14/400*x))
    sines.append(lambda x : 10*np.cos(2*3.14/150*(x+10)))
    sines.append(lambda x : 10*np.cos(2*3.14/25*(x+5)))
    
    # f_final = lambda x : np.sum([func(x) for func in sines])
    def f_final(x):
        f_all = [func(x) for func in sines]
        res = sum(f_all)
        return res
    # print([func(0) for func in sines])
    print(f_final(0))
    return f_final

if __name__==   "__main__":
    # np.random.seed(1)
    n = 100

    # Surface test
    func = gen_surface()
    print(func(0))
    x = np.linspace(0,100,n)
    y = [func(val) for val in x]

    fig,ax = plt.subplots()
    ax.plot(x,y)


    # propogate test
    x0 = [0,-25,-.1,0,0,0]
    motor_forces = [0.49, 0.49]
    x = np.zeros([n,6])
    x[0,:] = x0
    for i in range(1,n):
        x[i,:] = quad.propogate_step(x[i-1,:],motor_forces,0.1)

    particles = np.zeros([n,10,10])
    anim = quad.animate_traj(traj=x, particles=particles,surface=func)
    # quad.plot_trajectory(ax,x)
    # ax.set_aspect('equal', 'box')

    plt.show()

