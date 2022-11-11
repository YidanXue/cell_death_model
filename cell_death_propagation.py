# Cell death model with damage propagation - a 2D example
# Yidan Xue, Nov 2022

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import matplotlib.animation as animation
import os

os.chdir('./BCI')

# generating a test grid
nx = 50
ny = 50
rel_perf = np.ones([nx,ny])

# create a region of ischaemia
rel_perf[10:40,10:40] = 0.3
rel_perf[15:35,15:35] = 0.2
rel_perf[20:30,20:30] = 0.1
rel_perf[22:28,22:28] = 0

# initialisation
infarct = np.zeros([nx,ny])
toxin = np.zeros([nx,ny])

# define the ODE for cell death, h[0] - dead, h[1] - toxic, h[2] - relative perfusion, h[3] - vulnerable index
def cell_death(x,t):
    D,T,rCBF = x
    A = 1-D

    if T>Td:
        dDdt = kf*A*T
    else:
        dDdt = 0
    dTdt = kt*(1-rCBF)*(1-T)-kc*rCBF*A*T

    dxdt = [dDdt, dTdt, 0]
    return dxdt

# parameter value
Td,kf,kt,kc = 0.09163922, 0.00060644, 0.00429805, 0.0999227
Co = 0.25   # this parameter has not been fitted, but Co<=0.25

time = 360 # min
dt = 5 # min
t = np.linspace(0,dt*60,int(dt*12+1))
num_iter = int(time/dt)

# for making videos
# ims_toxin = []
# ims_infarct = []
# fig, ax = plt.subplots()

for n in range(num_iter):
    print(str(n+1)+'/'+str(num_iter))
    # define an array to update vulnerable index
    toxin_new = np.zeros([nx,ny])

    if n == 24:
        rel_perf = rel_perf+0.8*(1-rel_perf)

    # run cell death model first
    for i in range(nx):
        for j in range(ny):
            if rel_perf[i,j] < 1:
                hi = [infarct[i,j],toxin[i,j],rel_perf[i,j]]
                hs = odeint(cell_death, hi, t)
                infarct[i,j] = hs[-1,0]
                toxin[i,j] = hs[-1,1]

    # run the diffusion step using the Euler method
    for i in range(1,nx-1):
        for j in range(1,ny-1):
            toxin_new[i,j] = toxin[i,j]+Co*(toxin[i+1,j]+toxin[i-1,j]+toxin[i,j+1]+toxin[i,j-1]-4*toxin[i,j])

    toxin = np.copy(toxin_new)

    # im = ax.imshow(toxin, cmap=plt.cm.viridis, vmin=0, vmax=1, animated=True)
    # title = plt.text(0.5,1.1,'toxin, t=%i min' %(n*dt),color='white')
    # ims_toxin.append([im,title])

    # im = ax.imshow(infarct, cmap=plt.cm.viridis, vmin=0, vmax=1, animated=True)
    # title = plt.text(0.5,1.1,'infarct, t=%i min' %(n*dt),color='white')
    # ims_infarct.append([im,title])

# writervideo = animation.PillowWriter(fps=10) 

# ani_toxin = animation.ArtistAnimation(fig, ims_toxin, interval=100, blit=True,
#                                 repeat_delay=1000)
# ani_toxin.save("toxin_treatment.gif", writer=writervideo)

# ani_infarct = animation.ArtistAnimation(fig, ims_infarct, interval=100, blit=True,
#                                 repeat_delay=1000)
# ani_infarct.save("infarct_treatment.gif", writer=writervideo)