import numpy as np
import matplotlib.pyplot as plt
from dmp_lib.movement_primitives import DMP

np.random.seed(1)

T = 10.
dt = 0.01
time_steps = np.arange(0,T,dt)

x0 = 2*(np.random.rand(3)-0.5)
g0 = x0*4

N = 10

mp = DMP(num_basis=N, dt=dt, tau=T, x0=x0, g0=g0)
# mp.set_tau(mp.tau/2)

s = np.zeros((len(time_steps), 1))
x = np.zeros((len(time_steps), x0.size))
v = np.zeros((len(time_steps), x0.size))
g = np.zeros((len(time_steps), g0.size))
forcing = np.zeros((len(time_steps), x0.size))
for k, t in enumerate(time_steps):
    # goal switching    
    if k == int(len(time_steps)/3):
        mp.set_new_goal(1.5*g0, force_goal=False)
    s[k] = mp.get_phase()
    x[k,:], v[k,:] = mp.get_output()
    g[k,:] = mp.get_goal()
    forcing[k,:] = mp.evaluate_forcing(s[k])
    mp.step()

fig, axs = plt.subplots(nrows=2,ncols=1)

axs[0].plot(time_steps,x[:,0],color='tab:blue')
axs[0].plot(time_steps,x[:,1],color='tab:green')
axs[0].plot(time_steps,x[:,2],color='tab:red')
axs[0].plot(time_steps,g[:,0],'--',color='tab:blue')
axs[0].plot(time_steps,g[:,1],'--',color='tab:green')
axs[0].plot(time_steps,g[:,2],'--',color='tab:red')
axs[0].grid()
axs[0].set_title('Position')
axs[0].set(xlabel='Time [s]', ylabel=r'$\mathbf{x}$ [m]')

axs[1].plot(time_steps,v[:,0],color='tab:blue')
axs[1].plot(time_steps,v[:,1],color='tab:green')
axs[1].plot(time_steps,v[:,2],color='tab:red')
axs[1].grid()
axs[1].set_title('Velocity')
axs[1].set(xlabel='Time [s]', ylabel=r'$\dot{\mathbf{x}}$ [m/s]')

fig.tight_layout()
plt.show()

plt.figure()
plt.plot(time_steps,s)
plt.ylabel('s')
plt.xlabel('Time [s]')
plt.title('Phase')
plt.grid()
plt.show()

