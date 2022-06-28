#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import ode

get_ipython().run_line_magic('matplotlib', 'inline')
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['figure.figsize'] = 20,9


# In[ ]:


N_START_POWER = 1
N_END_POWER = 4
N_STEP_POWER = 1
A_START = 0.5
A_END = 0.5
A_STEPS = 100
E = 0.3
E_STEP = 0.02
TRANSIENT = int(10e10)
REPETITIONS = 4
THRESHOLD = 1000


# In[ ]:


def linz(t, xyz, a):
    x, y, z = xyz
    return [y, z, -a*z - y - np.abs(x) + 1]

def global_coupling_dynamic(eps, nodes, map_fun):
    mapped = map_fun(nodes)
    z_mean = np.mean(nodes[-1,:,2])
    return ((1 - eps) * mapped) + (eps * np.mean(mapped))

def validate_stability(start_conditions, dynamic):
    for _ in range(REPETITIONS):
        nodes = start_conditions()
        for _ in range(TRANSIENT):
            nodes = dynamic(nodes)
            if np.any(np.abs(nodes) > THRESHOLD):
                return False
    return True


# In[ ]:


xxyyzz0 = np.random.rand(2,3)


# In[ ]:


t = np.arange(0,100,0.1)
args = tuple([0.6])


# In[ ]:


def successful(ode_system):
    return np.all([ode.successful() for ode in ode_system])

def variables(ode_system):
    return np.array([ode.y for ode in ode_system])


# In[ ]:


def linz(t, xyz, a):
    x, y, z = xyz
    return [y, z, -a*z - y - np.abs(x) + 1]

system = np.array([ode(linz).set_integrator("dopri5").set_initial_value(xyz0).set_f_params(0.6) for xyz0 in xxyyzz0])
t1=100
dt=0.1
xyz = []

while successful(system) and system[0].t < t1:
    [ode.t+dt for ode in system]
    [ode.integrate(ode.t+dt) for ode in system]
    xyz.append(variables(system))


# In[ ]:


xyz = np.array(xyz)
xyz = np.swapaxes(xyz,0, 1)
xyz[:,-3:-1]


# In[ ]:


# [x for x in xyz]


# In[ ]:


fig = plt.figure()
ax = fig.add_subplot(projection='3d')
# ax.plot(xs=xxyyzz[0,:,0], ys=xxyyzz[0,:,1], zs=xxyyzz[0,:,2])
[ax.plot(xs=np.array(zs)[:,0], ys=np.array(zs)[:,1], zs=np.array(zs)[:,2]) for zs in xyz]


# In[ ]:




