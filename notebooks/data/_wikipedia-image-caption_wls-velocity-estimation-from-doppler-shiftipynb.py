#!/usr/bin/env python
# coding: utf-8

# This notebook computes the weighted-least-squares (WLS) velocity from  `PseudorangeRateMetersPerSecond` data at each epoch. This is an additional piece of information using Doppler shift, not just a difference in positions computed from time (pseudo ranges). The "epoch" is one timestep and each epoch is treated independently in this notebook; connecting multiple timesteps, e.g Kalman Filter, is a next step, beyond the scope of this notebook.
# 
# Warning:
# I am new to GPS and there is still some disagreement in the WLS positions compared to those provided as a baseline. Let me know in the comment for anything missing or misunderstood.

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import glob
import scipy.optimize
from tqdm.auto import tqdm

c = 299_792_458  # speed of light in vaccum [m/s]
omega = 7.292115e-5  # angular velocity [rad/s] in ECEF coordinate WGS 84


# In[ ]:


path = '/kaggle/input/smartphone-decimeter-2022/train/2020-05-15-US-MTV-1/GooglePixel4XL'
gnss = pd.read_csv('%s/device_gnss.csv' % path, dtype={'SignalType': str})
truth = pd.read_csv('%s/ground_truth.csv' % path)


# ## 1. Position estimation
# 
# Position estimation is necessary before velocity estimation.
# 
# See also:
# - https://www.kaggle.com/code/junkoda/deriving-baseline-wls-positions-in-progress
# - https://www.kaggle.com/code/hyperc/gsdc-reproducing-baseline-wls-on-one-measurement
# - https://www.kaggle.com/code/foreveryoung/least-squares-solution-from-gnss-derived-data
# 
# A short summary is:
# 
# - $\vec{x}_i$: satellite positions (known);
# - $\vec{y}$: receiver position;
# - $r_i = \lVert \vec{x}_i - \vec{y} \rVert$: distance to satellites;
# - $\rho_i = r_i + b$: pseudo range = distance including unknown receiver clock bias $b$;
# 
# Weighted Least Square find $(\vec{y}, b)$ that minimizes:
# 
# 
# $$ \sum_i \left( \frac{\lVert \vec{x}_i - \vec{y} \rVert - \rho_i + b}{\Delta \rho} \right)^2 $$
# 
# where the weight is the inverse of the error in the pseudo ranges $\Delta \rho$.
# 
# 

# ## 2. Velocity estimation
# 
# Velocites projected to the direction of satellites (line-of-sight velocities) are provided as,
# 
# v_los = `PseudorangeRateMetersPerSecond`,
# 
# which is proportional to the measured Dopper shift in frequency converted to m/s.
# 
# We find receiver velocity $\vec{v}$ that matches these line-of-sight velocities:
# 
# $$ v_{\mathrm{los}, i} = (\vec{v}_{\mathrm{sat},i} - \vec{v})\cdot \hat{\mathbf{r}}_i $$
# 
# where $\hat{\mathbf{r}}_i$ is the unit vector poiting sattelite $i$ from the receiver:
# 
# - $\vec{r}_i = \vec{x}_i - \vec{y} $,
# - $\hat{\mathbf{r}}_i = \vec{r}_i \,/\, \lVert \vec{r}_i \rVert $
# 
# Now, weighted least squares for velocity minimizes:
# 
# $$ \sum_i \left[ \frac{(\vec{v}_{\mathrm{sat},i} - \vec{v})\cdot \hat{\mathbf{r}}_i - v_{\mathrm{los},i} + v_b}{\Delta v_{\mathrm{los},i}} \right]^2$$

# ## On the revolutions of the heavenly spheres
# 
# The satellite velocities are provided in Earth-centered Earth-fixed (ECEF) frame, which is
# rotating with Earth. This is adding unphysical velocity to satellites from the rotating viewpoint and inappropriate for matching the Doppler shift.
# 
# Fixed point in ECEF frame rorates in intertial frame (x, y):
# 
# - $x = R \cos \omega t$,
# - $y = R \sin \omega t$,
# 
# where $R = \sqrt{x^2 + y^2}$ is fixed.
# 
# Their time derivatives:
# 
# - $\dot{x} = - R \omega \sin \omega t = - \omega y$,
# - $\dot{y} = R \omega \cos \omega t = \omega x $.
# 
# 
# Subtract this rotation velocity to obtain inertial-frame velocity:
# 
# - $v_x = v_x^\mathrm{ECEF} + \omega y$,
# - $v_y = v_y^\mathrm{ECEF} - \omega x$,
# - $v_z = v_z^\mathrm{ECEF}$.
# 
# 
# https://en.wikipedia.org/wiki/Rotating_reference_frame

# In[ ]:


m = len(truth)  # Number of timesteps
y_wls = np.zeros((m, 3))  # Receiver positions estimated here
v_wls = np.zeros((m, 3))  # Receiver velocities estimated in ECEF frame

for i, (t_nano, df1) in enumerate(tqdm(gnss.groupby('TimeNanos'), total=m)):
    #
    # 1. Position estimation
    #
    
    # Corrected pseudo range ρ [m]
    rho = (df1['RawPseudorangeMeters'] + df1['SvClockBiasMeters'] - df1['IsrbMeters']
           - df1['IonosphericDelayMeters'] - df1['TroposphericDelayMeters']).values

    # Satellite positions at emmision time t_i in ECEF(t_i)
    x_sat = df1[['SvPositionXEcefMeters', 'SvPositionYEcefMeters', 'SvPositionZEcefMeters']].values

    # Inverse uncertainty weight
    w = 1 / df1['RawPseudorangeUncertaintyMeters'].values

    def f(y):
        """
        Compute error for trial receiver position y

        y (y1, y2, y3, b):
          y: recerver position at receiving time
          b: receiver clock bias in meters
        """
        b = y[3]
        r = rho - b  # distance to each satellite [m]
        tau = r / c  # signal flight time

        # Rotate satellite positions at emission to present ECEF coordinate
        x = np.empty_like(x_sat)
        cosO = np.cos(omega * tau)
        sinO = np.sin(omega * tau)
        x[:, 0] =  cosO * x_sat[:, 0] + sinO * x_sat[:, 1]
        x[:, 1] = -sinO * x_sat[:, 0] + cosO * x_sat[:, 1]
        x[:, 2] = x_sat[:, 2]

        return w * (np.sqrt(np.sum((x - y[:3])**2, axis=1)) - r)

    
    # Fit receiver position y and clock bias b
    x0 = np.zeros(4)  # initial guess
    opt = scipy.optimize.least_squares(f, x0)
    y = opt.x[:3]
    b = opt.x[3]
    
    #
    # 2. Velocity estimation
    #
    
    # Use estimated position
    r = rho - b  # distance to each satellite [m]
    tau = r / c
    
    # Satellite positions at emission in present (signal arrival time) ECEF coordinate
    x = np.empty_like(x_sat)
    cosO = np.cos(omega * tau)
    sinO = np.sin(omega * tau)
    x[:, 0] =  cosO * x_sat[:, 0] + sinO * x_sat[:, 1]
    x[:, 1] = -sinO * x_sat[:, 0] + cosO * x_sat[:, 1]
    x[:, 2] = x_sat[:, 2]

    v_sat_ecef = df1[['SvVelocityXEcefMetersPerSecond',
                      'SvVelocityYEcefMetersPerSecond',
                      'SvVelocityZEcefMetersPerSecond']].values
        
    # Velocity in inertial frame (matching ECEF at signal emission time)
    v_sate = np.empty_like(v_sat_ecef)
    v_sate[:, 0] = v_sat_ecef[:, 0] - omega * x_sat[:, 1]
    v_sate[:, 1] = v_sat_ecef[:, 1] + omega * x_sat[:, 0]
    v_sate[:, 2] = v_sat_ecef[:, 2]

    # Rotate the velocity to another inertial frame matching ECEF at signal arrival time
    v_sat = np.empty_like(v_sat_ecef)
    v_sat[:, 0] =  cosO * v_sate[:, 0] + sinO * v_sate[:, 1]
    v_sat[:, 1] = -sinO * v_sate[:, 0] + cosO * v_sate[:, 1]
    v_sat[:, 2] = v_sate[:, 2]

    # Direction from receiver to sattelites
    r_vec = x - y.reshape(1, 3)
    r_hat = r_vec / np.linalg.norm(r_vec, axis=1).reshape(-1, 1)  # unit vector
    
    # Line-of-sight velocity from doppler shift data
    v_los = df1['PseudorangeRateMetersPerSecond'].values  

    # Inverse uncertainty in v_los for weights
    w_vel = 1 / df1['PseudorangeRateUncertaintyMetersPerSecond'].values
    
    def f_vel(v):
        """
        Return weighted error for velocity estimate v

        v (v1, v2, v3, v_b): Receiver velocity and velocity bias
        """    
        # Line-of-sight relative velocity for fitting parameter v
        v_rel = np.sum((v_sat - v[:3].reshape(1, 3)) * r_hat, axis=1)  # dot product to r_hat

        err = w_vel * (v_rel - v_los + v[3])

        return err
    
    v0 = np.zeros(4)  # initial guess
    opt = scipy.optimize.least_squares(f_vel, v0)
    v = opt.x[:3]
    vb = opt.x[3]
    
    # Receiver velocity in ECEF frame
    v_ecef = np.zeros(3)
    v_ecef[0] = v[0] + omega * y[1]
    v_ecef[1] = v[1] - omega * y[0]
    v_ecef[2] = v[2]

    # Save result
    y_wls[i, :] = y
    v_wls[i, :] = v_ecef


# # Observing the result

# In[ ]:


speed_wls = np.linalg.norm(v_wls, axis=1)
speed_truth = truth['SpeedMps'].values

bl_truth = truth[['LatitudeDegrees', 'LongitudeDegrees']].values

y_baseline = gnss.groupby('TimeNanos')[['WlsPositionXEcefMeters', 'WlsPositionYEcefMeters', 'WlsPositionZEcefMeters']].mean().values
y_baseline.shape, speed_truth.shape


# In[ ]:


plt.figure(figsize=(8, 6))
plt.subplot(2, 1, 1)
plt.title('Velocity estimation')
plt.ylabel('speed [m/s]')
plt.plot(speed_truth, label='truth')
plt.plot(speed_wls, alpha=0.5, label='this work')
plt.legend(frameon=False)

plt.subplot(2, 1, 2)
plt.xlabel('Timestep $\\approx$ sec')
plt.ylabel('Error speed [m/s]')
plt.axhline(0, color='gray', alpha=0.5)
plt.plot(speed_wls - speed_truth)
plt.show()


# In[ ]:


bins = np.linspace(-1, 1, 101)
plt.title('Speed error')
plt.xlabel('$v - v_\\mathrm{true}$ [m/s]')
plt.hist(speed_wls - speed_truth, bins, alpha=0.8)
plt.axvline(0, color='gray', alpha=0.5)
plt.show()


# In[ ]:


plt.figure(figsize=(14, 6))
plt.suptitle('Position estimation (xyz)')
for k in range(3):
    plt.subplot(2, 3, k + 1)
    plt.title('xyz'[k])
    plt.plot(y_baseline[:, k], label='Baseline WLS')
    plt.plot(y_wls[:, k], label='This notebook')
    
    if k == 0:
        plt.ylabel('Position [m]')
        plt.legend(frameon=False)
    
    plt.subplot(2, 3, k + 4)
    plt.axhline(0, color='gray', alpha=0.5)
    plt.plot(y_wls[:, k] - y_baseline[:, k], label='This notebook')
    
    if k == 0:
        plt.ylabel('WLS disagreement [m]')


# In[ ]:


"""
xyz to latitude longitude
Thanks to: Akio Saito
https://www.kaggle.com/code/saitodevel01/gsdc2-baseline-submission
"""
    
WGS84_SEMI_MAJOR_AXIS = 6378137.0
WGS84_SEMI_MINOR_AXIS = 6356752.314245

def to_blh(xyz, *, unit='degree'):
    """
    Args:
      x, y, z (float): ecef coordinate in meters
      unit (str): Unit for latitude longitude, degree or radian
    Returns:
      B: geodetic latitude
      L: geodesic longitude
      H: height above the ellipsoid
    """
    assert unit == 'degree' or unit == 'radian'
    
    x = xyz[:, 0]
    y = xyz[:, 1]
    z = xyz[:, 2]
    blh = np.empty_like(xyz)

    # Ellipsoidal parameters
    a = WGS84_SEMI_MAJOR_AXIS
    b = WGS84_SEMI_MINOR_AXIS
    f = (a - b) / a
    e2 = 2*f - f**2
    ep2 = (a**2 - b**2) / b**2

    # Transformation
    r = np.sqrt(x**2 + y**2)
    theta = np.arctan2(z * (a / b), r)
    B = np.arctan2(z + (ep2 * b) * np.sin(theta)**3, r - (e2 * a) * np.cos(theta)**3)
    blh[:, 0] = B
    blh[:, 1] = np.arctan2(y, x)
    n = a / np.sqrt(1 - e2 * np.sin(B)**2)
    blh[:, 2] = (r / np.cos(B)) - n

    if unit == 'degree':
        blh[:, 0] = np.degrees(blh[:, 0])
        blh[:, 1] = np.degrees(blh[:, 1])

    return blh

blh = to_blh(y_wls)
blh_baseline = to_blh(y_baseline)


# In[ ]:


plt.figure(figsize=(12, 6))

for k in range(2):
    plt.subplot(2, 2, k + 1)
    plt.title(['Latitude', 'Longitude'][k])
    plt.plot(bl_truth[:, k], color='gray', label='truth')
    plt.plot(blh_baseline[:, k], label='baseline')
    plt.plot(blh[:, k], label='this notebook')
    
    if k == 0:
        plt.ylabel('lat/lon [degree]')
    
    plt.subplot(2, 2, k + 3)
    plt.axhline(0, color='gray')
    plt.plot(blh_baseline[:, k] - bl_truth[:, k], label='baseline')
    plt.plot(blh[:, k] - bl_truth[:, k], alpha=0.5, label='this notebook')
    
    if k == 0:
        plt.ylabel('Error [degree]')
        plt.legend(frameon=False)


# ## Rough consistency check
# 
# xyz is not a good coordinate system on Earth, east north up (ENU) coordinate is more intuitive locally. However, which velocity representation is best depends on how you use the velocities and positions
# with time evolution. Since this notebook is limited to treating each timestep separately, I'll stop here with xyz velocity components and end with a rough consistency check between positions and velocities.
# 
# $$ \vec{v}_\mathrm{diff}(t_i) \approx \frac{\vec{x}(t_{i+1}) - \vec{x}(t_{i-1})}{2\Delta t}$$
# 
# $\Delta t$ is usually 1 s but sometimes varies.

# In[ ]:


v_diff = (y_wls[2:, :] - y_wls[:-2, :]) / 2
    
plt.figure(figsize=(14, 3))
plt.suptitle('Position Velocity consistency')

for k in range(3):
    plt.subplot(1, 3, k + 1)
    plt.title('xyz'[k])
    plt.xlabel('timestep ~ sec')
    plt.plot(v_diff[:, k], label='Difference')
    plt.plot(v_wls[:, k], label='Doppler')
    
    if k == 0:
        plt.ylabel('$v_k$ [m/s]')
        plt.legend(frameon=False)


# Nice. At least signs are correct.
