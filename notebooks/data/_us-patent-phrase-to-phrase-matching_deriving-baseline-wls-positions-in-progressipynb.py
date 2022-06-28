#!/usr/bin/env python
# coding: utf-8

# This notebook attempts to derive the weighted-least-square (WLS) positions, 
# provided as `WlsPositionXYZEcefMeters` from the GNSS data, but
# the agreement so far is ~ 1-2 m.
# 
# ### Update
# Version 4: I added satellite selection based on `CarrierFrequencyHz` and the accuracy improved.
# Two WSL results do not agree for 1-2 meters, but the score against ground truth are now similar
# for the first drive `2020-05-15-US-MTV-1`.
# 
# "in progress" forever: I realized writing "in progress" in the title was not a good idea because
# it was part of the URL. Although I am not sure if I can do anything beter, I leave it in the title.
# 
# ## Introduction
# 
# ### Simplest theory
# 
# We know the GNSS satelite positions, $\vec{x}_i$, from human civilization and the distance to them $r_i$ using the time differences. Our (receiver) location $\vec{y}$ satisfies,
# 
# $$ \lVert \vec{x}_i - \vec{y} \rVert = r_i, $$
# 
# for each satellite $i$, where $\lVert \cdot \rVert$ is the usual vector length, aka Euclidean norm, or
# the $L^2$ norm in 3 dimenssion.
# 
# ### More realistically
# 
# All data contain some uncertainties and after various corrections, the uncertainty
# in the receiver clock remains. This becomes an unknown offset $b$ to the distance $r_i$ and the distance including this bias is called the *pseudo range*: $\rho_i = r_i + b$.
# 
# WLS solution minimizes the error:
# 
# $$ \mathrm{WMSE} = \sum_i \left( \frac{\lVert \vec{x}_i - \vec{y} \rVert - \rho_i + b}{\Delta \rho_i} \right)^2$$
# 
# where $\Delta \rho_i$ is the uncertainty for $\rho_i$ provided as `RawPseudorangeUncertaintyMeters`.
# 
# 4 unknowns, receiver position $\vec{y}$ and clock bias $b$, can be solved with 4 or more satellites.
# 
# ### Earth rotation
# 
# The satellite positions `SvPositionXYZEcefMeters` are provided in Earth-centered Earth-fixed (ECEF) coordinate system at *signal emission time*, but the earth rotates while
# the signal reaches the receiver. We want the satellite position in the current ECEF *coordinate at receiving time* and need to rotate the given satellite coordinates. Note that this is 
# *not* satellite motion; we want satellite position at emission time, this is fixed, but the
# coordinate system is rotating. After the signal emission, our foucus is in the motion of signal and the electromagnetic signal certainly does not rotate together with Earth at 0th order, although there could be some corrections from the atmosphere that feels the Earth rotation. It must be closer to a straight line in the non-rotating frame than a straight line in the rotating ECEF coordinate.
# 
# ## Reference
# 
# There are excellent WLS notebooks in the 2021 Google Smartphone Decimeter Challenge and I basically only update the column names. 
# 
# [1] https://www.kaggle.com/code/foreveryoung/least-squares-solution-from-gnss-derived-data
# 
# [2] https://www.kaggle.com/code/hyperc/gsdc-reproducing-baseline-wls-on-one-measurement
# 
# (2) adds Earth rotations to (1). I add bais corrections for Earth roation in (2) but the change is negligible unless the receiver clock error is huge.
# 
# Excellent link in (2)
# 
# [3] Calculating Position from Raw GPS Data: https://www.telesens.co/2017/07/17/calculating-position-from-raw-gps-data/
# 
# I am watching a YouTube course:
# https://www.youtube.com/watch?v=o1Fyn_h6LKU&list=PLGvhNIiu1ubyEOJga50LJMzVXtbUq6CPo
# 
# linked from standford.edu:
# https://scpnt.stanford.edu/about/gps-mooc-massive-open-online-course
# 
# 
# ### Disclaimer
# 
# This is my first experience handling GPS/GNSS data with no prior knowledge.
# I am writing this for the sake of my own study; correction comments are welcome.
# 

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
import scipy.optimize
from tqdm.auto import tqdm

c = 299_792_458  # m/s in vacuum
omega = 7.292115e-5  # angular velocity [rad/s] in WGS 84


# # Data
# 
# One epoch (one time) contains multiple rows from many satellites

# In[ ]:


path = '/kaggle/input/smartphone-decimeter-2022/train/2020-05-15-US-MTV-1/GooglePixel4XL'
gnss = pd.read_csv('%s/device_gnss.csv' % path, dtype={'SignalType': str})
truth = pd.read_csv('%s/ground_truth.csv' % path)

# Add standard Frequency column
frequency_median = gnss.groupby('SignalType')['CarrierFrequencyHz'].median()
gnss = gnss.merge(frequency_median, how='left', on='SignalType', suffixes=('', 'Ref'))

# One epoch
for t_nano, df1 in gnss.groupby('TimeNanos'):
    break
df1[['Svid', 'ConstellationType', 'SignalType', 'CarrierFrequencyHz', 'CarrierFrequencyHzRef']]


# # WLS Position estimation

# In[ ]:


n = len(truth)
y_wls = np.zeros((n, 3))
errs = []
n_sat_sum = 0   # cumulative number of satellites in data
n_sat_used = 0  # cumulative number of satellites used for WLS

# Loop over epochs, i.e. single time with multiple satellites
for i, (t_nano, df1) in enumerate(tqdm(gnss.groupby('TimeNanos'), total=n)):
    n_sat_sum += len(df1)
        
    # Satellite selection
    idx = (df1['CarrierFrequencyHz'] - df1['CarrierFrequencyHzRef']) < 2.5e6
    # one-side cut seems fine for this drive but abs() < th is also natural
    #idx &= df1['Cn0DbHz'] >= 20.0
    df1 = df1[idx]
    
    n_sat_used += len(df1)
    
    # Corrected pseudo range ρ [m]
    rho = (df1['RawPseudorangeMeters'] + df1['SvClockBiasMeters'] - df1['IsrbMeters']
           - df1['IonosphericDelayMeters'] - df1['TroposphericDelayMeters']).values

    # Satellite positions at emmision time t_i in ECEF(t_i)
    x_sat = df1[['SvPositionXEcefMeters', 'SvPositionYEcefMeters', 'SvPositionZEcefMeters']].values

    # Inverse uncertainty weight
    w = 1 / df1['RawPseudorangeUncertaintyMeters'].values

    def f(y):
        """
        Compute error for guess y

        y (y1, y2, y3, b):
          y: recerver position at receiving time
          b: receiver clock bias in meters
        """
        b = y[3]
        r = rho - b  # distance to each satellite [m]
        tau = r / c  # signal flight time

        # Rotate satellite positions at emission for present ECEF coordinate
        x = np.empty_like(x_sat)
        cosO = np.cos(omega * tau)
        sinO = np.sin(omega * tau)
        x[:, 0] =  cosO * x_sat[:, 0] + sinO * x_sat[:, 1]
        x[:, 1] = -sinO * x_sat[:, 0] + cosO * x_sat[:, 1]
        x[:, 2] = x_sat[:, 2]

        return w * (np.sqrt(np.sum((x - y[:3])**2, axis=1)) - r)

    
    # Fit receiver position y and clock bias b
    x0 = np.zeros(4)  # Initial guess
    opt = scipy.optimize.least_squares(f, x0)
    y = opt.x[:3]
    b = opt.x[3]
    
    y_wls[i, :] = y
    errs.append(opt.fun)
    
print(n_sat_used, '/', n_sat_sum)


# In[ ]:


# Provided WLS position
y_baseline = gnss.groupby('TimeNanos')[['WlsPositionXEcefMeters', 'WlsPositionYEcefMeters', 'WlsPositionZEcefMeters']].mean().values
y_baseline.shape


# # Compare to WLS baseline
# 
# Comparison between two calculations supposed to be same in method and data.

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

# Error
print('rms error', np.sqrt(np.mean((y_wls - y_baseline)**2, axis=0)), '[m]')


# # Comprare to ground truth

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


def haversine_distance(x, y, *, unit='degree'):
    """
    Compute haversine on sphere
    x (latitude, longitude)
    y (latitude, longitude)
    """
    assert unit in ['degree', 'radian']
    HAVERSINE_RADIUS = 6371_000

    if unit == 'degree':
        x = np.radians(x)
        y = np.radians(y)
    
    phi_x = x[:, 0]
    phi_y = y[:, 0]
    db = phi_x - phi_y
    dl = x[:, 1] - y[:, 1]

    a = np.sin(db / 2)**2 + np.cos(phi_x) * np.cos(phi_y) * np.sin(dl / 2)**2
    dist = 2 * HAVERSINE_RADIUS * np.arcsin(np.sqrt(a))

    return dist

def score(x, y):
    dists = haversine_distance(x, y)
    score = np.mean([np.quantile(dists, 0.50), np.quantile(dists, 0.95)])    
    return score


# In[ ]:


blh = to_blh(y_wls)
blh_baseline = to_blh(y_baseline)
bl_truth = truth[['LatitudeDegrees', 'LongitudeDegrees']].values

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


# In[ ]:


print('Scores')
print('This work     %.4f [m]' % score(blh[:, :2], bl_truth))
print('Baseline WLS  %.4f [m]' % score(blh_baseline[:, :2], bl_truth))


# |                | RMSE [m] | score [m] | nunber of data  |
# |:---             |---      |---        |---              |
# |No selection    | 3.8214   | 5.4925    | 90153 / 90153   |
# |f cut 2.5MHz    | 3.0545   | 4.2920    | 89713 / 90153   |
# |+Cn0DbHz' ≧ 20.0| 3.1383   | 4.3750    | 86612 / 90153   |
# |Baseline        | 3.0707   | 4.3185    |                 |

# * My WLS disagree with the WLS baseline for ~ 1-2 m in each direction;
# * but both WLS perform similarly against ground truth after CarrierFrequency selection.
# * The error is obviously large without CarrierFrequency selection.
# * Carrier-to-noise ≧ 20 db-Hz does not improve the score.

# ## Remark
# 
# Other attempts I made.
# 
# ### 1. Signal-type dependent biases
# 
# In the discussion in 2021,
# https://www.kaggle.com/c/google-smartphone-decimeter-challenge/discussion/238583
#     
# The organizer said:
# 
# > It has 4+N states, where 4 refers to the user's position in ECEF and clock offset (x, y, z, t), and N states are inter-signal biases (ISB) for the number of non-GPS-L1 signal types. For instance, if the device measures signals of GPS L1 frequency, GLO G1 frequency, GPS L5 frequency, GAL E1 frequency at the same epoch, the number of non-GPS-L1 signal types equals 3 (i.e. N=3).
# 
# This probably means that there are biases for each signal types, 4 + N parameters in the least-square fitting.
# 
# However, this only made less than 1e-3 m change.
# 
# ### 2. Remove atmospheric correction for tau
# When we correct for the ECEF rotation during signal arrival, what we want is time and I assume speed of light in vacuum tau = r / c
# to get the time.
# The atmospheric corrections, IonosphericDelayMeters and TroposphericDelayMeters, are probably correction to the effective positions of satellites for non-vacuume speed of light, and the delays are true delays in time. Therefore using time difference without those atmospheric corrections might be better for tau:
# 
# ```
# tau_c = (df1['RawPseudorangeMeters'] + df1['SvClockBiasMeters'])  # without two delays
# tau = (tau_c - b) / c
# ```
# 
# However, this only makes 1e-3 m difference to the xyz estimation and I leave the equation simple.
# 
# ### 3. ReceivedSvTimeUncertaintyNanos
# 
# `ReceivedSvTimeUncertaintyNanos` is exactly proportional to `RawPseudorangeUncertaintyMeters`; using which of them for weight does not matter.
# 
# ### 4. Invalid measurements
# 
# The discussion above describes several criteria for "invalid measurements." Same critera are written in
# the data section for `[train/test]/[drive_id]/[phone_name]/supplemental/rinex.o`, e.g.,
# 
# - CN0 is less than 20 dB-Hz
# - Carrier frequency is out of nominal range of each band
# 
# I read the `supplemental/gnss_rinex.20o` and remove the data with blank pseudo range in that file, but
# agreement between two WLS did not change very much.
# 
# 

# ### Computation speed
# 
# This notebook is not aiming for computation speed. Textbook example of least-square fitting is linearizing the equation and using linear solver. The trigonometric functions for rotations can also be Taylor expended for small rotation angles.
