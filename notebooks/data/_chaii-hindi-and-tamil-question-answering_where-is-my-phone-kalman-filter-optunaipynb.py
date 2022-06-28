#!/usr/bin/env python
# coding: utf-8

# ## Introduction
# 
# This notebook is for "Google Smartphone Decimeter Challenge 2022" where we need to estimate the precise location (longitude, latitude) of a mobile phone using Global Navigation Satellite System (GNSS) signal.
# 
# However, because the organizer gave us a baseline for this challenge, we might firsly build a baseline by simply smoothering the baseline.
# 
# In this notebook you'll find:
# 
# - What is Kalman Filter
# - How to tune its parameters with Optuna
# 
# Credit to this discussion https://www.kaggle.com/competitions/smartphone-decimeter-2022/discussion/323548 from @Ravi Shad that I found many interesting resources
# - baseline https://www.kaggle.com/competitions/smartphone-decimeter-2022/discussion/323548 from @saitodevel01
# - Last year kalman filter approach https://www.kaggle.com/code/tqa236/kalman-filter-hyperparameter-search-with-bo by @Trinh Quoc Anh

# ### What is Kalman filter
# 
# - Actually, I don't really like the term "filter" in Kalman Filter, it makes me think about it simply smoothing the output. However, Kalman filter is more powerful than that, it has the learning process behind. It will readapt the parameters on the go by predicting and comparing with the real measurements with a goal is minimizing the uncertainty as much as possible (or covariance)
# 
# - There are 3 steps in Kalman filter:
#     1. Predicting the output with the current parameters 
#     2. Comparing with the real mesurement from sensors
#     3. Update the parameters to minimize the uncertainty
#     
# ![Kalman Filter steps](https://upload.wikimedia.org/wikipedia/commons/thumb/a/a5/Basic_concept_of_Kalman_filtering.svg/1200px-Basic_concept_of_Kalman_filtering.svg.png ) 
#    
# [Fun fact]: In the 1960s, the Kalman filter was applied to navigation for the Apollo Project, which required estimates of the trajectories of manned spacecraft going to the Moon and back . Source: https://ieeexplore.ieee.org/document/5466132
# 
# Disadvantage: However, Kalman filter have some unavoidable weakness:
# - We have to correctly modelling the model (which is most of the time not obvious)
# - The uncertainty is modeled by Gaussian noise, which is not always the case, and even if we have that, getting the value is not obvious neither ( It's why I use Optuna for this purpose )
# 

# ## Code

# In[ ]:


# install kalmal filter library
get_ipython().system('pip install simdkalman')


# In[ ]:


from tqdm.notebook import tqdm
from dataclasses import dataclass
from scipy.interpolate import InterpolatedUnivariateSpline
import glob
from joblib import Parallel, delayed
import random
import simdkalman
import optuna
from functools import partial
import numpy as np
import pandas as pd
pd.set_option('display.max_columns', 50)


# Many snippets below I copy from 2 notebooks you can find in the introduction

# In[ ]:


WGS84_SEMI_MAJOR_AXIS = 6378137.0
WGS84_SEMI_MINOR_AXIS = 6356752.314245
WGS84_SQUARED_FIRST_ECCENTRICITY  = 6.69437999013e-3
WGS84_SQUARED_SECOND_ECCENTRICITY = 6.73949674226e-3

HAVERSINE_RADIUS = 6_371_000


# In[ ]:


@dataclass
class ECEF:
    x: np.array
    y: np.array
    z: np.array

    def to_numpy(self):
        return np.stack([self.x, self.y, self.z], axis=0)

    @staticmethod
    def from_numpy(pos):
        x, y, z = [np.squeeze(w) for w in np.split(pos, 3, axis=-1)]
        return ECEF(x=x, y=y, z=z)

@dataclass
class BLH:
    lat : np.array
    lng : np.array
    hgt : np.array


# In[ ]:


def ECEF_to_BLH(ecef):
    a = WGS84_SEMI_MAJOR_AXIS
    b = WGS84_SEMI_MINOR_AXIS
    e2  = WGS84_SQUARED_FIRST_ECCENTRICITY
    e2_ = WGS84_SQUARED_SECOND_ECCENTRICITY
    x = ecef.x
    y = ecef.y
    z = ecef.z
    r = np.sqrt(x**2 + y**2)
    t = np.arctan2(z * (a/b), r)
    B = np.arctan2(z + (e2_*b)*np.sin(t)**3, r - (e2*a)*np.cos(t)**3)
    L = np.arctan2(y, x)
    n = a / np.sqrt(1 - e2*np.sin(B)**2)
    H = (r / np.cos(B)) - n
    return BLH(lat=B, lng=L, hgt=H)


# In[ ]:


def haversine_distance(blh_1, blh_2):
    dlat = blh_2.lat - blh_1.lat
    dlng = blh_2.lng - blh_1.lng
    a = np.sin(dlat/2)**2 + np.cos(blh_1.lat) * np.cos(blh_2.lat) * np.sin(dlng/2)**2
    dist = 2 * HAVERSINE_RADIUS * np.arcsin(np.sqrt(a))
    return dist

def pandas_haversine_distance(df1, df2):
    blh1 = BLH(
        lat=np.deg2rad(df1['LatitudeDegrees'].to_numpy()),
        lng=np.deg2rad(df1['LongitudeDegrees'].to_numpy()),
        hgt=0,
    )
    blh2 = BLH(
        lat=np.deg2rad(df2['LatitudeDegrees'].to_numpy()),
        lng=np.deg2rad(df2['LongitudeDegrees'].to_numpy()),
        hgt=0,
    )
    return haversine_distance(blh1, blh2)


# In[ ]:


def ecef_to_lat_lng(gnss_df, UnixTimeMillis):
    ecef_columns = ['WlsPositionXEcefMeters', 'WlsPositionYEcefMeters', 'WlsPositionZEcefMeters']
    columns = ['utcTimeMillis'] + ecef_columns
    ecef_df = (gnss_df.drop_duplicates(subset='utcTimeMillis')[columns]
               .dropna().reset_index(drop=True))
    ecef = ECEF.from_numpy(ecef_df[ecef_columns].to_numpy())
    blh  = ECEF_to_BLH(ecef)

    TIME = ecef_df['utcTimeMillis'].to_numpy()
    lat = InterpolatedUnivariateSpline(TIME, blh.lat, ext=3)(UnixTimeMillis)
    lng = InterpolatedUnivariateSpline(TIME, blh.lng, ext=3)(UnixTimeMillis)
    return pd.DataFrame({
        'UnixTimeMillis'   : UnixTimeMillis,
        'LatitudeDegrees'  : np.degrees(lat),
        'LongitudeDegrees' : np.degrees(lng),
    })


# The state transition in our problem is quite obvious with a highschool physics, we know that. D_new = D_old + V * T + 1/2 * A * T^2 
# 
# where as: D(Distance), V(Speed), A(Acceleration)
# 
# The order of states vector (6x1) are: [Latitude, Longitue, V_lat, V_lon, A_lat, A_lon] 
# 
# We only tune the observation noise and process noise and the sampling time (T)

# In[ ]:


def make_kalman_filter(T, process_cov_mat, obs_cov_mat):
    
    state_transition =  np.array([[1, 0, T, 0, 0.5 * T ** 2, 0], 
                             [0, 1, 0, T, 0, 0.5 * T ** 2], 
                             [0, 0, 1, 0, T, 0],
                             [0, 0, 0, 1, 0, T], 
                             [0, 0, 0, 0, 1, 0], 
                             [0, 0, 0, 0, 0, 1]])
    
    observation_model = np.array([[1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0]])
    
    observation_noise = obs_cov_mat
    
    process_noise = process_cov_mat

    kf = simdkalman.KalmanFilter(
            state_transition = state_transition,
            process_noise = process_noise,
            observation_model = observation_model,
            observation_noise = observation_noise)
    return kf


# In[ ]:


def calc_score(pred_df, gt_df):
    d = pandas_haversine_distance(pred_df, gt_df)
    score = np.mean([np.quantile(d, 0.50), np.quantile(d, 0.95)])    
    return score


# In[ ]:


INPUT_PATH = '../input/smartphone-decimeter-2022/'


# In[ ]:


def apply_kf_smoothing(df, _kf):
    df_filter = df.copy()
    data = df_filter[['LatitudeDegrees','LongitudeDegrees']].to_numpy()
    data = data.reshape(1, len(data), 2)
    smoothed = _kf.smooth(data)
    df_filter['LatitudeDegrees'] = smoothed.states.mean[0, :, 0]
    df_filter['LongitudeDegrees'] = smoothed.states.mean[0, :, 1]
    return df_filter


# In[ ]:


dirnames = sorted(glob.glob(f'{INPUT_PATH}/train/*/*'))


# I will try to tune all the diagonal values and one value for off-diagonal. I think lattitude and logitude is diagonal so normally they are not correlated. Question: Do I need to tune Speed and Acceleration, or it can just derived from the position ?

# ## Testing the process for 1 tripId

# In[ ]:


process_cov_00 = 10**-6
process_cov_11 = 10**-6
process_cov_22 = 10**-6
process_cov_33 = 10**-6
process_cov_44 = 10**-6
process_cov_55 = 10**-6
process_cov_66 = 10**-6
process_cov_off_diag = 10**-9


# In[ ]:


process_diag = [process_cov_00, process_cov_11, process_cov_22, process_cov_33, process_cov_44, process_cov_55]


# In[ ]:


def make_cov_mat(process_diag, process_cov_off_diag):
    rank = len(process_diag)
    process_cov_mat = np.zeros((rank,rank))
    np.fill_diagonal(process_cov_mat, process_diag)
    process_cov_mat = process_cov_mat + process_cov_off_diag
    return process_cov_mat


# In[ ]:


process_cov_mat = make_cov_mat(process_diag, process_cov_off_diag)


# In[ ]:


obs_cov_00 = 10**-6
obs_cov_11 = 10**-6
obs_cov_off_diag = 10**-9


# In[ ]:


obs_cov_mat = make_cov_mat([obs_cov_00, obs_cov_11], obs_cov_off_diag)


# In[ ]:


process_cov_mat


# In[ ]:


obs_cov_mat


# In[ ]:


T = 1


# In[ ]:


kf0 = make_kalman_filter(T, process_cov_mat, obs_cov_mat)


# In[ ]:


# Test for 1 tripId
dirname = dirnames[0]
pred_dfs = []
drive, phone = dirname.split('/')[-2:]
tripID  = f'{drive}/{phone}'
gnss_df = pd.read_csv(f'{dirname}/device_gnss.csv')
gt_df   = pd.read_csv(f'{dirname}/ground_truth.csv')
pred_df = ecef_to_lat_lng(gnss_df, gt_df['UnixTimeMillis'].to_numpy())
gnss_df = pd.read_csv(f'{dirnames[0]}/device_gnss.csv')
pred_dfs.append(pred_df)


# In[ ]:


score = calc_score(pred_df, gt_df)


# In[ ]:


score


# In[ ]:


pred_df_filter = apply_kf_smoothing(pred_df, kf0)


# In[ ]:


score = calc_score(pred_df_filter, gt_df)


# In[ ]:


score


# ### Training (Tuning with Optuna)

# In[ ]:


nb_file = len(dirnames)


# In[ ]:


train_dfs = [pd.read_csv(f'{dirname}/device_gnss.csv') for dirname in dirnames[:nb_file]]


# In[ ]:


train_gts = [pd.read_csv(f'{dirname}/ground_truth.csv') for dirname in dirnames[:nb_file]]


# In[ ]:


filter_fn = partial(apply_kf_smoothing, _kf=kf0)


# In[ ]:


def all_score(train_dfs, train_gts, filter_fn=None):
    """ Calculate the score for list of df"""
    scores = []
    for gnss_df, gt_df in zip(train_dfs, train_gts):
        pred_df = ecef_to_lat_lng(gnss_df, gt_df['UnixTimeMillis'].to_numpy())
        if filter_fn:
            pred_df = filter_fn(pred_df)
        score = calc_score(pred_df, gt_df)
        scores.append(score)
    return scores


# In[ ]:


def _make_kalman_filter(T, process_cov_mat, obs_cov_mat):
    state_transition =  np.array([[1, 0, T, 0, 0.5 * T ** 2, 0], 
                             [0, 1, 0, T, 0, 0.5 * T ** 2], 
                             [0, 0, 1, 0, T, 0],
                             [0, 0, 0, 1, 0, T], 
                             [0, 0, 0, 0, 1, 0], 
                             [0, 0, 0, 0, 0, 1]])
    
    observation_model = np.array([[1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0]])
    
    observation_noise = obs_cov_mat
    
    process_noise = process_cov_mat

    kf = simdkalman.KalmanFilter(
            state_transition = state_transition,
            process_noise = process_noise,
            observation_model = observation_model,
            observation_noise = observation_noise)
    return kf


# In[ ]:


def make_kalman_filter(param):
    process_diag = [param["process_cov_00"], 
                    param["process_cov_11"], 
                    param["process_cov_22"], 
                    param["process_cov_33"], 
                    param["process_cov_44"], 
                    param["process_cov_55"]]
    process_cov_off_diag = param["process_cov_off_diag"]

    obs_diag = [param["obs_cov_00"],
                param["obs_cov_11"]]
    obs_cov_off_diag = [param["obs_cov_off_diag"]]

    T = param['T']

    process_cov_mat = make_cov_mat(process_diag, process_cov_off_diag)
    obs_cov_mat = make_cov_mat([obs_cov_00, obs_cov_11], obs_cov_off_diag)

    _kf = _make_kalman_filter(T, process_cov_mat, obs_cov_mat)
    return _kf


# In[ ]:


def objective(trial):
    
    param = {
        'process_cov_00': trial.suggest_float("process_cov_00", 1e-8, 1e-4, log=True),
        'process_cov_11': trial.suggest_float("process_cov_11", 1e-8, 1e-4, log=True),
        'process_cov_22': trial.suggest_float("process_cov_22", 1e-8, 1e-4, log=True),
        'process_cov_33': trial.suggest_float("process_cov_33", 1e-8, 1e-4, log=True),
        'process_cov_44': trial.suggest_float("process_cov_44", 1e-8, 1e-4, log=True),
        'process_cov_55': trial.suggest_float("process_cov_55", 1e-8, 1e-4, log=True),
        'process_cov_off_diag': trial.suggest_float("process_cov_off_diag", 1e-12, 1e-8, log=True),
        
        'obs_cov_00': trial.suggest_float("obs_cov_00", 1e-8, 1e-4, log=True),
        'obs_cov_11': trial.suggest_float("obs_cov_11", 1e-8, 1e-4, log=True),
        'obs_cov_off_diag': trial.suggest_float("obs_cov_off_diag", 1e-12, 1e-8, log=True),
        
        'T': trial.suggest_float("T", 0.6, 1.4),
    }
    
    process_diag = [param["process_cov_00"], 
                    param["process_cov_11"], 
                    param["process_cov_22"], 
                    param["process_cov_33"], 
                    param["process_cov_44"], 
                    param["process_cov_55"]]
    process_cov_off_diag = param["process_cov_off_diag"]
    
    obs_diag = [param["obs_cov_00"],
                param["obs_cov_11"]]
    obs_cov_off_diag = [param["obs_cov_off_diag"]]
    
    T = param['T']
    
    process_cov_mat = make_cov_mat(process_diag, process_cov_off_diag)
    obs_cov_mat = make_cov_mat([obs_cov_00, obs_cov_11], obs_cov_off_diag)
    
    _kf = _make_kalman_filter(T, process_cov_mat, obs_cov_mat)
    filter_fn = partial(apply_kf_smoothing,  _kf=_kf)
    scores = all_score(train_dfs, train_gts, filter_fn=filter_fn)
    
    return np.mean(scores)


# In[ ]:


study = optuna.create_study(direction='minimize')


# In[ ]:


study.optimize(objective, n_trials=3)


# In[ ]:


# These parameters below I got after trying n_trials = 100 
param = {'process_cov_00': 1.2020309620263925e-08,
 'process_cov_11': 2.5818076069527384e-08,
 'process_cov_22': 1.7418556333561943e-05,
 'process_cov_33': 8.783630087500364e-06,
 'process_cov_44': 3.190877235713106e-07,
 'process_cov_55': 1.2137227780694694e-07,
 'process_cov_off_diag': 2.1258126417177658e-09,
 'obs_cov_00': 2.5228974904239324e-07,
 'obs_cov_11': 1.2085757874535475e-05,
 'obs_cov_off_diag': 7.79887179271382e-09,
 'T': 0.7190536201746304}


# In[ ]:


sample_df = pd.read_csv(f'{INPUT_PATH}/sample_submission.csv')
pred_dfs  = []
for dirname in sorted(glob.glob(f'{INPUT_PATH}/test/*/*')):
    drive, phone = dirname.split('/')[-2:]
    tripID  = f'{drive}/{phone}'
    gnss_df = pd.read_csv(f'{dirname}/device_gnss.csv')
    UnixTimeMillis = sample_df[sample_df['tripId'] == tripID]['UnixTimeMillis'].to_numpy()
    pred_df = ecef_to_lat_lng(gnss_df, UnixTimeMillis)
    _kf = make_kalman_filter(param)
    pred_df = apply_kf_smoothing(pred_df, _kf)
    pred_df.insert(0, 'tripId', tripID)
    pred_dfs.append(pred_df)
baseline_test_df = pd.concat(pred_dfs)
baseline_test_df.to_csv('baseline_test.csv', index=False)
baseline_test_df.to_csv('submission.csv', index=False)


# In[ ]:




