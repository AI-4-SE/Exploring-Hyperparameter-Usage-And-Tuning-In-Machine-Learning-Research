#!/usr/bin/env python
# coding: utf-8

# # Acknowledgment
# * https://www.kaggle.com/braquino/convert-to-regression
# * https://www.kaggle.com/challenge1a3/convert-to-regression

# # My upgrade:
# * Dropout = 0.2 (everywhere)
# * learning_rate = 0.02 (everywhere)
# * weights = {'lbg': 0.6, 'xgb': 0.25, 'nn': 0.15}

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import xgboost as xgb
from xgboost import XGBClassifier, XGBRegressor
from xgboost import plot_importance
from matplotlib import pyplot
import random
import shap

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
from time import time
from tqdm import tqdm_notebook as tqdm
from collections import Counter
from scipy import stats
import lightgbm as lgb
from sklearn.metrics import cohen_kappa_score, mean_squared_error
from sklearn.model_selection import KFold, StratifiedKFold
import gc
import json
pd.set_option('display.max_columns', 1000)
from IPython.core.display import display, HTML
display(HTML("<style>div.output_scroll { height: 88em; }</style>"))  
import warnings
warnings.filterwarnings('ignore')


# In[ ]:


def eval_qwk_lgb_regr(y_true, y_pred):
    """
    Fast cappa eval function for lgb.
    """
    #dist = Counter(reduce_train['accuracy_group'])
    dist = {3.0: 9936, 0.0: 4978, 2.0: 2447, 1.0: 2676}
    s = sum(list(dist.values()))
    for k in dist:
        #dist[k] /= len(reduce_train)
        dist[k] /= s
    #reduce_train['accuracy_group'].hist()
    acum = 0
    bound = {}
    for i in range(3):
        acum += dist[i]
        bound[i] = np.percentile(y_pred, acum * 100)
        
    print('bound =', bound)
    
    def classify(x):
        if x <= bound[0]:
            return 0
        elif x <= bound[1]:
            return 1
        elif x <= bound[2]:
            return 2
        else:
            return 3
    y_pred = np.array(list(map(classify, y_pred))).reshape(y_true.shape)
    return 'cappa', cohen_kappa_score(y_true, y_pred, weights='quadratic'), True


# In[ ]:


def cohenkappa(ypred, y):
    y = y.get_label().astype("int")
    ypred = ypred.reshape((4, -1)).argmax(axis = 0)
    loss = cohenkappascore(y, y_pred, weights = 'quadratic')
    return "cappa", loss, True


# In[ ]:


def read_data():
    print('Reading train.csv file....')
    train = pd.read_csv('/kaggle/input/data-science-bowl-2019/train.csv')
    print('Training.csv file have {} rows and {} columns'.format(train.shape[0], train.shape[1]))

    print('Reading test.csv file....')
    test = pd.read_csv('/kaggle/input/data-science-bowl-2019/test.csv')
    print('Test.csv file have {} rows and {} columns'.format(test.shape[0], test.shape[1]))

    print('Reading train_labels.csv file....')
    train_labels = pd.read_csv('/kaggle/input/data-science-bowl-2019/train_labels.csv')
    print('Train_labels.csv file have {} rows and {} columns'.format(train_labels.shape[0], train_labels.shape[1]))

    print('Reading specs.csv file....')
    specs = pd.read_csv('/kaggle/input/data-science-bowl-2019/specs.csv')
    print('Specs.csv file have {} rows and {} columns'.format(specs.shape[0], specs.shape[1]))

    print('Reading sample_submission.csv file....')
    sample_submission = pd.read_csv('/kaggle/input/data-science-bowl-2019/sample_submission.csv')
    print('Sample_submission.csv file have {} rows and {} columns'.format(sample_submission.shape[0], sample_submission.shape[1]))
    return train, test, train_labels, specs, sample_submission


# In[ ]:


def encode_title(train, test, train_labels):
    keep_id = train[train.type == "Assessment"][['installation_id']].drop_duplicates()
    train = pd.merge(train, keep_id, on="installation_id", how="inner") 
    del keep_id
    # encode title
    train['title_event_code'] = list(map(lambda x, y: str(x) + '_' + str(y), train['title'], train['event_code']))
    test['title_event_code'] = list(map(lambda x, y: str(x) + '_' + str(y), test['title'], test['event_code']))
    all_title_event_code = list(set(train["title_event_code"].unique()).union(test["title_event_code"].unique()))
    # make a list with all the unique 'titles' from the train and test set
    list_of_user_activities = list(set(train['title'].unique()).union(set(test['title'].unique())))
    # make a list with all the unique 'event_code' from the train and test set
    list_of_event_code = list(set(train['event_code'].unique()).union(set(test['event_code'].unique())))
    list_of_event_id = list(set(train['event_id'].unique()).union(set(test['event_id'].unique())))
    # make a list with all the unique worlds from the train and test set
    list_of_worlds = list(set(train['world'].unique()).union(set(test['world'].unique())))
    # create a dictionary numerating the titles
    activities_map = dict(zip(list_of_user_activities, np.arange(len(list_of_user_activities))))
    activities_labels = dict(zip(np.arange(len(list_of_user_activities)), list_of_user_activities))
    activities_world = dict(zip(list_of_worlds, np.arange(len(list_of_worlds))))
    assess_titles = list(set(train[train['type'] == 'Assessment']['title'].value_counts().index).union(set(test[test['type'] == 'Assessment']['title'].value_counts().index)))
    # replace the text titles with the number titles from the dict
    train['title'] = train['title'].map(activities_map)
    test['title'] = test['title'].map(activities_map)
    train['world'] = train['world'].map(activities_world)
    test['world'] = test['world'].map(activities_world)
    train_labels['title'] = train_labels['title'].map(activities_map)
    win_code = dict(zip(activities_map.values(), (4100*np.ones(len(activities_map))).astype('int')))
    # then, it set one element, the 'Bird Measurer (Assessment)' as 4110, 10 more than the rest
    win_code[activities_map['Bird Measurer (Assessment)']] = 4110
    # convert text into datetime
    train['timestamp'] = pd.to_datetime(train['timestamp'])
    test['timestamp'] = pd.to_datetime(test['timestamp'])

    train['hour'] = train['timestamp'].dt.hour
    test['hour'] = test['timestamp'].dt.hour
    
    return train, test, train_labels, win_code, list_of_user_activities, list_of_event_code, activities_labels, assess_titles, list_of_event_id, all_title_event_code


# In[ ]:


clip_time = {'Welcome to Lost Lagoon!':19,'Tree Top City - Level 1':17,'Ordering Spheres':61, 'Costume Box':61,
        '12 Monkeys':109,'Tree Top City - Level 2':25, 'Pirate\'s Tale':80, 'Treasure Map':156,'Tree Top City - Level 3':26,
        'Rulers':126, 'Magma Peak - Level 1':20, 'Slop Problem':60, 'Magma Peak - Level 2':22, 'Crystal Caves - Level 1':18,
        'Balancing Act':72, 'Lifting Heavy Things':118,'Crystal Caves - Level 2':24, 'Honey Cake':142, 'Crystal Caves - Level 3':19,
        'Heavy, Heavier, Heaviest':61}


# In[ ]:


def init_event_features():
    event_feats = {}
    event_feats['event_code'] = event_code_count.copy() # key: event_code
    event_feats['event_id'] = event_id_count.copy() #key: event_id
    event_feats['title'] = title_count.copy() #key:title
    event_feats['title_event_code'] = title_event_code_count.copy() # key:title_event_code
    #event_feats['last_accuracy_title'] = {'acc_' + title: -1 for title in [title_labels[i] for i in assess_title_set]}
    return event_feats

def init_user_activities_features():
    user_act_feats = {}
    user_act_feats['career_count'] = {'CareerClip':0, 'CareerActivity': 0, 'CareerAssessment': 0, 'CareerGame':0}
    user_act_feats['career_time'] = {'CareerClipTime':0, 'CareerActivityTime':0, 'CareerAssessmentTime':0, 'CareerGameTime':0}
    user_act_feats['career_avg_time'] = {'CareerAVGClipTime':0, 'CareerAVGActivityTime':0,'CareerAVGAssessmentTime':0, 'CareerAVGGameTime':0}
    return user_act_feats

def update_counters(session, event_feats, col):
    counter = event_feats[col]
    num_of_session_count = Counter(session[col])
    for k in num_of_session_count.keys():
        x = k
        if col == 'title':
            x = activities_labels[k]
        counter[x] += num_of_session_count[k]
    return counter

def get_group(accuracy):
    if accuracy <= 0:
        return 0
    elif accuracy ==1:
        return 3
    elif accuracy == 0.5:
        return 2
    else:
        return 1
    
# collect basic features from original data:
def get_context(user_sample, test_set = False):

    # generate raw data chart in train:uid1: session1, session2 ...
    
    # in test: uid: session1, session2 (if is the last session append to final_test df)
    # normal test chart used to get user info and assessment info, and then merge to final test for prediction.

    user_act_feats = init_user_activities_features()
    all_assessments = []
    accumulated_true_attempts = 0
    accumulated_false_attempts = 0
    init = 1
    
    
    # iterate by session, n sessions + assessment is a cycle
    for i, session in user_sample.groupby('game_session', sort = False):
        if init == 1:
            # Cycle restart
            features = {'Clip':0, 'Activity': 0, 'Assessment': 0, 'Game':0}
            features.update({'ClipTime':0, 'ActivityTime':0, 'AssessmentTime':0, 'GameTime':0})
            event_feats = init_event_features()
            accuracy_group = {0:0, 1:0, 2:0, 3:0}
            true_attempts, false_attempts = 0, 0
            init = 0
            
        # session info 
        session_type, session_title  = session['type'].iloc[0], session['title'].iloc[0]
        session_title_text = activities_labels[session_title]
        session_time = (session.iloc[-1,2] - session.iloc[0,2]).seconds
        
        
        features[session_type] +=1
        features[session_type + 'Time'] += session_time
        
        if session_type != 'Assessment':
            user_act_feats['career_time']['Career'+ session_type +'Time'] += session_time
            user_act_feats['career_count']['Career'+ session_type] += 1
        # assessment
        if (session_type == 'Assessment') & (test_set or len(session) > 1):
            all_attempts = session.query(f'event_code == {win_code[session_title]}')
            true_attempts = all_attempts['event_data'].str.contains('true').sum()
            false_attempts = all_attempts['event_data'].str.contains('false').sum()
            # if no attempts, skip.
            if (true_attempts + false_attempts):
                accuracy = true_attempts/(true_attempts + false_attempts)
            elif test_set:
                accuracy = 0
            else:
                init = 1
                continue
            # constructing features
            features.update(event_feats['event_code'])
            features.update(event_feats['event_id'])
            features.update(event_feats['title'])
            features.update(event_feats['title_event_code'])
            features.update(user_act_feats['career_count'])
            features.update(user_act_feats['career_time'])
            features.update(user_act_feats['career_avg_time'])
            features['installation_id'] = session['installation_id'].iloc[-1]
            features['session_title'] = session_title
            features['true_attempts'] = true_attempts
            features['false_attempts'] = false_attempts
            features['accuracy'] = accuracy
            features['accuracy_group'] = get_group(accuracy)
            features['accumulated_true_attempts'] = accumulated_true_attempts
            features['accumulated_false_attempts'] = accumulated_false_attempts
            num = accumulated_true_attempts + accumulated_false_attempts
            features['accumulated_accuracy'] = accumulated_true_attempts/(num-1) if num > 1 else 0

            all_assessments.append(features)
            init = 1
            # update features afterwards to prevent data leakage.
            accumulated_true_attempts += true_attempts
            accumulated_false_attempts += false_attempts
            user_act_feats['career_time']['Career'+ session_type +'Time'] += session_time
            user_act_feats['career_count']['Career'+ session_type] += 1
            user_act_feats['career_avg_time']['CareerAVG'+session_type + 'Time'] = user_act_feats['career_time']['Career'+ session_type +'Time']/user_act_feats['career_count']['Career'+ session_type]
            
        event_feats['event_code'] = update_counters(session, event_feats, 'event_code')    
        event_feats['event_id'] = update_counters(session, event_feats, 'event_id')
        event_feats['title'] = update_counters(session, event_feats, 'title')
        event_feats['title_event_code'] = update_counters(session, event_feats, 'title_event_code')
        
    return all_assessments
    
#def fix_json_colname(df):
    #df.columns = ["".join (c if c.isalnum() else "_" for c in str(x)) for x in df.columns]
    
def get_train_and_test(train, test):
    compiled_train = []
    compiled_test_history = []
    compiled_test = []
    
    for _,(ins_id, user_sample) in tqdm(enumerate(train.groupby('installation_id', sort = False)), total = train['installation_id'].nunique()):
        compiled_train.extend(get_context(user_sample, test_set = False))
        
    for ins_id, user_sample in tqdm(test.groupby('installation_id', sort = False), total = test['installation_id'].nunique()):
        test_data = get_context(user_sample, test_set = True)
        compiled_test_history.extend(test_data[:-1])
        compiled_test.append(test_data[-1])
    
    reduce_train = pd.DataFrame(compiled_train)
    reduce_test = pd.DataFrame(compiled_test)
    reduce_test_history = pd.DataFrame(compiled_test_history)
    
    #fix_json_colname(reduce_train)
    #fix_json_colname(reduce_test)
    #fix_json_colname(reduce_test_history)
    for df in [reduce_train, reduce_test, reduce_test_history]:
        for col in df.columns:
            if df[col].dtypes == np.float64 or df[col].dtypes == np.int64:
                df[col] = df[col].astype(np.float32)
                
    print('reduce_train shape:' + str(reduce_train.shape) + ', reduce_test shape: ' + str(reduce_test.shape) +'.' + 'reduce_test_history shape: ' + str(reduce_test_history.shape))
    print('Totally ' + str(reduce_train['installation_id'].nunique()) + ' users.')
    gc.collect()
    reduce_train_merge = pd.concat([reduce_train, reduce_test_history], axis = 0).reset_index().drop(columns = ['index'])
    categorical = ['session_title']
    # merge reduce_test_history to reduce_train
    return reduce_train_merge, reduce_test, categorical


# In[ ]:


'''# this is the function that convert the raw data into processed features
def get_data(user_sample, test_set=False):
    
    #The user_sample is a DataFrame from train or test where the only one 
    #installation_id is filtered
    #And the test_set parameter is related with the labels processing, that is only requered
    #if test_set=False
    
    # Constants and parameters declaration
    last_activity = 0
    
    user_activities_count = {'Clip':0, 'Activity': 0, 'Assessment': 0, 'Game':0}
    
    # new features: time spent in each activity
    last_session_time_sec = 0
    accuracy_groups = {0:0, 1:0, 2:0, 3:0}
    all_assessments = []
    accumulated_accuracy_group = 0
    accumulated_accuracy = 0
    accumulated_correct_attempts = 0 
    accumulated_uncorrect_attempts = 0
    accumulated_actions = 0
    counter = 0
    time_first_activity = float(user_sample['timestamp'].values[0])
    durations = []
    clip_durations = []
    Activity_durations = []
    Game_durations = []
    
    last_accuracy_title = {'acc_' + title: -1 for title in assess_titles}
    event_code_count: Dict[str, int] = {ev: 0 for ev in list_of_event_code}
    event_id_count: Dict[str, int] = {eve: 0 for eve in list_of_event_id}
    title_count: Dict[str, int] = {eve: 0 for eve in activities_labels.values()} 
    title_event_code_count: Dict[str, int] = {t_eve: 0 for t_eve in all_title_event_code}
        
    # last features
    sessions_count = 0
    
    # itarates through each session of one instalation_id
    for i, session in user_sample.groupby('game_session', sort=False):
        # i = game_session_id
        # session is a DataFrame that contain only one game_session
        
        # get some sessions information
        session_type = session['type'].iloc[0]
        session_title = session['title'].iloc[0]
        session_title_text = activities_labels[session_title]
                    
        if session_type == 'Clip':
            clip_durations.append((clip_time[activities_labels[session_title]]))
        
        if session_type == 'Activity':
            Activity_durations.append((session.iloc[-1, 2] - session.iloc[0, 2] ).seconds)
        
        if session_type == 'Game':
            Game_durations.append((session.iloc[-1, 2] - session.iloc[0, 2] ).seconds)
            
        # for each assessment, and only this kind off session, the features below are processed
        # and a register are generated
        if (session_type == 'Assessment') & (test_set or len(session)>1):
            # search for event_code 4100, that represents the assessments trial
            all_attempts = session.query(f'event_code == {win_code[session_title]}')
            # then, check the numbers of wins and the number of losses
            true_attempts = all_attempts['event_data'].str.contains('true').sum()
            false_attempts = all_attempts['event_data'].str.contains('false').sum()
            # copy a dict to use as feature template, it's initialized with some itens: 
            # {'Clip':0, 'Activity': 0, 'Assessment': 0, 'Game':0}
            features = user_activities_count.copy()
            features.update(last_accuracy_title.copy())
            features.update(event_code_count.copy())
            features.update(event_id_count.copy())
            features.update(title_count.copy())
            features.update(title_event_code_count.copy())
            features.update(last_accuracy_title.copy())
            features['installation_session_count'] = sessions_count
#             features['hour'] = session['hour'].iloc[-1]
            
            variety_features = [('var_event_code', event_code_count),
                              ('var_event_id', event_id_count),
                               ('var_title', title_count),
                               ('var_title_event_code', title_event_code_count)]
            
            for name, dict_counts in variety_features:
                arr = np.array(list(dict_counts.values()))
                features[name] = np.count_nonzero(arr)
                 
            # get installation_id for aggregated features
            features['installation_id'] = session['installation_id'].iloc[-1]
            # add title as feature, remembering that title represents the name of the game
            features['session_title'] = session['title'].iloc[0]
            # the 4 lines below add the feature of the history of the trials of this player
            # this is based on the all time attempts so far, at the moment of this assessment
            features['accumulated_correct_attempts'] = accumulated_correct_attempts
            features['accumulated_uncorrect_attempts'] = accumulated_uncorrect_attempts
            accumulated_correct_attempts += true_attempts 
            accumulated_uncorrect_attempts += false_attempts
            # the time spent in the app so far
            if durations == []:
                features['duration_mean'] = 0
                features['duration_std'] = 0
            else:
                features['duration_mean'] = np.mean(durations)
                features['duration_std'] = np.std(durations)
                
            if clip_durations == []:
                features['Clip_duration_mean'] = 0
                features['Clip_duration_std'] = 0
            else:
                features['Clip_duration_mean'] = np.mean(clip_durations)
                features['Clip_duration_std'] = np.std(clip_durations)
                
            if Activity_durations == []:
                features['Activity_duration_mean'] = 0
                features['Activity_duration_std'] = 0
            else:
                features['Activity_duration_mean'] = np.mean(Activity_durations)
                features['Activity_duration_std'] = np.std(Activity_durations)
                
            if Game_durations == []:
                features['Game_duration_mean'] = 0
                features['Game_duration_std'] = 0
            else:
                features['Game_duration_mean'] = np.mean(Game_durations)
                features['Game_duration_std'] = np.std(Game_durations)
                
            durations.append((session.iloc[-1, 2] - session.iloc[0, 2] ).seconds)
            # the accurace is the all time wins divided by the all time attempts
            features['accumulated_accuracy'] = accumulated_accuracy/counter if counter > 0 else 0
            accuracy = true_attempts/(true_attempts+false_attempts) if (true_attempts+false_attempts) != 0 else 0
            accumulated_accuracy += accuracy
            last_accuracy_title['acc_' + session_title_text] = accuracy
            # a feature of the current accuracy categorized
            # it is a counter of how many times this player was in each accuracy group
            if accuracy == 0:
                features['accuracy_group'] = 0
            elif accuracy == 1:
                features['accuracy_group'] = 3
            elif accuracy == 0.5:
                features['accuracy_group'] = 2
            else:
                features['accuracy_group'] = 1
            features.update(accuracy_groups)
            accuracy_groups[features['accuracy_group']] += 1
            # mean of the all accuracy groups of this player
            features['accumulated_accuracy_group'] = accumulated_accuracy_group/counter if counter > 0 else 0
            accumulated_accuracy_group += features['accuracy_group']
            # how many actions the player has done so far, it is initialized as 0 and updated some lines below
            features['accumulated_actions'] = accumulated_actions
            
            # there are some conditions to allow this features to be inserted in the datasets
            # if it's a test set, all sessions belong to the final dataset
            # it it's a train, needs to be passed throught this clausule: session.query(f'event_code == {win_code[session_title]}')
            # that means, must exist an event_code 4100 or 4110
            if test_set:
                all_assessments.append(features)
            elif true_attempts+false_attempts > 0:
                all_assessments.append(features)
                
            counter += 1
        
        sessions_count += 1
        # this piece counts how many actions was made in each event_code so far
        def update_counters(counter: dict, col: str):
                num_of_session_count = Counter(session[col])
                for k in num_of_session_count.keys():
                    x = k
                    if col == 'title':
                        x = activities_labels[k]
                    counter[x] += num_of_session_count[k]
                return counter
            
        event_code_count = update_counters(event_code_count, "event_code")
        event_id_count = update_counters(event_id_count, "event_id")
        title_count = update_counters(title_count, 'title')
        title_event_code_count = update_counters(title_event_code_count, 'title_event_code')

        # counts how many actions the player has done so far, used in the feature of the same name
        accumulated_actions += len(session)
        if last_activity != session_type:
            user_activities_count[session_type] += 1
            last_activitiy = session_type 
                        
    # if it't the test_set, only the last assessment must be predicted, the previous are scraped
    if test_set:
        return all_assessments[-1]
    # in the train_set, all assessments goes to the dataset
   
    return all_assessments
import random

def get_train_and_test(train, test):
    compiled_train = []
    compiled_test = []
    for i, (ins_id, user_sample) in tqdm(enumerate(train.groupby('installation_id', sort = False)), total = 17000):
        t = get_data(user_sample)
        for xt in range(len(t)):
            if xt != len(t) - 1:
                if random.random() >= 0.25:
                    compiled_train.append(t[xt])
            else:
                compiled_train.append(t[xt])
    for ins_id, user_sample in tqdm(test.groupby('installation_id', sort = False), total = 1000):
        test_data = get_data(user_sample, test_set = True)
        compiled_test.append(test_data)
    reduce_train = pd.DataFrame(compiled_train)
    reduce_test = pd.DataFrame(compiled_test)
    categoricals = ['session_title']
    return reduce_train, reduce_test, categoricals'''


# In[ ]:


def time_sequence_kfolds_split2(df, k, random_state):
    random.seed(random_state)
    one_shot_ids = []
    train_index = []
    test_index = []
    for ins_id, uid in df.groupby('installation_id', sort = False):
        shape = uid.shape
        if shape[0] > k: # if number of records is larger than k
            for n in range(k): 
                train_index.extend(uid.iloc[:-1].index.values.tolist())
                test_index.extend(uid.iloc[-1:].index.values.tolist())
        elif shape[0] == 1:
            one_shot_ids.extend(uid.index.values.tolist())
    for i in range(k):
        random_train = random.sample(train_index, int(.8*len(train_index)))
        random_test = random.sample(test_index, int(.8*len(test_index)))
        yield random_train, random_test     


# In[ ]:


import heapq
class Base_Model(object):
    def __init__(self, train_df, test_df, features, categoricals=[], n_best = 4, n_splits = 5, verbose=True):
        self.train_df = train_df
        self.test_df = test_df
        self.n_best = n_best
        print('train data shape: (%d, %d)'%(train_df.shape[0], train_df.shape[1]))
        print('test data shape: (%d, %d)'%(test_df.shape[0], test_df.shape[1]))
        self.features = features
        self.n_splits = n_splits
        self.categoricals = categoricals
        self.target = 'accuracy_group'
        self.cv = self.get_cv()
        self.verbose = verbose
        self.params = self.get_params()
        self.y_pred, self.score, self.model = self.fit()
        
    def train_model(self, train_set, val_set):
        raise NotImplementedError
        
    def get_cv(self):
        #cv = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=42)
        return time_sequence_kfolds_split2(self.train_df, k = self.n_splits, random_state = int(round(random.random())))
    
    def get_params(self):
        raise NotImplementedError
        
    def convert_dataset(self, x_train, y_train, x_val, y_val):
        raise NotImplementedError
        
    def convert_x(self, x):
        return x
        
    def fit(self):
        oof_pred = np.zeros((len(reduce_train), ))
        y_pred = np.zeros((len(reduce_test), ))
        y_pred_list = []
        score_square_sum = 0
        score_sum = 0
        
              
              
        for fold, (train_idx, val_idx) in enumerate(self.cv):
            x_train, x_val = self.train_df[self.features].iloc[train_idx], self.train_df[self.features].iloc[val_idx]
            y_train, y_val = self.train_df[self.target][train_idx], self.train_df[self.target][val_idx]
            train_set, val_set = self.convert_dataset(x_train, y_train, x_val, y_val)
            model = self.train_model(train_set, val_set)
            conv_x_val = self.convert_x(x_val)
            
            oof_pred[val_idx] = model.predict(conv_x_val).reshape(oof_pred[val_idx].shape)
            x_test = self.convert_x(self.test_df[self.features])
            partial_score = eval_qwk_lgb_regr(y_val, oof_pred[val_idx])[1]
            y_pred_list.append((partial_score, model.predict(x_test).reshape(y_pred.shape)))
            score_sum += partial_score/self.n_splits
            score_square_sum += (partial_score)**2
            print('Partial score of fold {} is: {}'.format(fold, partial_score))
            
        score_sum_std = (score_square_sum - score_sum**2)/((self.n_splits - 1)*self.n_splits)
        
        best_score = 0
        best_score_square = 0
        for i, (score, y_pred_ele) in enumerate(heapq.nlargest(self.n_best, y_pred_list)):
            best_score+= score/self.n_best
            best_score_square += score**2
            y_pred += y_pred_ele/self.n_best
        
        best_score_std = (best_score_square - best_score**2)/((self.n_best -1)*self.n_best)
        
        _, loss_score, _ = eval_qwk_lgb_regr(self.train_df[self.target], oof_pred)
        if self.verbose:
            print('Average partial score is: ' +str(score_sum) +' +/- ' +str(score_sum_std) + 'std')
            print('Best n partial score is: ' + str(best_score) + '+/- ' + str(best_score_std) + 'std')
    
        return y_pred, best_score, model
    
class Lgb_Model(Base_Model):
    
    def train_model(self, train_set, val_set):
        verbosity = 100 if self.verbose else 0
        return lgb.train(self.params, train_set, valid_sets=[train_set, val_set], verbose_eval=verbosity)
        
    def convert_dataset(self, x_train, y_train, x_val, y_val):
        train_set = lgb.Dataset(x_train, y_train, categorical_feature=self.categoricals)
        val_set = lgb.Dataset(x_val, y_val, categorical_feature=self.categoricals)
        return train_set, val_set
        
    def get_params(self):
        objective = 'tweedie' #poisson, tweebie, regression(rmse/mae)
        params = {'n_estimators':5000,
                    'boosting_type': 'gbdt',
                    'objective': 'tweedie',
                    'metric': 'tweedie',
                    'subsample': 0.66,
                    'num_leaves': 236,
                    'subsample_freq': 1,
                    'learning_rate': 0.02,
                    'feature_fraction': 0.9,
                    'max_depth': 16,
                    'lambda_l1': .02,  
                    'lambda_l2': .8,
                    'early_stopping_rounds': 100,
                    'verbose': 0
                    }
        return params
    
class Xgb_Model(Base_Model):
    
    def train_model(self, train_set, val_set):
        verbosity = 100 if self.verbose else 0
        return xgb.train(self.params, train_set, 
                         num_boost_round=5000, evals=[(train_set, 'train'), (val_set, 'val')], 
                         verbose_eval=verbosity, early_stopping_rounds=100)
        
    def convert_dataset(self, x_train, y_train, x_val, y_val):
        train_set = xgb.DMatrix(x_train, y_train)
        val_set = xgb.DMatrix(x_val, y_val)
        return train_set, val_set
    
    def convert_x(self, x):
        return xgb.DMatrix(x)
        
    def get_params(self):
        params = {'colsample_bytree': 0.8,                 
            'learning_rate': 0.02,
            'max_depth': 10,
            'subsample': 1,
            'objective':'reg:squarederror',
            #'eval_metric':'rmse',
            'min_child_weight':3,
            'gamma':0.25,
            'n_estimators':5000}
        return params
    
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
class Nn_Model(Base_Model):
    def __init__(self, train_df, test_df, features, categoricals=[], n_splits=10, verbose=True):
        features = features.copy()
        if len(categoricals) > 0:
            for cat in categoricals:
                enc = OneHotEncoder()
                train_cats = enc.fit_transform(train_df[[cat]])
                test_cats = enc.transform(test_df[[cat]])
                cat_cols = ['{}_{}'.format(cat, str(col)) for col in enc.active_features_]
                features += cat_cols
                train_cats = pd.DataFrame(train_cats.toarray(), columns=cat_cols)
                test_cats = pd.DataFrame(test_cats.toarray(), columns=cat_cols)
                train_df = pd.concat([train_df, train_cats], axis=1)
                test_df = pd.concat([test_df, test_cats], axis=1)
        scalar = MinMaxScaler()
        train_df[features] = scalar.fit_transform(train_df[features])
        test_df[features] = scalar.transform(test_df[features])
        print(train_df[features].shape)
        super().__init__(train_df, test_df, features, categoricals, n_splits, verbose)
        
    def train_model(self, train_set, val_set):
        verbosity = 100 if self.verbose else 0
        model = tf.keras.models.Sequential([
            tf.keras.layers.Input(shape=(train_set['X'].shape[1],)),
            tf.keras.layers.Dense(200, activation='relu'),
            tf.keras.layers.LayerNormalization(),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(100, activation='relu'),
            tf.keras.layers.LayerNormalization(),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(50, activation='relu'),
            tf.keras.layers.LayerNormalization(),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(25, activation='relu'),
            tf.keras.layers.LayerNormalization(),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(1, activation='relu')
        ])
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=4e-4), loss='mse')
        print(model.summary())
        save_best = tf.keras.callbacks.ModelCheckpoint('nn_model.w8', save_weights_only=True, save_best_only=True, verbose=1)
        early_stop = tf.keras.callbacks.EarlyStopping(patience=20)
        model.fit(train_set['X'], 
                train_set['y'], 
                validation_data=(val_set['X'], val_set['y']),
                epochs=100,
                 callbacks=[save_best, early_stop])
        model.load_weights('nn_model.w8')
        return model
        
    def convert_dataset(self, x_train, y_train, x_val, y_val):
        train_set = {'X': x_train, 'y': y_train}
        val_set = {'X': x_val, 'y': y_val}
        return train_set, val_set
        
    def get_params(self):
        return None    


# In[ ]:


from __future__ import absolute_import, division, print_function, unicode_literals
from random import choice
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Model, Sequential
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer,Dense
from tensorflow.keras.optimizers import Adam,SGD
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow import metrics
from sklearn.decomposition import PCA
K.clear_session()

class FactorizationMachine(Layer):
    def __init__(self, params):
        super (FactorizationMachine, self).__init__()
        self.params = params
        self.v_shape, self.v2_shape = self.params['v_shape'], self.params['v2_shape']
        self.first_order_dense = Dense(1, activation = 'relu')
        #self.output_dense = Dense(1, activatinon = 'relu')
        
    def build(self, input_shape):
        shape1, shape2 = input_shape
        self.V = self.add_variable('V', shape = [int(shape2[-1]), self.v_shape])

    def call(self, input):
        '''
        input: stacked embeddings with shape (batch,fields,embdim)
        '''
        x_train, embedding = input
        squared_of_sum = tf.reduce_sum(tf.square(tf.matmul(embedding, self.V)), axis = -1) # (b,f,e) * (e,k) ->(b,f,k) ->b,f
        #print(self.squared_of_sum.shape)
        sum_of_squared = tf.reduce_sum(tf.matmul(tf.square(embedding), tf.square(self.V)), axis = -1) # (b,f,e) * (e,k) ->(b,f,k) ->b,f
        second_order = tf.reduce_sum(0.5 * tf.subtract(squared_of_sum, sum_of_squared), axis = -1)
        #devider_order = tf.reduce_sum(0.5 * tf.subtract(squared_of_sum, sum_of_squared), axis = -1)
        first_order = tf.reduce_sum(self.first_order_dense(x_train), axis = -1)
        return second_order + first_order, tf.concat([tf.reshape(second_order, (-1,1)),  tf.reshape(first_order, (-1,1))], axis = 1)
    
class DFM_Model(Base_Model):
    
    def __init__(self, train_df, test_df, features, categoricals=[], n_splits=5, verbose=True):
        features = features.copy()
        if categoricals:
            for cat in categoricals:
                enc = OneHotEncoder()
                train_cats = enc.fit_transform(train_df[[cat]])
                test_cats = enc.transform(test_df[[cat]])
                self.cat_cols = ['{}_{}'.format(cat, str(col)) for col in enc.active_features_]
                features += self.cat_cols
                train_cats = pd.DataFrame(train_cats.toarray(), columns = self.cat_cols)
                test_cats = pd.DataFrame(test_cats.toarray(), columns = self.cat_cols)
                train_df = pd.concat([train_df, train_cats], axis=1)
                test_df = pd.concat([test_df, test_cats], axis=1)
        scaler = MinMaxScaler()
        train_df[features] = scaler.fit_transform(train_df[features])
        test_df[features] = scaler.transform(test_df[features])
        self.look_up = self.get_lookup(train_df[features].reset_index().drop(columns = 'index'))
        super().__init__(train_df, test_df, features, categoricals, n_splits, verbose)
        
    def get_lookup(self, df):
        fields = {} # embedding fields in dnnfm
        fields['event_code'] = list(event_code_count.keys()) # key: event_code
        fields['event_id'] = list(event_id_count.keys()) #key: event_id
        fields['title'] = list(title_count.keys()) #key:title
        fields['title_event_code'] = list(title_event_code_count.keys()) # key:title_event_code
        fields['session_title'] = self.cat_cols
        fields_list = []
        for k, v in fields.items():
            if len(v) < 4:
                continue
            idx = [df.columns.get_loc(c) for c in v if c in df.columns]
            fields[k] = idx
            fields_list.extend(v)
        #raw_list = self.reduce_train.columns.tolist()
        #fields['linear'] = [self.reduce_train.columns.get_loc(c) for c in raw_list if c not in fields_list]
        return fields
    
    def get_params(self):
        params = {'fmlr': 4e-4,
                    'dfmlr': 4e-4,
                    'v_shape': 8,
                    'v2_shape': 8,
                    #'batch_size': 128,
                    'emb_dim': 65,
                    'dnn_dim': 89,
                    'l1': .41,
                    'l2': .27,
                    'fm_l1': .14,
                    'fm_l2': .95,
                    'epoch':1000,
                    'objective':'mse',
                    'metrics':['mse'],
                    }
        return params
    
    def convert_dataset(self, x_train, y_train, x_val, y_val):
        train_set = {'X': x_train.values, 'y': y_train.values}
        val_set = {'X': x_val.values, 'y': y_val.values}
        return train_set, val_set
    
    def train_model(self, train_set, val_set):
        x_in = tf.keras.layers.Input(shape = (train_set['X'].shape[1],))
        #print(train_set['X'].shape)
        embedding_list = []
        fm_list = []
        b_l = tf.shape(x_in)[0]
        for k,v in self.look_up.items():
            if len(set(v)) < 4:
                continue
            begin = min(v)
            length = max(v) - min(v) +1
            field = tf.slice(x_in, [0, begin], [b_l, length]) # (b, n)
            embedding = Dense(self.params['emb_dim'], activation = 'relu')(field)
            #if k != 'linear':
            fm_list.append(embedding)
            embedding_list.append(embedding)
            
        fm_in = tf.stack(fm_list, axis = 1) #(b,f,e)
        dnn_in = tf.concat(embedding_list, axis = 1) #(b, f*e)
        fm_layer = FactorizationMachine(self.params)
        fm_out, fm_out_2 = fm_layer((x_in, fm_in))
        
        dnn1 = Dense(self.params['dnn_dim'], 'relu', kernel_regularizer=keras.regularizers.l1_l2(l1 = self.params['l1'], l2 = self.params['l2']))(dnn_in)
        dnn1 = tf.keras.layers.Dropout(0.2)(dnn1)
        dnn2 = Dense(self.params['dnn_dim'], 'relu', kernel_regularizer=keras.regularizers.l1_l2(l1 = self.params['l1'], l2 = self.params['l2']))(dnn1)
        dnn2 = tf.keras.layers.Dropout(0.2)(dnn2)
        dnn3 = Dense(self.params['dnn_dim'], 'relu', kernel_regularizer=keras.regularizers.l1_l2(l1 = self.params['l1'], l2 = self.params['l2']))(dnn2)
        dnn3 = tf.keras.layers.Dropout(0.2)(dnn3)
        dnn_out = Dense(self.params['dnn_dim'], 'relu', kernel_regularizer=keras.regularizers.l1_l2(l1 = self.params['l1'], l2 = self.params['l2']))(dnn3)
        final_out = Dense(1, kernel_regularizer=keras.regularizers.l1_l2(l1 = self.params['l1'], l2 = self.params['l2']))(tf.concat([dnn_out, fm_out_2], axis = 1))
        
        self.fm_model = Model(inputs = x_in, outputs = fm_out)
        self.dfm_model = Model(inputs = x_in, outputs = final_out) 
        opt = Adam(lr = self.params['fmlr'], amsgrad=True)
        self.fm_model.compile(optimizer = opt, loss = self.params['objective'], metrics = self.params['metrics'])
        opt2 = Adam(lr = self.params['dfmlr'], amsgrad=True)
        self.dfm_model.compile(optimizer = opt2, loss = self.params['objective'], metrics = self.params['metrics'])
        
        save_best1 = tf.keras.callbacks.ModelCheckpoint('fm_model.w8', save_weights_only=True, save_best_only=True, verbose = 1)
        save_best2 = tf.keras.callbacks.ModelCheckpoint('dfm_model.w8', save_weights_only=True, save_best_only=True, verbose = 1)
        early_stop = tf.keras.callbacks.EarlyStopping(patience=20)
        print(self.fm_model.summary())
        self.fm_model.fit(train_set['X'],
                     train_set['y'],
                     validation_data = (val_set['X'], val_set['y']),
                     epochs = self.params['epoch'],
                     callbacks = [save_best1, early_stop],
                     verbose = 1)
        self.fm_model.load_weights('fm_model.w8')
        print(self.dfm_model.summary())
        self.dfm_model.fit(train_set['X'], 
                      train_set['y'], 
                     validation_data=(val_set['X'], val_set['y']),
                     epochs = self.params['epoch'],
                     callbacks = [save_best2, early_stop],
                     verbose = 1)
        self.dfm_model.load_weights('dfm_model.w8')
        
        return self.dfm_model
        


# In[ ]:


# read data
train, test, train_labels, specs, sample_submission = read_data()
# get usefull dict with maping encode
train, test, train_labels, win_code, list_of_user_activities, list_of_event_code, activities_labels, assess_titles, list_of_event_id, all_title_event_code = encode_title(train, test, train_labels)
# tranform function to get the train and test set
# event features dict:
event_code_count = {ev:0 for ev in list_of_event_code}
event_id_count = {ev:0 for ev in list_of_event_id}
title_count = {ev:0 for ev in activities_labels.values()}
title_event_code_count = {t_ev:0 for t_ev in all_title_event_code}
reduce_train, reduce_test, categoricals = get_train_and_test(train, test)


# In[ ]:


def stract_hists(feature, train=reduce_train, test=reduce_test, adjust=False, plot=False):
    n_bins = 10
    train_data = train[feature]
    test_data = test[feature]
    if adjust:
        test_data *= train_data.mean() / test_data.mean()
    perc_90 = np.percentile(train_data, 95)
    train_data = np.clip(train_data, 0, perc_90)
    test_data = np.clip(test_data, 0, perc_90)
    train_hist = np.histogram(train_data, bins=n_bins)[0] / len(train_data)
    test_hist = np.histogram(test_data, bins=n_bins)[0] / len(test_data)
    msre = mean_squared_error(train_hist, test_hist)
    if plot:
        print(msre)
        plt.bar(range(n_bins), train_hist, color='blue', alpha=0.5)
        plt.bar(range(n_bins), test_hist, color='red', alpha=0.5)
        plt.show()
    return msre


# In[ ]:


# call feature engineering function
features = reduce_train.loc[(reduce_train.sum(axis=1) != 0), (reduce_train.sum(axis=0) != 0)].columns # delete useless columns
features = [x for x in features if x not in ['accuracy_group', 'installation_id']]


# In[ ]:


to_exclude = [] 
ajusted_test = reduce_test.copy()
for feature in ajusted_test.columns:
    if feature not in ['accuracy_group', 'installation_id', 'accuracy_group', 'session_title']:
        data = reduce_train[feature]
        train_mean = data.mean()
        data = ajusted_test[feature] 
        test_mean = data.mean()
        try:
            error = stract_hists(feature, adjust=True)
            ajust_factor = train_mean / test_mean
            if ajust_factor > 10 or ajust_factor < 0.1:# or error > 0.01:
                to_exclude.append(feature)
                print(feature, train_mean, test_mean, error)
            else:
                ajusted_test[feature] *= ajust_factor
        except:
            to_exclude.append(feature)
            print(feature, train_mean, test_mean)


# In[ ]:


features = [x for x in features if x not in to_exclude]
reduce_train[features].shape


# In[ ]:


import seaborn as sns
def plot_feature_importance(features, importances):
    d = {}
    d['features'] = features
    d['importance'] = importances
    important_features = pd.DataFrame(d)
    plt.figure(figsize = (12,15))
    important_features = important_features.groupby('features')['importance'].mean().reset_index().sort_values('importance')
    if len(features) < 80:
        n = len(features)-1
    else:
        n = 80
    sns.barplot(important_features['importance'][-n:], important_features['features'][-n:])
    return important_features


# In[ ]:


'''
Original Data:
poisson
Average partial score is: 0.5623904605653245 +/- 0.03164414073670059std
Best n partial score is: 0.5728333816362448+/- 0.06563117983605811std
tweedie/tweedie:
Average partial score is: 0.5657056330290424 +/- 0.0320165459609263std
Best n partial score is: 0.5742538311494599+/- 0.06597108256274661std
Average partial score is: 0.6030431598354409 +/- 0.03639031569695544std
Best n partial score is: 0.6138975396315688+/- 0.0753835155343371std

MyData
regression+rmse
Average partial score is: 0.5533183263386379
Best n partial score is: 0.5598111624907085
poisson + rmse:
Average partial score is: 0.5664034221359218
Best n partial score is: 0.5716126280759017 
poisson + poisson:
Average partial score is: 0.5627788516987987 +/- 0.031678317312866584std
Best n partial score is: 0.5689541833964443+/- 0.06474670116987956std
tweedie + tweedie:
Average partial score is: 0.5660214276251782 +/- 0.032042277869784405std
Best n partial score is: 0.5712556169932934+/- 0.0652691109376978std
'''
lgb_model = Lgb_Model(reduce_train, ajusted_test, features, categoricals=categoricals)
plot_feature_importance(features, lgb_model.model.feature_importance())


# In[ ]:


'''
original dataset
Average partial score is: 0.5478419481588803 +/- 0.030029940362635992std
Best n partial score is: 0.5576519402978031+/- 0.0622026002160719std
my dataset:

'''
xgb_model = Xgb_Model(reduce_train, ajusted_test, features, categoricals=categoricals)


# In[ ]:


nn_model = Nn_Model(reduce_train, ajusted_test, features, categoricals=categoricals)


# In[ ]:


#dfm_model = DFM_Model(reduce_train, ajusted_test, features, categoricals=categoricals)


# In[ ]:


weights = {'lbg': .6, 'xgb': 0.25, 'nn': 0.15}
#weights = {'lbg': 1, 'xgb': 0, 'nn': 0}
final_pred = (lgb_model.y_pred * weights['lbg'])
             + (xgb_model.y_pred * weights['xgb'])
             + (nn_model.y_pred * weights['nn'])
print(final_pred.shape)


# In[ ]:


dist = Counter(reduce_train['accuracy_group'])
for k in dist:
    dist[k] /= len(reduce_train)
reduce_train['accuracy_group'].hist()

acum = 0
bound = {}
for i in range(3):
    acum += dist[i]
    bound[i] = np.percentile(final_pred, acum * 100)
    
print('bound =', bound)

def classify(x):
    if x <= bound[0]:
        return 0
    elif x <= bound[1]:
        return 1
    elif x <= bound[2]:
        return 2
    else:
        return 3
    
final_pred = np.array(list(map(classify, final_pred)))

sample_submission['accuracy_group'] = final_pred.astype(int)
sample_submission.to_csv('submission.csv', index=False)
sample_submission['accuracy_group'].value_counts(normalize=True)

