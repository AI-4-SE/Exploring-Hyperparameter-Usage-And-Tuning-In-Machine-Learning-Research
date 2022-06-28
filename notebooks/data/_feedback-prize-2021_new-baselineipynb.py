#!/usr/bin/env python
# coding: utf-8

# In[ ]:


ls -l ../input


# In[ ]:


get_ipython().system('pip install ../input/pipwheels/tokenizers-0.10.3-cp37-cp37m-manylinux_2_5_x86_64.manylinux1_x86_64.manylinux_2_12_x86_64.manylinux2010_x86_64.whl')
get_ipython().system('pip install ../input/pipwheels/sentencepiece-0.1.96-cp37-cp37m-manylinux_2_17_x86_64.manylinux2014_x86_64.whl')
get_ipython().system('pip install ../input/pipwheels/transformers-4.16.0.dev0-py3-none-any.whl')
get_ipython().system('pip install ../input/pipwheels/torch_scatter-2.0.9-cp37-cp37m-linux_x86_64.whl')


# In[ ]:


get_ipython().system('cp -r ../input/feedbackrepo/kaggle-feedback-clean/* ./')


# In[ ]:


import numpy as np
import pandas as pd
import scipy as sp
import os
import json
import sys
import importlib
import multiprocessing as mp
import gc
from tqdm.auto import tqdm
import glob
import torch
from copy import copy
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
import torch_scatter
import pickle
import collections


# In[ ]:


torch.backends.cudnn.benchmark = True

sys.path.append('./configs')
sys.path.append('./data')
sys.path.append('./models')
sys.path.append('./scripts')
sys.path.append('./blend')


# In[ ]:


FOLD = 0

cache_allowed = False
THRESHOLD = True
nposn = 3000
NMODELS = 1


# In[ ]:


ppcfg = importlib.import_module('cfg_dh_12G').cfg
map_clip = ppcfg.map_clip_init
load_configs = ppcfg.load_configs.split()
model_weights = [ppcfg.baseline_weights[c] for c in load_configs]
start_threshold = ppcfg.baseline_start_threshold
position_proba_threshold = ppcfg.baseline_position_proba_threshold
name_map = ppcfg.name_map
proba_thresh = ppcfg.proba_thresh
print(f'load order \n{load_configs}\n')
print(f'map clip \n{json.dumps(map_clip, indent = 4)}\n')
print(f'model weights \n{json.dumps(ppcfg.baseline_weights, indent = 4)}\n')
print(f'model weights order \n{json.dumps(model_weights, indent = 4)}\n')
print(f'name map \n{json.dumps(name_map, indent = 4)}\n')
print(f'proba thresh \n{json.dumps(proba_thresh, indent = 4)}\n')
print(f'start threshold \n{start_threshold}\n')
print(f'position_proba_threshold \n{position_proba_threshold}\n')


# In[ ]:


COMP_FOLDER = '../input/feedback-prize-2021/'

train = pd.read_csv(COMP_FOLDER + 'train.csv')
SAMPLE_SUBMISSION = pd.read_csv(COMP_FOLDER + 'sample_submission.csv')
N_CORES = mp.cpu_count()

RAM_CHECK = False
OOF_CHECK = False

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
PUBLIC_RUN = len(SAMPLE_SUBMISSION) == 5
TEST_FOLDER = COMP_FOLDER + 'test/'

ids = SAMPLE_SUBMISSION["id"].unique()

print(SAMPLE_SUBMISSION.shape)


# In[ ]:


def get_cfg(CFG):
    cfg = importlib.import_module('default_config')
    importlib.reload(cfg)
    cfg = importlib.import_module(CFG)
    importlib.reload(cfg)
    cfg = copy(cfg.cfg)
    print(CFG, cfg.model, cfg.dataset, cfg.backbone, cfg.tokenizer, cfg.pretrained_weights)

    cfg.data_dir = COMP_FOLDER
    cfg.tokenizer = '../input/feedback-huggingface-models/' + cfg.tokenizer
    cfg.backbone = '../input/feedback-huggingface-models/' + cfg.backbone
    cfg.data_folder = TEST_FOLDER
    cfg.pretrained = False
    cfg.pretrained_weights = False
    cfg.batch_size = 1
    cfg.offline_inference = True
    return cfg
    
def get_dl(cfg):
    ds = importlib.import_module(cfg.dataset)
    importlib.reload(ds)

    CustomDataset = ds.CustomDataset
    batch_to_device = ds.batch_to_device
    val_collate_fn = ds.val_collate_fn

    test_ds = CustomDataset(SAMPLE_SUBMISSION, cfg, cfg.val_aug, mode="test")
    test_dl = DataLoader(
        test_ds,
        shuffle=False,
        batch_size=cfg.batch_size,
        num_workers=N_CORES,
        pin_memory=True,
        collate_fn=val_collate_fn
    )

    return test_dl, batch_to_device

def get_state_dict(sd_fp):
    sd = torch.load(sd_fp, map_location="cpu")#['model']
    sd = {k.replace("module.", ""):v for k,v in sd.items()}
    return sd

def get_nets(cfg,state_dicts,test_ds, regenerate_pos_embeddings=False):
    model = importlib.import_module(cfg.model)
    importlib.reload(model)
    Net = model.Net

    nets = []

    for i,state_dict in enumerate(state_dicts):
        net = Net(cfg).eval().to(DEVICE)
        print("loading dict")
        sd = get_state_dict(state_dict)
        if "model" in sd.keys():
            sd = sd["model"]
        if regenerate_pos_embeddings:
            sd.update({'backbone.embeddings.position_ids':torch.arange(cfg.max_length).unsqueeze(0)})
        net.load_state_dict(sd, strict=True)
        net.return_preds = True
        nets += [net.half()]
        del sd
        gc.collect()
    return nets


# In[ ]:


def get_preds(name, FOLD, cfg, state_dict_fps, cache_allowed=False, regenerate_pos_embeddings=False):
    cache_dir = "./cache/"
    os.makedirs(cache_dir, exist_ok=True)
    
    if os.path.isfile(f'{cache_dir}class_preds_word_{name}_fold{FOLD}.data') and os.path.isfile(f'{cache_dir}sep_preds_word_{name}_fold{FOLD}.data') and cache_allowed:
        print(f"{name} cached preds exist, loading from cache ...")

        with open(f"{cache_dir}class_preds_word_{name}_fold{FOLD}.data", "rb") as filehandle:
            # read the data as binary data stream
            class_preds_word = pickle.load(filehandle)

        with open(f"{cache_dir}sep_preds_word_{name}_fold{FOLD}.data", "rb") as filehandle:
            # read the data as binary data stream
            sep_preds_word = pickle.load(filehandle)
    else:
        print(f"{name} cached preds do not exist, running inference ...")
        
        test_dl, batch_to_device = get_dl(cfg)

        if OOF_CHECK:
            nets = get_nets(cfg, [state_dict_fps[FOLD]], test_dl.dataset, regenerate_pos_embeddings=regenerate_pos_embeddings)
        else:
            nets = get_nets(cfg, state_dict_fps, test_dl.dataset, regenerate_pos_embeddings=regenerate_pos_embeddings)

        class_preds_word = []
        sep_preds_word = []

        with torch.inference_mode():
            for batch in tqdm(test_dl):
                batch = batch_to_device(batch,DEVICE)

                outs = [net(batch) for net in nets] 
                class_preds = torch.stack([out['class_preds'] for out in outs]).mean(0)
                sep_preds = torch.stack([out['sep_preds'] for out in outs]).mean(0).sigmoid()
                wordposition = outs[0]['wordposition']
                attention_mask = outs[0]["attention_mask"]
                class_preds[:, :, 8][attention_mask == 0] = 5000

                for i in range(class_preds.shape[0]):

                    class_pred = class_preds[i]
                    sep_pred = sep_preds[i]

                    word_pos = wordposition[i].contiguous()
                    word_pos[word_pos==-1] = word_pos.max() + 1
                    class_pred_word = torch_scatter.scatter_mean(class_pred.permute(1,0),word_pos).permute(1,0)[:-1]
                    sep_pred_word, _ = torch_scatter.scatter_max(sep_pred.permute(1,0),word_pos)
                    sep_pred_word = sep_pred_word.permute(1,0)[:-1]
                    class_preds_word += [class_pred_word.cpu()]
                    sep_preds_word += [sep_pred_word.cpu()]


        del nets, test_dl
        gc.collect()
        torch.cuda.empty_cache()

        with open(f'{cache_dir}class_preds_word_{name}_fold{FOLD}.data', 'wb') as filehandle:
            # store the data as binary data stream
            pickle.dump(class_preds_word, filehandle)

        with open(f'{cache_dir}sep_preds_word_{name}_fold{FOLD}.data', 'wb') as filehandle:
            # store the data as binary data stream
            pickle.dump(sep_preds_word, filehandle)

    return class_preds_word, sep_preds_word


# In[ ]:


class_preds_word_blend = []
sep_preds_word_blend = []
namecheck = []


# In[ ]:


name = 'cfg_ch_10c'
namecheck.append(name)
seeds = [245426]  # , 369687 946118
cfg = get_cfg(name)
cfg.dataset = "ds_ch_4_var_tok_fix"
cfg.model = "mdl_ch_2d"
cfg.max_length = 4096
cfg.max_length_test = 4096

state_dict_fps = [f'../input/feedback/cfg_ch_10c/fold-1/checkpoint_last_seed{seed}.pth' for seed in seeds]

class_preds_word, sep_preds_word = get_preds(name, FOLD, cfg, state_dict_fps, cache_allowed=cache_allowed)

class_preds_word_blend += [class_preds_word]
sep_preds_word_blend += [sep_preds_word]


# In[ ]:


name = 'cfg_ch_11'

namecheck.append(name)
seeds = [295209]  # , 634549 447068
cfg = get_cfg(name)
cfg.dataset = 'ds_ch_4_var_tok_fix'
cfg.model = "mdl_ch_4"
cfg.max_length = 1024
cfg.max_length_test = 1024

state_dict_fps = [f'../input/feedback/cfg_ch_11/fold-1/checkpoint_last_seed{seed}.pth' for seed in seeds]

class_preds_word, sep_preds_word = get_preds(name, FOLD, cfg, state_dict_fps, cache_allowed=cache_allowed)

class_preds_word_blend += [class_preds_word]
sep_preds_word_blend += [sep_preds_word]


# In[ ]:


name = 'cfg_pp_12b'
namecheck.append(name)

seeds = [743507] # , 830398
cfg = get_cfg(name)
cfg.dataset = 'ds_pp_6d'
cfg.model = "mdl_pp_7e"
cfg.max_length = 4096
cfg.max_length_test = 4096

state_dict_fps = [f'../input/feedback/cfg_pp_12b/fold-1/checkpoint_last_seed{seed}.pth' for seed in seeds]

class_preds_word, sep_preds_word = get_preds(name, FOLD, cfg, state_dict_fps, cache_allowed=cache_allowed)

class_preds_word_blend += [class_preds_word]
sep_preds_word_blend += [sep_preds_word]


# In[ ]:


name = 'cfg_ch_33d'
namecheck.append(name)

seeds = [313148] 
cfg = get_cfg(name)
cfg.model = 'mdl_ch_8_deberta_varlen'
cfg.dataset = 'ds_ch_6_varlen'
cfg.max_length = 4096
cfg.tokenizer = '../input/feedback-huggingface-models/transformers/deberta-v2-xlarge'

state_dict_fps = [f'../input/feedback/cfg_ch_33d/fold-1/checkpoint_last_seed{seed}.pth' for seed in seeds]

class_preds_word, sep_preds_word = get_preds(name, FOLD, cfg, state_dict_fps, cache_allowed=cache_allowed, regenerate_pos_embeddings=True)

class_preds_word_blend += [class_preds_word]
sep_preds_word_blend += [sep_preds_word]


# In[ ]:


name = 'cfg_ch_34d'
namecheck.append(name)

seeds = [520521] # , 827251
cfg = get_cfg(name)
cfg.model = 'mdl_ch_8_deberta_varlen'
cfg.dataset = 'ds_ch_6_varlen'
cfg.max_length = 1250
cfg.tokenizer = '../input/feedback-huggingface-models/transformers/deberta-v2-xxlarge'

state_dict_fps = [f'../input/feedback/cfg_ch_34d/fold-1/checkpoint_last_seed{seed}.pth' for seed in seeds]

class_preds_word, sep_preds_word = get_preds(name, FOLD, cfg, state_dict_fps, cache_allowed=cache_allowed, regenerate_pos_embeddings=True)

class_preds_word_blend += [class_preds_word]
sep_preds_word_blend += [sep_preds_word]


# In[ ]:


name = 'cfg_dh_02C'
namecheck.append(name)

seeds = [892929] 
cfg = get_cfg(name)
cfg.model = "mdl_dh_1A_bigbird"
cfg.tokenizer = '../input/bigbird-roberta-large/'
cfg.max_length = 4096
cfg.max_length_test = 4096
cfg.verify_sample = False
cfg.config_path = "./configs/cfg_dh_02_bigbird_config.json"
state_dict_fps = [f'../input/feedback/cfg_dh_02C/fold-1/checkpoint_last_seed{seed}.pth' for seed in seeds]

class_preds_word, sep_preds_word = get_preds(name, FOLD, cfg, state_dict_fps, cache_allowed=cache_allowed)

class_preds_word_blend += [class_preds_word]
sep_preds_word_blend += [sep_preds_word]


# In[ ]:


name = 'cfg_dh_03E'
namecheck.append(name)

seeds = [461966] # , 860867
cfg = get_cfg(name)
cfg.model = "mdl_dh_2_bart_pp"
cfg.max_length = 1024
cfg.max_length_test = 1024
cfg.verify_sample = False
cfg.config_path = "./configs/cfg_dh_03_bart_conifg.json"
state_dict_fps = [f'../input/feedback/cfg_dh_03E/fold-1/checkpoint_last_seed{seed}.pth' for seed in seeds]

class_preds_word, sep_preds_word = get_preds(name, FOLD, cfg, state_dict_fps, cache_allowed=cache_allowed)

class_preds_word_blend += [class_preds_word]
sep_preds_word_blend += [sep_preds_word]


# In[ ]:


name = 'cfg_ch_32c'
namecheck.append(name)

seeds = [453209]  # ,529051 ,863515
cfg = get_cfg(name)
cfg.model = 'mdl_ch_8_deberta_varlen'
cfg.dataset = 'ds_ch_6_varlen'
cfg.max_length = 4096
cfg.tokenizer = '../input/feedback-huggingface-models/transformers/deberta-v3-large'

state_dict_fps = [f'../input/feedback/cfg_ch_32c/fold-1/checkpoint_last_seed{seed}.pth' for seed in seeds]

class_preds_word, sep_preds_word = get_preds(name, FOLD, cfg, state_dict_fps, cache_allowed=cache_allowed, regenerate_pos_embeddings=True)

class_preds_word_blend += [class_preds_word]
sep_preds_word_blend += [sep_preds_word]


# In[ ]:


name = 'cfg_dh_04C'
namecheck.append(name)

seeds = [430400] # , 860867
cfg = get_cfg(name)
cfg.model = "mdl_dh_3_deberta"
cfg.tokenizer = '../input/feedback-huggingface-models/transformers/deberta-large'
cfg.max_length = 2000
cfg.max_length_test = 2000
cfg.verify_sample = False
cfg.config_path = "./configs/cfg_dh_03_deberta_config.json"
state_dict_fps = [f'../input/feedback/cfg_dh_04C/fold-1/checkpoint_last_seed{seed}.pth' for seed in seeds]

class_preds_word, sep_preds_word = get_preds(name, FOLD, cfg, state_dict_fps, cache_allowed=cache_allowed)

class_preds_word_blend += [class_preds_word]
sep_preds_word_blend += [sep_preds_word]


# In[ ]:


name = 'cfg_dh_05D'
namecheck.append(name)

seeds = [181857] 
cfg = get_cfg(name)
cfg.model = "mdl_dh_3_deberta"
cfg.tokenizer = '../input/deberta-xlarge'
cfg.dataset = 'ds_dh_2A'
cfg.max_length = 2000
cfg.max_length_test = 2000
cfg.verify_sample = False
cfg.verify_sample = False
cfg.config_path = "./configs/cfg_dh_03_debertaxl_config.json"
state_dict_fps = [f'../input/feedback/cfg_dh_05D/fold-1/checkpoint_last_seed{seed}.pth' for seed in seeds]

class_preds_word, sep_preds_word = get_preds(name, FOLD, cfg, state_dict_fps, cache_allowed=cache_allowed)

class_preds_word_blend += [class_preds_word]
sep_preds_word_blend += [sep_preds_word]


# In[ ]:


name = 'cfg_dh_14A'
namecheck.append(name)

seeds = [840631] 
cfg = get_cfg(name)
cfg.model = "mdl_dh_8_deberta"
cfg.tokenizer = '../input/feedback-huggingface-models/transformers/deberta-large'
cfg.max_length = 1536
cfg.max_length_test = 1536
cfg.verify_sample = False
cfg.offline_inference = True
cfg.config_path = "./configs/cfg_dh_03_deberta_config.json"
state_dict_fps = [f'../input/feedback/cfg_dh_14A/fold-1/checkpoint_last_seed{seed}.pth' for seed in seeds]

class_preds_word, sep_preds_word = get_preds(name, FOLD, cfg, state_dict_fps, cache_allowed=cache_allowed)

class_preds_word_blend += [class_preds_word]
sep_preds_word_blend += [sep_preds_word]


# In[ ]:


name = 'cfg_dh_14F'
namecheck.append(name)

seeds = [310279] 
cfg = get_cfg(name)
cfg.model = "mdl_dh_8C_deberta"
cfg.tokenizer = '../input/deberta-xlarge'
cfg.dataset = 'ds_dh_1A'
cfg.max_length = 1536
cfg.max_length_test = 1536
cfg.verify_sample = False
cfg.offline_inference = True
cfg.config_path = "./configs/cfg_dh_03_debertaxl_config.json"
state_dict_fps = [f'../input/feedback/cfg_dh_14F/fold-1/checkpoint_last_seed{seed}.pth' for seed in seeds]

class_preds_word, sep_preds_word = get_preds(name, FOLD, cfg, state_dict_fps, cache_allowed=cache_allowed)

class_preds_word_blend += [class_preds_word]
sep_preds_word_blend += [sep_preds_word]


# In[ ]:


# Check the load order is the same
assert namecheck == load_configs


# In[ ]:


def blendfn(preds_word_blend, 
            predidls, 
            start_threshold = 0.55, 
            position_proba_thresh = 0.5,
            map_clip_curr = {},
            nposn = 3000):
    
    preds = torch.zeros(10, *preds_word_blend.shape[1:])
    preds[:9] = preds_word_blend.clone()
    nposn  = preds.shape[-1]
    name_map1 = copy(name_map)
    name_map1[8] = 'Other'
    
    # Start here
    filemat = np.expand_dims(predidls, 1).repeat(nposn, 1)
    posnmat = np.expand_dims(np.arange(nposn), 0).repeat(len(predidls), 0)
    logitmat = preds.view(10, -1)
    
    idx = logitmat.sum(0)!=0
    predfnm  = filemat.flatten()[idx]
    predpos = torch.from_numpy(posnmat.flatten()[idx])
    logitall = logitmat[:, idx].transpose(1,0)
    
    probscls = logitall[:,:8].numpy() #* proba_weight
    probsbrk = logitall[:,8:].numpy()
    
    classidx = pd.Series(probscls.argmax(1))#.map(name_map)
    
    preddfls = []
    for cidx in tqdm(range(7)):
        aggdf = pd.DataFrame(probsbrk.copy(), columns=['start', 'end'])
        aggdf['discourse_prob'] = pd.Series(probscls.max(1))
        aggdf['discourse_type'] = pd.Series(probscls.argmax(1))#.map(name_map)
        aggdf['discourse_type'][(classidx != cidx)] = 8
        aggdf['predictionstring'] = predpos.numpy()
        
        
        aggdf['grp8'] = range(len(aggdf))
        aggdf['grp8'][aggdf['discourse_type']==8] += 9999999 - np.arange(len(aggdf['grp8'][aggdf['discourse_type']==8]))
        aggdf['grp8'][aggdf['discourse_type']!=8] -= np.arange(len(aggdf['grp8'][aggdf['discourse_type']!=8]))
        aggdf['grp8len'] = aggdf.groupby('grp8')['end'].transform(lambda x: len(x))
        aggdf['grp8avg'] = aggdf.groupby('grp8')['discourse_prob'].transform(np.mean)

        aggdf['start'][((aggdf['discourse_type'] == 8) & \
                (aggdf['grp8avg'] > 0.6)) & \
                (aggdf['grp8len'] > 4)] = 1.

        aggdf = aggdf.groupby(predfnm)['start', 'end', 'discourse_type', 'predictionstring', 'discourse_prob'].agg(lambda x: list(x))
        aggdf['startcut'] = aggdf['start'].apply(lambda x: (np.array(x)>start_threshold).astype(np.int32))
    
        u, ind = np.unique(predfnm, return_index=True)
        aggdf = aggdf.loc[u[np.argsort(ind)]]
        aggdf = pd.DataFrame({'discourse_type':  flatten(aggdf.discourse_type.tolist()), 
                              'discourse_prob':  flatten(aggdf.discourse_prob.apply(list).tolist()), 
                                  'predictionstring':  flatten(aggdf.predictionstring.tolist()),
                                  'start': flatten(aggdf.start.apply(list).tolist()),
                                  'startcut': flatten(aggdf.startcut.apply(list).tolist()),
                                  'end': flatten(aggdf.end.apply(list).tolist())})
        aggdf['id'] = predfnm
        aggdf['seq'] = (aggdf['startcut']).cumsum()
        aggdf['seq'][aggdf['discourse_type']==8] -= -999999999 + (aggdf['startcut'][aggdf['discourse_type']==8]).cumsum()
        
        aggdf['discourse_type'] = aggdf['discourse_type'].map(name_map1)
        preddf = aggdf.groupby(['id', 'discourse_type', 'seq'])['predictionstring']\
                    .apply(list).reset_index()\
                        .query('discourse_type != "None"')
        preddfprob = aggdf.groupby(['id', 'discourse_type', 'seq'])['discourse_prob'].mean()\
                            .reset_index().query('discourse_type != "None"')
        preddf = pd.merge(preddf, preddfprob, on=['id', 'discourse_type', 'seq'])#.drop('seq', 1)
        preddf = preddf.drop('seq', 1)
        preddfls .append(preddf.query('discourse_type != "Other"'))
    
    def threshold(df, map_clip_curr):
        df = df.copy().reset_index(drop= True)
        df['len'] = df['predictionstring'].apply(len)
        #for key, value in cfg.map_clip.items():
        for key, value in map_clip_curr.items():
            index = df.loc[df['discourse_type']==key].query(f'len<{value}').index
            df.drop(index, inplace = True)
        return df
    preddf = pd.concat(preddfls)
    preddf = threshold(preddf, map_clip_curr)
    preddf.predictionstring = preddf.predictionstring.apply(lambda x: ' '.join(map(str, x)))
    preddf['class'] = preddf['discourse_type']
    preddf = preddf[preddf.discourse_prob > position_proba_thresh]
    
    return preddf

def link_class_meta(preddf):
    preddf = link_class(preddf, class_name="Evidence", min_gap=1, min_span_dist=38, min_len=8)
    preddf = link_class(preddf, class_name="Position", min_gap=3, min_span_dist=100, min_len=2)
    preddf = link_class(preddf, class_name="Concluding Statement", min_gap=25, min_span_dist=80, max_spans=1, direction="backward", min_len=3)
    preddf = link_class(preddf, class_name="Lead", min_gap=10, min_span_dist=50, max_spans=1, min_len=3)
    return preddf


# In[ ]:





# In[ ]:


from blend_dh_01 import class_thresh_2_proba_fn, flatten, get_cfg, get_dl, val_data_2_ls, link_class
from torch.utils.data import DataLoader, Dataset
nmodels = len(load_configs)

name_map_rev = dict((v,k) for k,v in name_map.items())
proba_thresh_weights = class_thresh_2_proba_fn(ppcfg.proba_thresh, ppcfg.name_map)


# In[ ]:


model_weights = np.array(model_weights)
proba_thresh_wts = torch.tensor(proba_thresh_weights).permute(1,0)
clip_index = [load_configs.index(c) for c in ['cfg_ch_33d', 'cfg_ch_34d','cfg_ch_32c']]
non_clip_index = [i for i in range(nmodels) if i not in clip_index]
rawpreds = torch.zeros(9, 
                   1, 
                   len(class_preds_word_blend[0]), 
                   nposn)


# In[ ]:


clip_posn = []
for step in range(len(class_preds_word_blend[0])):
    step_class_preds = torch.zeros(8, nmodels, nposn)
    for i in range(nmodels):
        step_class_pred = class_preds_word_blend[i][step][:nposn,:8].permute(1,0)
        step_class_preds[:,i,:step_class_pred.shape[-1]] = step_class_pred
    # Clip where deberta extended on past the labels
    max_len_deb = torch.where(step_class_preds[:,clip_index].sum((0,1))!=0)[0][-1]
    max_len_other = torch.where(step_class_preds[:,non_clip_index].sum((0,1))!=0)[0][-1]
    if max_len_other == (max_len_deb-1):
        step_class_preds[:,:,max_len_other+1:] = 0.
        clip_posn.append(max_len_other)
    else:
        clip_posn.append(max(max_len_other, max_len_deb))
    step_class_preds  = step_class_preds * model_weights[None,:,None]
    msk = (step_class_preds!=0).float() * model_weights[None,:,None]
    msk = msk.sum(1)
    step_class_preds = step_class_preds.sum(1) / msk
    step_class_preds = torch.nan_to_num(step_class_preds, 0)
    step_class_preds = torch.softmax(step_class_preds, 0)
    step_class_preds = step_class_preds * proba_thresh_wts
    step_class_preds[msk==0] = 0
    rawpreds[:8,0,step] = step_class_preds
clip_posn = torch.tensor(clip_posn)


# In[ ]:


for step in range(len(sep_preds_word_blend[0])):
    step_sep_preds = torch.zeros(nmodels, nposn)
    for i in range(nmodels):
        step_sep_pred = sep_preds_word_blend[i][step][:clip_posn[step]].squeeze(1)
        step_sep_preds[i,:len(step_sep_pred)] = step_sep_pred
    step_sep_preds = step_sep_preds * model_weights[:,None]
    msk = (step_sep_preds!=0).float() * model_weights[:,None]
    step_sep_preds = step_sep_preds.sum(0)/ msk.sum(0)
    step_sep_preds = torch.nan_to_num(step_sep_preds, 0)
    rawpreds[8,0,step] = step_sep_preds


# In[ ]:


preddf = blendfn(rawpreds,ids, start_threshold, position_proba_threshold, map_clip)


# In[ ]:


preddf = link_class_meta(preddf)


# In[ ]:


sub = preddf[["id","class","predictionstring"]].copy()
sub.to_csv('submission.csv',index=False)


# In[ ]:


sub


# In[ ]:




