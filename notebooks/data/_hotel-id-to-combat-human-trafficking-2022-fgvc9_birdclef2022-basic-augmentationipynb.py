#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import random
import librosa
import librosa.display as display
import numpy as np
import soundfile as sf
import torch
import torch.utils.data as data
import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import Audio
import os
from pathlib import Path
from typing import Optional, List

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # type: ignore
    torch.backends.cudnn.deterministic = True  # type: ignore
    torch.backends.cudnn.benchmark = True  # type: ignore
    
set_seed(1213)
ROOT = Path("..")
TRAIN_AUDIO_ROOT = Path("../input/birdclef-2022/train_audio")


# # Read csv & make list

# In[ ]:


train_metadata_df = pd.read_csv("../input/birdclef-2022/train_metadata.csv")
label_dic = {v:i for i, v in enumerate(train_metadata_df["primary_label"].unique())}
n_labels = len(label_dic)

train_list = train_metadata_df[["primary_label", "filename"]].values.tolist()


# # Install audiomentations & acoustics
# ## Reference
# + [audiomentations](https://github.com/iver56/audiomentations)
# + [acoustics](https://github.com/python-acoustics/python-acoustics)

# In[ ]:


pip install audiomentations


# In[ ]:


pip install acoustics


# # Make pink & brown noise

# In[ ]:


SR = 32000


# In[ ]:


# I make pink & brown noise before training.
# And I load these noise with random position in training.
# Speed-up is possible by making noise before traing.

import acoustics
brown_noise = acoustics.generator.brown(120*SR)
pink_noise = acoustics.generator.pink(120*SR)

# define directories
if not os.path.exists('../noise'):
    os.mkdir('../noise')
noise_dir = ROOT / 'noise'

sf.write(noise_dir / "brown_noise.wav", brown_noise, samplerate=SR)
sf.write(noise_dir / "pink_noise.wav", pink_noise, samplerate=SR)


# # Dataset

# In[ ]:


# if you use mixing other audio dataset, you can write this.
# I saved other audio dataset as as npy, because numpy is faster than reading audio data.

#noise_np = np.load("***.npy")


# In[ ]:


def show_melspec(y):
    melspec = librosa.power_to_db(librosa.feature.melspectrogram(y, sr=SR, n_mels=128))
    librosa.display.specshow(melspec, sr=SR, x_axis="time", y_axis="mel")
    plt.colorbar()
    plt.show()


# In[ ]:


PERIOD = 5
LENGTH = 3

from audiomentations import Compose, AddGaussianNoise
from audiomentations import AddGaussianSNR, Gain
from audiomentations import AddShortNoises, AddBackgroundNoise

augmenter = Compose([AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.5),
                     AddGaussianSNR(p=0.5),
                     Gain(min_gain_in_db=-12, max_gain_in_db=12, p=0.5),
                     AddBackgroundNoise(sounds_path=noise_dir, min_snr_in_db=3, max_snr_in_db=30, p=0.5),
                     AddShortNoises(noise_dir)
                    ])

class Dataset(data.Dataset):
    def __init__(
            self,
            file_list: List[List[str]],
            phase):
        self.file_list = file_list  # list of list: [file_path, ebird_code]
        self.phase = phase

    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, idx: int):
        target, wav_path = self.file_list[idx]
        
        y, sr = sf.read(TRAIN_AUDIO_ROOT / wav_path)
        if len(y.shape) == 2: # stereo to mono
            y = np.mean(y, axis=-1)
            
        # load 3 sec
        y = y[:SR*LENGTH]
        
        # time shift
        y_new = np.zeros(PERIOD*sr) # 5sec
        if self.phase == "train": 
            begin_no = torch.randint(0, PERIOD*sr-LENGTH*SR, (1,)).detach().cpu().numpy()[0]
        else: # validation
            begin_no = 0
        y_new[begin_no: begin_no + LENGTH*SR] = y
        y = y_new.astype(np.float32)
        
        # label smoothing
        labels = (0.1/n_labels) * np.ones(n_labels, dtype="f")
        labels[label_dic[target]] = 0.9        

        # add noise 
        if self.phase == "train":
            y = augmenter(samples=y, sample_rate=SR).astype(np.float32)
            
            # If you use mixing other audio dataset, you can below
            #noise = noise_np[torch.randint(0, len(noise_np),(1,))] # random load
            #y = 0.9*y + 0.1*noise[:SR*PERIOD]
            y = y.astype(np.float32)

        return {"waveform": y, "targets": labels}
    
train_dataset = Dataset(train_list, phase="val")
Output = train_dataset.__getitem__(0)

plt.plot(Output["targets"])
plt.title("Label")
plt.show()

print("Original sound")
show_melspec(Output["waveform"])
Audio(Output["waveform"], rate=SR)


# In[ ]:


train_dataset = Dataset(train_list, phase="train")
Output = train_dataset.__getitem__(0)

print("Sound with augmentation")
show_melspec(Output["waveform"])
Audio(Output["waveform"], rate=SR)


# In[ ]:




