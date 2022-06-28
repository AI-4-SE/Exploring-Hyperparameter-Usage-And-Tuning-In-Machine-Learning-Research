#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from fastai.vision.all import *
import torchaudio
from joblib import Parallel, delayed
from tqdm.notebook import tqdm
from torchvision.utils import save_image


# In[ ]:


base_folder = Path('../input/birdclef-2022')


# In[ ]:


train = pd.read_csv('../input/birdclef-2022/train_metadata.csv')


# In[ ]:


items = get_files(base_folder, extensions='.ogg')


# In[ ]:


items


# In[ ]:


N_FFT = 2048
HOP_LEN = 1024


# In[ ]:


def create_spectrogram(filename):
    audio, sr = torchaudio.load(filename)
    specgram = torchaudio.transforms.MelSpectrogram(sample_rate=sr, 
                                                    n_fft=N_FFT, 
                                                    win_length=N_FFT, 
                                                    hop_length=HOP_LEN
                                                    ,
                                                    center=True,
                                                    pad_mode="reflect",
                                                    power=2.0,
                                                    norm='slaney',
                                                    onesided=True,
                                                    n_mels=224,
                                                    mel_scale="htk"
                                                   )(audio).mean(axis=0)
    specgram = torchaudio.transforms.AmplitudeToDB()(specgram)
    specgram = specgram - specgram.min()
    specgram = specgram/specgram.max()
    
    
    return specgram


# In[ ]:


filename = items[2]
spec_default = create_spectrogram(filename)


# In[ ]:


filename.parent


# In[ ]:


Path('spectograms/train_img').mkdir(parents=True, exist_ok=True)
Path('spectograms/test_img').mkdir(parents=True, exist_ok=True)


# In[ ]:


train.primary_label.unique()


# In[ ]:


def create_image(filename):
    specgram = create_spectrogram(filename)
    if filename.parent.name in train.primary_label.unique():
        Path('spectograms/train_img/{}'.format(filename.parent.name)).mkdir(parents=True, exist_ok=True)
        dest = Path('spectograms/train_img/{}'.format(filename.parent.name))/f'{filename.stem}.png'
    else:
         dest = Path('spectograms/test_img')/f'{filename.stem}.png'
    print(dest)
    save_image(specgram, dest)


# In[ ]:


_ = Parallel(n_jobs=-1)(delayed(create_image)(file) for file in tqdm(items))


# In[ ]:




