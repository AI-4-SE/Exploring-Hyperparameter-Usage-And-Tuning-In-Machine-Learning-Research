#!/usr/bin/env python
# coding: utf-8

# In[ ]:


f = open('../input/Music/Dastgah/Label.txt','r')
f.read().splitlines()[:10]


# In[ ]:


f = open('../input/Music/Dastgah/Class.txt','r')
print(f.read().splitlines())


# In[ ]:


from scipy.io import wavfile
import matplotlib.pyplot as plt
samplerate, data = wavfile.read('../input/Data/Music/SoundSamples/000001.mp3.wav')
print(samplerate)
plt.imshow(data)

