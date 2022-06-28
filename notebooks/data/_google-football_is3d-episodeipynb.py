#!/usr/bin/env python
# coding: utf-8

# Today, the criteria for generating 3D episode rendering is episode % 6 == 0
# 
# Before that, the criteria was episode % 3 == 0
# 
# To check if the rendering of an old episode is 2D or 3D, we need to check the videos for ourselves.

# In[ ]:


import cv2
import numpy as np
import matplotlib.pyplot as plt


# In[ ]:


# check if episode is rendered in 3D or 2D
# returns:
# 1: 3D
# 0: 2D
# None: could not open episode

def is3d(episode, verbose=False):

    green_channel = 1
    threshold = 100
    
    # open video
    url = f'https://www.kaggleusercontent.com/episodes/{episode}.webm'
    vcap = cv2.VideoCapture(url)
    # get first frame
    ret, frame = vcap.read()
    
    #close video
    vcap.release()

    if ret == False:
        if verbose:
            print(f'cannot read episode: {episode}!')
        return False
    else:
        mean = np.mean(frame[:,:,green_channel])
        if mean > threshold:
            # 3D
            result = 1
        else:
            result = 0
        if verbose:
            print(f'Green Channel mean: {mean}')
            if result == 1:
                print("3D")
            else:
                print("2D")
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            plt.imshow(frame)
            plt.show()
        return result


# In[ ]:


print(is3d(episode='4682501', verbose=True))


# In[ ]:


print(is3d(episode='4576290', verbose=True))


# In[ ]:


print(is3d(episode='asdsa', verbose=True)) 


# In[ ]:




