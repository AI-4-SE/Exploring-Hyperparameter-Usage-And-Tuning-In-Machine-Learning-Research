#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Make sure you have Internet option selected. You need to verify your account to enable such in the settings.


# In[ ]:


get_ipython().system('pip install --upgrade azure-cognitiveservices-vision-computervision')


# In[ ]:


get_ipython().system('pip install pillow')


# In[ ]:


from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from azure.cognitiveservices.vision.computervision.models import OperationStatusCodes
from azure.cognitiveservices.vision.computervision.models import VisualFeatureTypes
from msrest.authentication import CognitiveServicesCredentials

from array import array
import os
from PIL import Image
import sys
import time


# In[ ]:


subscription_key = "56d629f17c484de2b2cf718cdb1c9221"
endpoint = "https://azurecv.cognitiveservices.azure.com/"
computervision_client = ComputerVisionClient(endpoint, CognitiveServicesCredentials(subscription_key))


# In[ ]:


remote_image_url = "https://cdn.britannica.com/54/75854-050-E27E66C0/Eiffel-Tower-Paris.jpg"


# In[ ]:


import urllib.request  
import PIL

urllib.request.urlretrieve(
  remote_image_url,
   "input.jpg")
  
PIL.Image.open("input.jpg")


# In[ ]:


'''
Describe an Image - remote
This example describes the contents of an image with the confidence score.
'''
print("===== Describe an image - remote =====")
# Call API
description_results = computervision_client.describe_image(remote_image_url )

# Get the captions (descriptions) from the response, with confidence level
print("Description of remote image: ")
if (len(description_results.captions) == 0):
    print("No description detected.")
else:
    for caption in description_results.captions:
        print("'{}' with confidence {:.2f}%".format(caption.text, caption.confidence * 100))


# In[ ]:


# Call API with content type (landmarks) and URL
detect_domain_results_landmarks = computervision_client.analyze_image_by_domain("landmarks", remote_image_url)
print()
print("Landmarks in the remote image:")
if len(detect_domain_results_landmarks.result["landmarks"]) == 0:
    print("No landmarks detected.")
else:
    for landmark in detect_domain_results_landmarks.result["landmarks"]:
        print(landmark["name"]+" : "+str(landmark["confidence"]))

