#!/usr/bin/env python
# coding: utf-8

# # <B><u>Whale&Dolphin Background Removed Images (2/4)</u></B>

# # <B>[1] Introduction</B>

# There is no annotation data, which shows position of whale and dolphin, in "Happywhale - Whale and Dolphin Identification" competition dataset. It is difficult to create annotation data in a kernel because of limitations of kaggle notebook. So it is created in the series of the following kernels,
# - ["Whale&Dolphin Background Removed Images (1/4)"](https://www.kaggle.com/code/acchiko/whale-dolphin-background-removed-images-1-4),
# - ["Whale&Dolphin Background Removed Images (2/4)"](https://www.kaggle.com/code/acchiko/whale-dolphin-background-removed-images-2-4) (This kernel),
# - ["Whale&Dolphin Background Removed Images (3/4)"](https://www.kaggle.com/code/acchiko/whale-dolphin-background-removed-images-3-4),
# - ["Whale&Dolphin Background Removed Images (4/4)"](https://www.kaggle.com/code/acchiko/whale-dolphin-background-removed-images-4-4),
# - ["Annotation Data for Detecting Whale&Dolphin"](https://www.kaggle.com/code/acchiko/annotation-data-for-detecting-whale-dolphin),
# - ["Whale&Dolphin Cropped Images (1/3)"](https://www.kaggle.com/code/acchiko/whale-dolphin-cropped-images-1-3),
# - ["Whale&Dolphin Cropped Images (2/3)"](https://www.kaggle.com/code/acchiko/whale-dolphin-cropped-images-2-3),
# - ["Whale&Dolphin Cropped Images (3/3)"](https://www.kaggle.com/code/acchiko/whale-dolphin-cropped-images-3-3).
# 
# If the kernels are loaded, created annotation data and cropped images can be available from the following path for example,
# - Annotation data : /kaggle/input/annotation-data-for-detecting-whale-dolphin/train_with_annotation.csv,
# - Background removed images : /kaggle/input/annotation-data-for-detecting-whale-dolphin/nobg/train_image/*.png,
# - Cropped images : /kaggle/input/whale-dolphin-cropped-images-3-3/cropped/train_image/*.png.

# NOTE1 : The author is a beginner of Kaggle/MachineLearning/Python/English. So the kernel may have several bugs/wrongs. I am happy to get your comments. Thank you in advance for your kind advice to make the kernel so NICE! and to make me NICE deep learning guy!! 

# NOTE2 : Utility scripts for visualization of dataset for "Happywhale - Whale and Dolphin Identification" competition is defined in my other kernel ["Utility Functions for Visualization of Dataset"](https://www.kaggle.com/code/acchiko/utility-functions-for-visualization-of-dataset). The way to create/use utility scripts is summarized in my other kernel ["How to Create Utility Scripts"](https://www.kaggle.com/code/acchiko/utility-functions-for-visualization-of-dataset).

# NOTE3 : Dataset for "Happywhale - Whale and Dolphin Identification" competition is visualized with [Plotly](https://plotly.com/python/) and [Matplotlib](https://matplotlib.org/) in my other kernel ["Preview of Whale&Dolphin Dataset with Plotly/Matplotlib"](https://www.kaggle.com/acchiko/preview-of-whale-dolphin-dataset-with-plotly-matpl). It may help us to get some insight into strategy of training, data augumentation, etc.

# # <B>[2] Preparation of dataset</B>

# ## [2-1] Loading dataset

# Dataset for "Happywhale - Whale and Dolphin Identification" competition is loaded by clicking the following items in the sidebar of kaggle notebook,

# ###  "+ Add data" -> "Competition Data" -> "Add (Happywhale - Whale and Dolphin Identification)".

# If it succeeds, the dataset are loaded to the following path.

# In[ ]:


path_to_dir_happywhale_data = "/kaggle/input/happy-whale-and-dolphin"
get_ipython().system('ls {path_to_dir_happywhale_data}')


# ## [2-2] Showing contents of dataset 

# Contents of metadata for train images is shown.

# In[ ]:


path_to_happywhale_train_metadata = "%s/train.csv" % path_to_dir_happywhale_data
get_ipython().system('head -5 {path_to_happywhale_train_metadata}')


# List of train/test images are shown.

# In[ ]:


path_to_dir_happywhale_train_images = "%s/train_images" % path_to_dir_happywhale_data
get_ipython().system('ls {path_to_dir_happywhale_train_images} | head -5')


# In[ ]:


path_to_dir_happywhale_test_images = "%s/test_images" % path_to_dir_happywhale_data
get_ipython().system('ls {path_to_dir_happywhale_test_images} | head -5')


# ## [2-3] Creation of metadata for test images

# The metadata for train images exists, but the one for test images is not exists. So the dummy metadata for test images is created.

# In[ ]:


# Creates metadata for test images. 
path_to_happywhale_test_metadata = "/kaggle/working/test.csv"

get_ipython().system('echo "image,species,individual_id" > {path_to_happywhale_test_metadata}')
get_ipython().system('ls {path_to_dir_happywhale_test_images} | sed "s/.jpg/.jpg,unknown,unknown/g" >> {path_to_happywhale_test_metadata}')

# Shows contents of created metadata.
get_ipython().system('head -5 {path_to_happywhale_test_metadata}')


# ## [2-4] Creation of working directories

# Working directories for saving processed images(background removed/cropped images) are created.

# In[ ]:


# Creates working directory for saving background removed images.
path_to_dir_nobg_train_images = "/kaggle/working/nobg/train_images"
get_ipython().system('mkdir -p {path_to_dir_nobg_train_images}')


# In[ ]:


# Creates working directory for saving cropped images.
path_to_dir_cropped_train_images = "/kaggle/working/cropped/train_images"
get_ipython().system('mkdir -p {path_to_dir_cropped_train_images}')


# In[ ]:


# Shows created directories.
get_ipython().system('ls /kaggle/working/*')


# ## [2-5] Loading utility scripts

# Utility scripts for visualization of "Happywhale - Whale and Dolphin Identification" competition dataset, which is defined in the other kernel ["Utility Functions for Visualization of Dataset"](https://www.kaggle.com/code/acchiko/utility-functions-for-visualization-of-dataset), are loaded. The way to use/create utility scripts is summarized in the other kernel ["How to Create Utility Scripts"](https://www.kaggle.com/code/acchiko/how-to-use-create-utility-scripts).

# In[ ]:


import utility_functions_for_visualization_of_dataset as myutils


# ## [2-6] Showing images

# Train/Test images are shown. First, class for processing train/test images and metadata is defined.

# In[ ]:


# Imports required libs.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
from tqdm import tqdm


# In[ ]:


# Defines class for processing train/test images and metadata.
class WhaleAndDolphin():
    def __init__(self, path_to_metadata, path_to_dir_images, \
                 path_to_dir_nobg_images, path_to_dir_cropped_images):
        self._path_to_metadata = path_to_metadata
        self._path_to_dir_images = path_to_dir_images
        self._path_to_dir_nobg_images = path_to_dir_nobg_images
        self._path_to_dir_cropped_images = path_to_dir_cropped_images
        
        # Loads metadata to variable "_metadata_all"
        self._metadata_all = pd.read_csv(path_to_metadata)
        
        # Adds several colmuns.
        path_to_images = \
            ["%s/%s" % (path_to_dir_images, row.image) \
             for row in self._metadata_all.itertuples()]
        self._metadata_all["path_to_image"] = path_to_images
        
        path_to_nobg_images = \
            ["%s/%s" % (path_to_dir_nobg_images, row.image.replace(".jpg", ".png")) \
             for row in self._metadata_all.itertuples()]
        self._metadata_all["path_to_nobg_image"] = path_to_nobg_images
        
        path_to_cropped_images = \
            ["%s/%s" % (path_to_dir_cropped_images, row.image.replace(".jpg", ".png")) \
             for row in self._metadata_all.itertuples()]
        self._metadata_all["path_to_cropped_image"] = path_to_cropped_images
        
        annotations_xyxy = \
            [[] for row in self._metadata_all.itertuples()]
        self._metadata_all["annotations_xyxy"] = annotations_xyxy
        
        # Copies the metadata for processing it.
        self._metadata = self._metadata_all.copy()
        
        self._all_species = self.getSpecies()
        self._all_individual_ids = self.getIndividualIDs()
        
    def resetMetadata(self, initialize=False):
        if hasattr(self, "_metadata_tmp") and not initialize:
            self._metadata = self._metadata_tmp.copy()
        else:
            self._metadata = self._metadata_all.copy()
            
    def saveMetadataTemporary(self):
        self._metadata_tmp = self._metadata.copy()
        
    def filterMetadata(self, query="index > -1"):
        sliced_metadata = \
            self._metadata.query(query).reset_index(drop=True)
        self._metadata = sliced_metadata.copy()
        
    def filterMetadataNoBackgroundImageExistence(self):
        indices = []
        for row in self._metadata.itertuples():
            if not os.path.exists(row.path_to_nobg_image):
                indices.append(row.Index)
                
        sliced_metadata = self._metadata.drop(index=indices).reset_index(drop=True)
        self._metadata = sliced_metadata.copy()
        
    def writeMetadata(self, path_to_metadata):
        self._metadata.to_csv(path_to_metadata, index=False)
        
    def getMetadata(self):
        return self._metadata
    
    def getSpecies(self):
        return self._metadata["species"].unique()
    
    def _species2id(self, species):
        return np.where(self._all_species == species)
    
    def getIndividualIDs(self):
        return self._metadata["individual_id"].unique()
    
    def showImagesTile(self, num_cols=4, draw_annotations=False):
        metadata = self._metadata
        titles = [row.image for row in metadata.itertuples()]
        path_to_images = [row.path_to_image \
                          for row in metadata.itertuples()]
        images = myutils.getImages(path_to_images)
        if "annotations_xyxy" in metadata.columns and draw_annotations:
            annotations_xyxy_for_images = [row.annotations_xyxy \
                                           for row in metadata.itertuples()]
            texts_for_images = [["" for _ in \
                                 range(len(row.annotations_xyxy))] \
                                 for row in metadata.itertuples()]
            myutils.drawAnnotations( \
                images, \
                annotations_xyxy_for_images=annotations_xyxy_for_images, \
                texts_for_images=texts_for_images, \
                line_color="green", line_width=3, text_color="green" \
            )
        myutils.showImagesTile(titles, images, num_cols=num_cols)
        
    def showProcessedImagesTile(self, num_cols=3, draw_annotations=False):
        metadata = self._metadata
        titles, path_to_images = [], []
        for row in metadata.itertuples():
            titles.append("%s (Org.)" % row.image)
            path_to_images.append(row.path_to_image)
            
            titles.append("%s (BG. removed)" % row.image)
            path_to_images.append(row.path_to_nobg_image)
            
            titles.append("%s (Cropped)" % row.image)
            path_to_images.append(row.path_to_cropped_image)
        
        images = myutils.getImages(path_to_images)
        if "annotations_xyxy" in metadata.columns and draw_annotations:
            annotations_xyxy_for_images = [row.annotations_xyxy \
                                           for row in metadata.itertuples()]
            texts_for_images = [["" for _ in \
                                 range(len(row.annotations_xyxy))] \
                                 for row in metadata.itertuples()]
            myutils.drawAnnotations( \
                images[::3], \
                annotations_xyxy_for_images=annotations_xyxy_for_images, \
                texts_for_images=texts_for_images, \
                line_color="red", line_width=3, text_color="red" \
            ) # For only org. images.
        myutils.showImagesTile(titles, images, num_cols=num_cols)
        
    def showIndividualImagesTile(self, num_cols=4, \
                                 max_num_individual_images=4, \
                                 max_num_individuals=10, \
                                 draw_annotations=False):
        self.saveMetadataTemporary()
        
        individual_ids = self.getIndividualIDs()
        for individual_id in individual_ids[:max_num_individuals]:
            print()
            print("Individual ID : %s" % individual_id)
            self.filterMetadata(query="individual_id == \"%s\"" % individual_id)
            self.filterMetadata(query="index < %d" % max_num_individual_images)
            self.showImagesTile( \
                num_cols=num_cols, \
                draw_annotations=draw_annotations
            )
            self.resetMetadata()
            
    def removeBackground(self):
        metadata = self._metadata
        path_to_inputs = [row.path_to_image \
                          for row in metadata.itertuples()]
        path_to_outputs = [row.path_to_nobg_image \
                           for row in metadata.itertuples()]
        
        for path_to_input, path_to_output in \
            zip(path_to_inputs, path_to_outputs):
            get_ipython().system('backgroundremover -i {path_to_input} -o {path_to_output}')
        
    def calculateAnnotationsXyxy(self):
        batch_size = 100
        num_batches = len(self._metadata) // batch_size + 1
        
        for i_batch in range(num_batches):
            i_start = i_batch * batch_size
            i_end = i_start + batch_size
            metadata = self._metadata.iloc[i_start:i_end]
            
            path_to_nobg_images = [row.path_to_nobg_image \
                                   for row in metadata.itertuples()]
            nobg_images = myutils.getImages(path_to_nobg_images)
            fixed_images = [Image.eval(nobg_image, self._removeBugPixel) \
                            for nobg_image in nobg_images]
            class_ids = [self._species2id(row.species) for row \
                         in metadata.itertuples()]
            
            annotations_xyxy = []
            for nobg_image, class_id in zip(fixed_images, class_ids):
                x_min, y_min, x_max, y_max = nobg_image.getbbox() # Bounding box of non-zero region
                confidence = 1.0 # Dummy
                annotation_xyxy = myutils._annotationXyxy(class_id, \
                                                          x_min, y_min, \
                                                          x_max, y_max, \
                                                          confidence)
                annotations_xyxy.append([annotation_xyxy])
                
            self._metadata["annotations_xyxy"].iloc[i_start:i_end] = \
                annotations_xyxy
    
    def _removeBugPixel(self, pixel_value):
        if pixel_value == 1:
            return 0
        else:
            return pixel_value
        
    def cropObject(self):
        batch_size = 100
        num_batches = len(self._metadata) // batch_size + 1
        
        for i_batch in range(num_batches):
            i_start = i_batch * batch_size
            i_end = i_start + batch_size
            metadata = self._metadata.iloc[i_start:i_end]
            
            path_to_inputs = [row.path_to_image \
                              for row in metadata.itertuples()]
            path_to_outputs = [row.path_to_cropped_image \
                               for row in metadata.itertuples()]
            annotations_xyxy = [row.annotations_xyxy for row \
                                in metadata.itertuples()]
            
            for path_to_input, path_to_output, annotations_xyxy in \
                zip(path_to_inputs, path_to_outputs, annotations_xyxy):
                
                image = Image.open(path_to_input)
                annotation_xyxy = \
                    self._maxConfidenceAnnotation(annotations_xyxy)
                x_min = annotation_xyxy["x_min"]
                y_min = annotation_xyxy["y_min"]
                x_max = annotation_xyxy["x_max"]
                y_max = annotation_xyxy["y_max"]
                image_cropped = image.crop((x_min, y_min, x_max, y_max))
                image_cropped.save(path_to_output)
            
    def _maxConfidenceAnnotation(self, annotations_xyxy):
        confidences = np.array([annotation_xyxy["confidence"] \
                                for annotation_xyxy in annotations_xyxy])
        index = np.argmax(confidences)
        return annotations_xyxy[index]


# Some of train images are shown as example. Test images can be shown using the same class.

# In[ ]:


# Loads metadata for train images.
whale_and_dolphin = WhaleAndDolphin(
    path_to_metadata=path_to_happywhale_train_metadata,
    path_to_dir_images=path_to_dir_happywhale_train_images,
    path_to_dir_nobg_images=path_to_dir_nobg_train_images,
    path_to_dir_cropped_images=path_to_dir_cropped_train_images
)


# In[ ]:


# Shows train images for the first 3 individuals for each species.
num_cols = 4
max_num_individual_images = 4
max_num_individuals = 3

all_species = whale_and_dolphin.getSpecies()
for i, species in enumerate(all_species):
    whale_and_dolphin.filterMetadata(query="species == \"%s\"" % species)
    
    print()
    print("--------------------------------------------------")
    print()
    print("   Images for species No.%02d %s" % (i, species))
    print()
    print("--------------------------------------------------")
    whale_and_dolphin.showIndividualImagesTile( \
        num_cols=num_cols, \
        max_num_individual_images=max_num_individual_images, \
        max_num_individuals=max_num_individuals \
    )
    print()
    print()
    
    whale_and_dolphin.resetMetadata(initialize=True)


# # <B>[3] Creation of annotation data</B>

# Annotation data is created with the following steps,
# - Removing background of image,
# - Calculating bounding box,
# - Cropping bounding box,
# - Showing processed images.

# ## [3-1] Removing background of image

# Background of image is removed with the command line tool ["backgroundremover"](https://github.com/nadermx/backgroundremover) to make easy to calculate bounding box. Sometimes background is not removed properly with the tool, but it works well for almost cases.

# It is difficult to remove background with the tool in a kernel because of RAM utilization limitation of kaggle notebook (~4000 images can be processed in a kernel within the limitation.). So it is done in the series of the following kernels,
# - ["Whale&Dolphin Background Removed Images (1/4)"](https://www.kaggle.com/code/acchiko/whale-dolphin-background-removed-images-1-4),
# - ["Whale&Dolphin Background Removed Images (2/4)"](https://www.kaggle.com/code/acchiko/whale-dolphin-background-removed-images-2-4) (This kernel),
# - ["Whale&Dolphin Background Removed Images (3/4)"](https://www.kaggle.com/code/acchiko/whale-dolphin-background-removed-images-3-4),
# - ["Whale&Dolphin Background Removed Images (4/4)"](https://www.kaggle.com/code/acchiko/whale-dolphin-background-removed-images-4-4).

# First, the tool is installed.

# In[ ]:


get_ipython().system('pip install backgroundremover')


# Then, background is removed.

# In[ ]:


# Loads metadata for train images.
whale_and_dolphin = WhaleAndDolphin(
    path_to_metadata=path_to_happywhale_train_metadata,
    path_to_dir_images=path_to_dir_happywhale_train_images,
    path_to_dir_nobg_images=path_to_dir_nobg_train_images,
    path_to_dir_cropped_images=path_to_dir_cropped_train_images
)


# In[ ]:


# Removes background. Limits number of processing images because of kaggle notebook limitation. Sets same number for each species as possible.
num_images_per_species = int(4000 / len(all_species))
i_start = 1 * num_images_per_species
i_end = i_start + num_images_per_species

all_species = whale_and_dolphin.getSpecies()
whale_and_dolphin.saveMetadataTemporary()

for i, species in enumerate(all_species):
    whale_and_dolphin.filterMetadata(query="species == \"%s\"" % species)
    whale_and_dolphin.filterMetadata(query="%d <= index < %d" % (i_start, i_end))
    print()
    print("--------------------------------------------------")
    print()
    print("   Processing images for species No.%02d %s" % (i, species))
    print()
    print("--------------------------------------------------")
    whale_and_dolphin.removeBackground()
    print()
    
    whale_and_dolphin.resetMetadata()


# If it succeeds, background removed images are saved in the following directory.

# In[ ]:


get_ipython().system('ls {path_to_dir_nobg_train_images} | cat -n | tail -5')

