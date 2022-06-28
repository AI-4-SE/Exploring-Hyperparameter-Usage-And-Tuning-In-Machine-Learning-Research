#!/usr/bin/env python
# coding: utf-8

# # Hand pose and shape estimation
# 
# #### Author: [Debabrata Mandal]()

# <div class="alert alert-block alert-info"> <b> This notebook previously contained the entire implementation of part 1 of this series, which has been shifted to a new notebook. Look at the Notebooks section to see check all of them out.</b> </div>

# # Table of contents
# 
# 1. [Introduction](#introduction)
# 2. [Challenges](#challenges)
# 3. [Tools](#tools)
# 4. [Notebooks](#notebooks)

# # Introduction <a name="introduction"></a>

# | ![](https://i.ibb.co/PmJdVkL/1.png) | 
# |:--:| 
# | *Image source: [MANO](https://mano.is.tue.mpg.de/) page [3]()* |

# This [5-part](#notebooks) notebook series will give an introduction to hand pose and shape estimation from RGB and depth images. This is a well researched problem by now with exciting solutions emerging to it every year. With the rise of CNNs and deep neural networks, even this domain has observed a shift from traditional computer graphics only approaches to ones mixing computer vision with the former to achieve phenomenal results. That being said, there is still a lot of challenges that need to solved to achieve near perfect predictions of the human hand from low dimensional (and often unsatisfactory) inputs like images. 
# 

# | ![](https://i.ibb.co/C8TGLy9/Screenshot-2022-05-27-at-4-18-40-PM-min.png) | ![](https://i.ibb.co/1ZtdfjG/Screenshot-2022-05-27-at-4-17-16-PM-min.png) |
# |:--:|:--:|
# | *Clear hand RGB image (FreiHand dataset)* | *Unclear synthetic hand image (RHD Dataset)* |
# 

# 
# This notebook series takes it up a notch and tries to augment RGB images with [depth](https://en.wikipedia.org/wiki/Depth_map) images to form a **2.5D** representation of hand i.e. a **pointcloud** which is served as inputs to our models. The reason why it is not a full 3D representation even though such an input has all `x`,`y` & `z` coordinates is because of partial occlusion. **Self-occlusion** has been a major problem in accurately determining hand poses and shapes from images because of the large number of orientations that even the occluded part of hand can contain. Under such circumstances, it becomes essential to smartly restrict the number of degree of freedoms that can be allowed to represent a hand parametrically. 
# 

# | ![](https://i.ibb.co/PtHtBpD/Screenshot-2022-05-27-at-4-12-59-PM-min.png) | 
# |:--:| 
# | *First 2 are examples of good point cloud representations of hand while last one is partially occluded* |
# 

# Our target predictions will vary from very simple **3D** keypoints representing the hand pose to **semantic segmentation** of input hand images and eventually to the very complicted **pose and shape** estimation problem. Obviously, since the challenge levels are different for each problem we will have to cater our model choice to each of them separately. 
# 
# Not excited yet?! Well, look into the [challenges](#challenges) section as to what makes this problem even more interesting to know more about.

# # Challenges <a name="challenges"></a>

# Some of the major challenges in hand pose/shape estimation from pointclouds are **self-occlusion** (discussed above), **computational complexity**, **pointcloud sparsity** and **capturing local relations**. 
# 
# I already gave an informal explanation to why self-occlucsion is bad for input pointclouds in the introduction. The next issue, computational complexity is more of a practical limitation i.e. a pointcloud is a 3 dimensional data which means the well knwon 2D convolutions applied over images cannot work well with point clouds. Thus there are 2 alternatives, either choose 3D convolutions operating on 3D grids (voxels) or shift to algorithms which operate on point sets. The former has its own challenges since requires its input data to be in 3-dimensions which requires a lot of computational capacity. 
# 
# This would have been useful and probably popular if not for the large proportion of **sparsely** populated areas in hand pointclouds (see the images above). Hence, more efficient ways have been devised to deal with pointcloud data. A big requirement for all these alternatives is to maintain the **permutation-invariant** property of point cloud representations i.e. the order in which the points are sent as inputs should not affect the output features/predictions.
# 
# 
# Some well known neural networks have emerged over the last few years to deal with point clouds thanks to the research boost in **self-driving cars**, **augmented reality** etc. We will be using [PointNet](https://arxiv.org/abs/1612.00593) and [PointConv](https://arxiv.org/abs/1811.07246) (U-Net like) in our notebooks to get hand pose and shape predictions.

# # Tools <a name="tools"></a>

# [Tensorflow](https://www.tensorflow.org/) has been used extensively in this short project to showcase its effectiveness when dealing with complicated input pipelines or implementing deep neural architectures. 
# 
# More specifically the packages being used are:
# * **[tensorflow]()** - Create model ([tf.keras](https://www.tensorflow.org/api_docs/python/tf/keras)), dataset pipeline ([tf.data](https://www.tensorflow.org/guide/data)), metrics, callbacks, losses etc.
# * **[tensorflow-graphics]()** - Useful [graphics](https://www.tensorflow.org/graphics/api_docs/python/tfg), [rendering](https://www.tensorflow.org/graphics/api_docs/python/tfg/rendering), [neural network layers](https://www.tensorflow.org/graphics/api_docs/python/tfg/nn), etc. tools and our PointNet model.
# * **[tensorflow-datasets](https://www.tensorflow.org/datasets)** - [Build](https://www.tensorflow.org/datasets) a TFrecords dataset hosted on Google cloud storage for fast I/O (with [TPU]()).
# * **[tensorflow-addons](https://www.tensorflow.org/addons/api_docs/python/tfa)/[probability](https://www.tensorflow.org/probability/overview)** - Needed for some utilities that tensorflow does not provide.
# * **[plotly](https://plotly.com/python/getting-started/#overview)** - Build most of our visualisation and plotting tools.
# * **[open3d](http://www.open3d.org/docs/0.7.0/python_api/open3d.geometry.html)** - Pointcloud generation from depth images.
# 
# We showcase using both [GPU]() and [TPU](https://cloud.google.com/tpu/docs/tpus) devices with tensorflow in [separate](#notebooks) notebooks.

# # Notebooks <a name="notebooks"></a>

# 1. [x] [Generating pointclouds and predicting hand pose](https://www.kaggle.com/code/debman/playing-with-hands-using-tensorflow-part-1/) - Create **clean pointclouds** from input rgbd images and build an fast input data pipeline. Then predict **hand pose** using **3D** keypoints using the simplest of all point cloud architectures i.e. a vanilla [Pointnet](https://arxiv.org/abs/1612.00593).
# 
# 2. [x] [Shifting to TPUs]() - The previous notebook makes use of a GPU device to accelerate training and inference. This notebook showcases a simple way to switch to a TPU device and make use of a custom [TFRecordsDataset](https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset) as workaround to some issues faced when doing the shift.
# 
# 3. [Improving our basic PointNet model](#) - In the first notebook, the failures of a basic PointNet are highlighted and one of the ways to solve it is to make use of a more advanced model to predict hand pose from pointclouds. In this notebook, we explore a version of the [PointNetv2]() architecture inspired by this [paper]() to make better pose estimations.
# 
# 4. [Semantic hand segmentation](#) - This notebook implements a recent [U-net]() like model for pointclouds to achieve good semantic segmentation of fingers and palm. This is useful for hand gesture recognition where the most accurate representation is not necessary and segmentation is sufficient. This will implement a [modified]() version of [PointConv]() architecture inspired by this [paper]().
# 
# 5. [Hand pose and shape estimation using PointClouds](#) - Finally, this notebook tries to accumulate all the learnings from previous notebooks and build a modified [PointConv]() model to predict hand pose and **shape** using a differential hand mesh renderer ([MANO]()) written using [tensorflow-graphics](). This will also make use of [JAX]() to implement a **tailor-optimiser** inspired from this [paper](). Currently, there is no implementation of **PointConv** in tensorflow hence this notebook also serves as a substitute for the official Pytorch [implementation]().
# 
# 6. (Optional) [Hand mesh prediction from RGB images](#) - This is an exciting work done in this [paper]() where hand meshes are needed as inputs to predict pose and shape paramaters from images in a semi-supervised manner. This was inspired by this [research work]().

# In[ ]:




