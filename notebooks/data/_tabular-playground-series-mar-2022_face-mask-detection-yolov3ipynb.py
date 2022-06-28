#!/usr/bin/env python
# coding: utf-8

# ## 导入依赖

# ## 导入依赖

# In[ ]:


import xml.etree.ElementTree as ET
import glob
import os
import json
import shutil
from tqdm.notebook import tqdm
from PIL import Image
from IPython.display import HTML
from base64 import b64encode

def play_video(filename):
    html = ''
    video = open(filename,'rb').read()
    src = 'data:video/mp4;base64,' + b64encode(video).decode()
    html += '<video width=1000 controls autoplay loop><source src="%s" type="video/mp4"></video>' % src 
    return HTML(html)


# ## 从github下载yolov3并安装依赖

# In[ ]:


# !git clone https://github.com/ultralytics/yolov3
# !pip install -r yolov3/requirements.txt 


# ### 使用yolov3自带的预训练模型进行推理

# In[ ]:


get_ipython().system('python /kaggle/working/yolov3/detect.py --source /kaggle/working/demo.jpg --weights /kaggle/working/yolov3/yolov3.pt --project /kaggle/working/yolov3/runs/ --name exp --exist-ok')
img = Image.open('/kaggle/working/yolov3/runs/exp/demo.jpg')
img


# ### 使用yolov3调用本地摄像头进行目标检测

# In[ ]:


play_video('/kaggle/working/yolov3.mp4')


# ## 使用yolov3训练自己的数据集

# ### 数据集介绍

# - 数据集来自kaggle https://www.kaggle.com/datasets/andrewmvd/face-mask-detection
# - 该数据集是voc格式的，包含三种类别，['正确佩戴口罩', '错误佩戴口罩', '未佩戴口罩']

# ### 将voc数据处理成yolo所需格式

# In[ ]:


def xml_to_yolo_bbox(bbox, w, h):
    # xmin, ymin, xmax, ymax
    x_center = ((bbox[2] + bbox[0]) / 2) / w
    y_center = ((bbox[3] + bbox[1]) / 2) / h
    width = (bbox[2] - bbox[0]) / w
    height = (bbox[3] - bbox[1]) / h
    return [x_center, y_center, width, height]

classes = []
input_dir = '/kaggle/input/annotations/'
image_dir = '/kaggle/input/images/'
output_dir = '/kaggle/working/labels/'

# 创建labels输出目录
shutil.rmtree(output_dir, ignore_errors=True)
os.mkdir(output_dir)

# 把所有xml文件处理成yolo所需要的txt文件
files = glob.glob(os.path.join(input_dir, '*.xml'))
for fil in tqdm(files):
    basename = os.path.basename(fil)
    filename = os.path.splitext(basename)[0]
    # check if the label contains the corresponding image file
    if not os.path.exists(os.path.join(image_dir, f'{filename}.png')):
        print(f'{filename} image does not exist!')
        continue
    result = []
    tree = ET.parse(fil)
    root = tree.getroot()
    width = int(root.find('size').find('width').text)
    height = int(root.find('size').find('height').text)

    for obj in root.findall('object'):
        label = obj.find('name').text
        if label not in classes:
            classes.append(label)
        index = classes.index(label)
        pil_bbox = [int(x.text) for x in obj.find('bndbox')]
        yolo_bbox = xml_to_yolo_bbox(pil_bbox, width, height)
        bbox_string = ' '.join([str(x) for x in yolo_bbox])
        result.append(f'{index} {bbox_string}')
    if result:
        with open(os.path.join(output_dir, f'{filename}.txt'), 'w', encoding='utf-8') as f:
            f.write('\n'.join(result))


# ### 训练

# In[ ]:


lr0: 0.001  # initial learning rate (SGD=1E-2, Adam=1E-3)
lrf: 0.1  # final OneCycleLR learning rate (lr0 * lrf)
momentum: 0.937  # SGD momentum/Adam beta1
weight_decay: 0.0005  # optimizer weight decay 5e-4
warmup_epochs: 3.0  # warmup epochs (fractions ok)
warmup_momentum: 0.8  # warmup initial momentum
warmup_bias_lr: 0.1  # warmup initial bias lr
box: 0.05  # box loss gain
cls: 0.5  # cls loss gain
cls_pw: 1.0  # cls BCELoss positive_weight
obj: 1.0  # obj loss gain (scale with pixels)
obj_pw: 1.0  # obj BCELoss positive_weight
iou_t: 0.20  # IoU training threshold
anchor_t: 4.0  # anchor-multiple threshold
# anchors: 3  # anchors per output layer (0 to ignore)
fl_gamma: 0.0  # focal loss gamma (efficientDet default gamma=1.5)
hsv_h: 0.015  # image HSV-Hue augmentation (fraction)
hsv_s: 0.7  # image HSV-Saturation augmentation (fraction)
hsv_v: 0.4  # image HSV-Value augmentation (fraction)
degrees: 0.0  # image rotation (+/- deg)
translate: 0.10  # image translation (+/- fraction)
scale: 0.5  # image scale (+/- gain)
shear: 0.0  # image shear (+/- deg)
perspective: 0.0  # image perspective (+/- fraction), range 0-0.001
flipud: 0.5  # image flip up-down (probability)
fliplr: 0.5  # image flip left-right (probability)
mosaic: 0.5  # image mosaic (probability)
mixup: 0.5 # image mixup (probability)
copy_paste: 0.0  # segment copy-paste (probability)


# In[ ]:


# 在一台ubuntu20.04+RTX3080 机器上进行训练，这里跳过

# !python train.py \
# --img 640 \
# --batch 32 \
# --epochs 500 \
# --data input/rmfd.yaml \
# --hyp input/hyp.yaml \
# --weight yolov3/yolov3-tiny.pt \
# --project 'rmfd-yolov5-openvino' \
# --name 'yolov3-tiny-img640-exp1' \
# --exist-ok \
# --workers 4


# ### 训练结果

# ![metrics.jpg](attachment:57ac8cf4-5b86-43aa-825c-31aa5fc7f739.jpg)

# ![loss.jpg](attachment:a26d3741-58e5-4de3-ab5c-1318f31dbe47.jpg)

# ### 推理

# #### 正确佩戴口罩示例

# In[ ]:


file_name = 'maksssksksss4.png'
path = '/kaggle/input/face-mask-detection/images/' + file_name
get_ipython().system('python /kaggle/working/yolov3/detect.py --source {path} --weights /kaggle/working/yolov3/rmfd_yolov3.pt --project /kaggle/working/yolov3/runs/ --name exp --exist-ok')
img = Image.open('/kaggle/working/yolov3/runs/exp/' + file_name)
img


# #### 未佩戴口罩示例

# In[ ]:


file_name = 'maksssksksss14.png'
path = '/kaggle/input/face-mask-detection/images/' + file_name
get_ipython().system('python /kaggle/working/yolov3/detect.py --source {path} --weights /kaggle/working/yolov3/rmfd_yolov3.pt --project /kaggle/working/yolov3/runs/ --name exp --exist-ok')
img = Image.open('/kaggle/working/yolov3/runs/exp/' + file_name)
img


# #### 错误佩戴口罩示例

# In[ ]:


file_name = 'maksssksksss371.png'
path = '/kaggle/input/face-mask-detection/images/' + file_name
get_ipython().system('python /kaggle/working/yolov3/detect.py --source {path} --weights /kaggle/working/yolov3/rmfd_yolov3.pt --project /kaggle/working/yolov3/runs/ --name exp --exist-ok')
img = Image.open('/kaggle/working/yolov3/runs/exp/' + file_name)
img


# #### 调用本地摄像头进行口罩检测

# In[ ]:


play_video('/kaggle/working/rmfd_yolov3.mp4')

