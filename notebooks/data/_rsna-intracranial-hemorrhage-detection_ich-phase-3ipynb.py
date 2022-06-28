#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Input the datasets

dir_csv = '../input/rsna-intracranial-hemorrhage-detection/rsna-intracranial-hemorrhage-detection'
dir_train_img = '../input/rsna-train-stage-1-images-png-224x/stage_1_train_png_224x'
dir_test_img = '../input/rsna-test-stage-1-images-png-224x/stage_1_test_png_224x'



# In[ ]:


# Parameters

n_classes = 6
n_epochs = 50
batch_size = 64


# In[ ]:


#from apex import amp
import os
import cv2
import glob
import pydicom
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from albumentations import Compose, ShiftScaleRotate, Resize, CenterCrop, HorizontalFlip, RandomBrightnessContrast
from albumentations.pytorch import ToTensor
from torch.utils.data import Dataset,Subset
from tqdm import tqdm_notebook as tqdm
from matplotlib import pyplot as plt
from torchvision import transforms
from sklearn.metrics import accuracy_score,jaccard_similarity_score,f1_score,recall_score


# In[ ]:


# Functions

class IntracranialDataset(Dataset):

    def __init__(self, csv_file, path, labels, transform=None):
        
        self.path = path
        self.data = pd.read_csv(csv_file)
        self.transform = transform
        self.labels = labels

    def __len__(self):
        
        return len(self.data)

    def __getitem__(self, idx):
        
        img_name = os.path.join(self.path, self.data.loc[idx, 'Image'] + '.png')
        img = cv2.imread(img_name)   
        
        if self.transform:       
            
            augmented = self.transform(image=img)
            img = augmented['image']   
            
        if self.labels:
            
            labels = torch.tensor(
                self.data.loc[idx, ['epidural', 'intraparenchymal', 'intraventricular', 'subarachnoid', 'subdural', 'any']])
            return {'image': img, 'labels': labels}    
        
        else:      
            
            return {'image': img}
    
    


# In[ ]:


train_tmp = os.path.join(dir_csv, 'stage_2_train.csv')
os.path.exists(train_tmp)


# # CSV

# In[ ]:


# CSVs

train = pd.read_csv(os.path.join(dir_csv, 'stage_2_train.csv'))
test = pd.read_csv(os.path.join(dir_csv, 'stage_2_sample_submission.csv'))


# In[ ]:


# Split train out into row per image and save a sample

train[['ID', 'Image', 'Diagnosis']] = train['ID'].str.split('_', expand=True)
train = train[['Image', 'Diagnosis', 'Label']]
train.drop_duplicates(inplace=True)
train = train.pivot(index='Image', columns='Diagnosis', values='Label').reset_index()
train['Image'] = 'ID_' + train['Image']
train.head()


# 

# In[ ]:


test


# In[ ]:


# Some files didn't contain legitimate images, so we need to remove them

png = glob.glob(os.path.join(dir_train_img, '*.png'))
png = [os.path.basename(png)[:-4] for png in png]
png = np.array(png)

train = train[train['Image'].isin(png)]
train.to_csv('train.csv', index=False)


# In[ ]:


test


# In[ ]:


test[['ID','Image','Diagnosis']] = test['ID'].str.split('_', expand=True)


# In[ ]:


test


# In[ ]:


test['Image'] = 'ID_' + test['Image']


# In[ ]:


test


# In[ ]:


test = test[['Image', 'Label']]


# In[ ]:


test.drop_duplicates(inplace=True)


# In[ ]:


test.to_csv('test.csv', index=False)


# In[ ]:


test.head()


# In[ ]:


# Data loaders

transform_train = Compose([CenterCrop(200, 200),
                           #Resize(224, 224),
                           HorizontalFlip(),
                           RandomBrightnessContrast(),
    ShiftScaleRotate(),
    ToTensor()
])

transform_test= Compose([CenterCrop(200, 200),
                         #Resize(224, 224),
    ToTensor()
])

train_dataset = IntracranialDataset(
    csv_file='train.csv', path=dir_train_img, transform=transform_train, labels=True)

train_dataset=torch.utils.data.Subset(train_dataset, range(0,25000))

test_dataset = IntracranialDataset(
    csv_file='test.csv', path=dir_test_img, transform=transform_test, labels=False)

test_dataset=torch.utils.data.Subset(test_dataset, range(0,2000))

#newtest_dataset=torch.utils.data.Subset(train_dataset,range(25000,27000))

data_loader_train = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

data_loader_test = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
                                        
#data_loader_newtest = torch.utils.data.DataLoader(newtest_dataset, batch_size=batch_size, shuffle=False, num_workers=4)


# # DataLoaders

# In[ ]:


newtest_dataset = IntracranialDataset(
    csv_file='train.csv', path=dir_train_img, transform=transform_train, labels=True)
newtest_dataset=torch.utils.data.Subset(newtest_dataset,range(25000,27000))


# In[ ]:


data_loader_newtest = torch.utils.data.DataLoader(newtest_dataset, batch_size=batch_size, shuffle=False, num_workers=4)


# # Model

# In[ ]:


device = torch.device("cuda:0")
from torchvision.models.resnet import ResNet, Bottleneck


# In[ ]:


from torch.hub import load_state_dict_from_url
from torchvision.models.resnet import ResNet, Bottleneck


model_urls = {
    'resnext101_32x8d': 'https://download.pytorch.org/models/ig_resnext101_32x8-c38310e5.pth',
    'resnext101_32x16d': 'https://download.pytorch.org/models/ig_resnext101_32x16-c6f796b0.pth',
    'resnext101_32x32d': 'https://download.pytorch.org/models/ig_resnext101_32x32-e4b90b00.pth',
    'resnext101_32x48d': 'https://download.pytorch.org/models/ig_resnext101_32x48-3e41cc8a.pth',
}


def _resnext(arch, block, layers, pretrained, progress, **kwargs):
    model = ResNet(block, layers, **kwargs)
    #print('a')
    state_dict = load_state_dict_from_url(model_urls[arch], progress=progress)
   # print('b')
    model.load_state_dict(state_dict)
    #print('c')
    return model


# In[ ]:


def resnext101_32x8d_wsl(progress=True, **kwargs):
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 8
    return _resnext('resnext101_32x8d', Bottleneck, [3, 4, 23, 3], True, progress, **kwargs)


# In[ ]:


model = resnext101_32x8d_wsl()


# In[ ]:


# Model

#device = torch.device("cuda:0")
#model = torch.hub.load('facebookresearch/WSL-Images', 'resnext101_32x8d_wsl')
model.fc = torch.nn.Linear(2048, n_classes)

model.to(device)

criterion = torch.nn.BCEWithLogitsLoss()
plist = [{'params': model.parameters(), 'lr': 2e-5}]
optimizer = optim.Adam(plist, lr=2e-5)

#model, optimizer = amp.initialize(model, optimizer, opt_level="O1")


# In[ ]:


model.load_state_dict(torch.load('../input/params/save.pth'))


# # Training

# In[ ]:


# Train


#for epoch in range(n_epochs):
    
#    print('Epoch {}/{}'.format(epoch, n_epochs - 1))
#    print('-' * 10)

 #   model.train()    
  #  tr_loss = 0
    
  #  tk0 = tqdm(data_loader_train, desc="Iteration")

  #  for step, batch in enumerate(tk0):

   #     inputs = batch["image"]
    #    labels = batch["labels"]

     #   inputs = inputs.to(device, dtype=torch.float)
     #   labels = labels.to(device, dtype=torch.float)

     #   outputs = model(inputs)
     #  loss = criterion(outputs, labels)

       # with amp.scale_loss(loss, optimizer) as scaled_loss:
            #scaled_loss.backward()
      #  loss.backward()
      #  tr_loss += loss.item()

       # optimizer.step()
       # optimizer.zero_grad()
        
      #  if epoch == 1 and step > 6000:
          #  epoch_loss = tr_loss / 6000
          #  print('Training Loss: {:.4f}'.format(epoch_loss))
           # break

    #epoch_loss = tr_loss / len(data_loader_train)
    #print('Training Loss: {:.4f}'.format(epoch_loss))


# 

# In[ ]:


#boi working final
for param in model.parameters():
    param.requires_grad = False
jacc=[]
acc=[]
f1=[]
model.eval()
new = []
test_pred = np.zeros((len(newtest_dataset) * n_classes, 1))
for i, x_batch in enumerate(tqdm(data_loader_newtest)):
    
    x_image = x_batch["image"]
    x_image = x_image.to(device, dtype=torch.float)
    x_label = x_batch["labels"]
    x_label  = x_label.to(device,dtype=torch.float)
    with torch.no_grad():
        
        pred = model(x_image)
        #print(pred)
        #print(pred.size())
        #break
        test_pred[(i * batch_size * n_classes):((i + 1) * batch_size * n_classes)] = torch.sigmoid(
            pred).detach().cpu().reshape((len(x_image) * n_classes, 1))
        new.append(torch.sigmoid(pred).detach().cpu())
        new1=torch.sigmoid(pred).detach().cpu()>=0.5
        #print(new1,x_label)
        #ans = (new1.float()==x_label.cpu())
        jacc.append(jaccard_similarity_score(new1.float(),x_label.cpu()))
        acc.append(accuracy_score(new1.float(),x_label.cpu()))
        #print(accuracy_score(new1.float(),x_label.cpu()))
        f1.append(f1_score(new1.float(),x_label.cpu(),average='micro'))
        #print(f1_score(new1.float(),x_label.cpu(),average='micro'))


# In[ ]:


jacc


# In[ ]:


ans= sum(acc)/len(acc)
ans


# In[ ]:


JC final = 91.36
ACC final= 89.11

