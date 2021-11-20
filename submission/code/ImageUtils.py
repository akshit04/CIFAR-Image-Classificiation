import numpy as np
import torch
from torchvision import transforms as tf
import random

""" This script implements the functions for data augmentation and preprocessing.
"""

class MyDataset(torch.utils.data.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, data, labels, training=False):
        'Initialization'
        self.labels = labels
        self.data = torch.tensor(data.reshape(data.shape[0],3,32,32)/255, dtype=torch.float32)
        # self.data = self.data.reshape(-1,3,32,32)
        # print(self.data.shape)
        # self.data = (self.data-torch.mean(self.data,dim=(-2,-1))) / (torch.std(self.data,dim=(-2,-1)))
        self.training = training
        if training:
            self.transform = tf.Compose([#tf.ToPILImage(),
                    tf.RandomHorizontalFlip(),
                    tf.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)), # internet says these are good values for cifar 10
                    tf.RandomCrop((32,32), 4, padding_mode='edge'),
                    # tf.RandomRotation(15),
                    tf.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.05, 20), value=0, inplace=False),
                    ])
        else:
            self.transform = tf.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))

  def __len__(self):
        'Denotes the total number of samples'
        return self.data.shape[0]

  def __getitem__(self, index):
        'Generates one sample of data'
        x = self.data[index]
        # X = torch.reshape(self.data[index],(3,32,32))
        # X = X.permute(1,2,0)

        # X = (X-torch.mean(X, dim=(0,1)))/(torch.std(X,dim=(0,1)))
        # X = X.permute(2,0,1)
        X = self.transform(x)
        if self.labels is not None:
            y = self.labels[index]
            return X, y
        else:
            return X
