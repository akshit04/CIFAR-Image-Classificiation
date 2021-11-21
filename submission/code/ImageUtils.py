import numpy as np
import torch
from torchvision import transforms as tf
import random

""" This script implements the functions for data augmentation and preprocessing.
"""

class MyDataset(torch.utils.data.Dataset):
  def __init__(self, data, labels, training=False):
        self.labels = labels
        self.data = torch.tensor(data.reshape(data.shape[0],3,32,32)/255, dtype=torch.float32)

        self.training = training
        if training:
            self.transform = tf.Compose([
                    tf.RandomHorizontalFlip(),
                    tf.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)), # values picked from internet
                    tf.RandomCrop((32,32), 4, padding_mode='edge'),
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
        X = self.transform(x)
        if self.labels is not None:
            y = self.labels[index]
            return X, y
        else:
            return X
