### YOUR CODE HERE
import torch
import torch.nn as nn
import torch.nn.functional as F
"""This script defines the network.
"""

class ImprovedBlock(nn.Module):
    def __init__(self, in_features, out_features, stride=1, stage=0):
        # print("ImprovedBlock called successfully...")
        super(ImprovedBlock, self).__init__()
        self.stage = stage
        mid_features = int(out_features/4)
        if stage == 0 or stage == 1:
            self.bn1 = nn.BatchNorm2d(mid_features)
            self.bn2 = nn.BatchNorm2d(mid_features)
            if stage == 0:
                self.bn3 = nn.BatchNorm2d(out_features)
        elif stage == 2 or stage == 3:
            self.bn1 = nn.BatchNorm2d(in_features)
            self.bn2 = nn.BatchNorm2d(mid_features)
            self.bn3 = nn.BatchNorm2d(mid_features)
            if stage == 3:
                self.bn4 = nn.BatchNorm2d(out_features)

        self.conv1 = nn.Conv2d(in_features, mid_features, kernel_size=1, stride=stride, bias=False)
        self.conv2 = nn.Conv2d(mid_features, mid_features, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv3 = nn.Conv2d(mid_features, out_features, kernel_size=1, stride=1, bias=False)


        self.projection = None
        if stride != 1 or in_features != out_features:
            self.projection = nn.Conv2d(in_features, out_features, kernel_size=1, stride=stride, bias=False)

    def forward(self, x):
        if self.stage == 1:
            out = F.relu(x)
        elif self.stage == 2 or self.stage == 3:
            out = F.relu(self.bn1(x))


        shortcut = self.projection(x) if self.projection is not None else x
        out = self.conv1(x) if self.stage == 0 else self.conv1(out)

        if self.stage == 0 or self.stage == 1:
            out = self.conv2(F.relu(self.bn1(out)))
            out = self.conv3(F.relu(self.bn2(out)))
            if self.stage == 0:
                out = self.bn3(out)
        elif self.stage == 2 or self.stage == 3:
            out = self.conv2(F.relu(self.bn2(out)))
            out = self.conv3(F.relu(self.bn3(out)))

        out += shortcut
        if self.stage == 3:
            out = F.relu(self.bn4(out))

        return out


class ResNet(nn.Module):
    def __init__(self, resnet_size):
        super(ResNet, self).__init__()

        self.start_features = 16  # starting features for first stack
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False) # first convolution layer
        self.convStack1 = self._stack_layer(64, resnet_size, stride=1) # first layer doesn't cut dimensions, so stride = 1
        self.convStack2 = self._stack_layer(128, resnet_size, stride=2)
        self.convStack3 = self._stack_layer(256, resnet_size, stride=2)
        self.linear = nn.Linear(self.start_features, 10) # linear layer (without softmax)

    def _stack_layer(self, out_features, num_blocks, stride):
        layers = []
        for i in range(num_blocks):
            if i != 0: # only the first block will use a stride of 2 for each stack (excluding first stack)
                stride = 1

            # stage = i if i < 2 else (2 if i<num_blocks-1 else 3)

            stage = 2
            if i < 2:
                stage = i
            elif i >= num_blocks-1:
                stage = 3

            # print("calling ImprovedBlock")
            layers.append(ImprovedBlock(self.start_features, out_features, stride, stage=stage))
            self.start_features = out_features # update the start features after the first convolution in each stack
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.convStack1(out)
        out = self.convStack2(out)
        out = self.convStack3(out)
        out = F.adaptive_avg_pool2d(out,output_size=1)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

### END CODE HERE
