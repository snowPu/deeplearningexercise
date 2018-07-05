"""SegmentationNN"""
import torch
import torch.nn as nn
import torchvision
from torchvision import models

from ..utils import get_upsampling_weight
from .config import vgg16_path



class SegmentationNN(nn.Module):

    def __init__(self, num_classes=23, pretrained=True):
        super(SegmentationNN, self).__init__()

        ########################################################################
        #                             YOUR CODE                                #
        ########################################################################
        vgg = models.vgg16()
        if pretrained:
            vgg.load_state_dict(torch.load(vgg16_path))
        features, classifier = list(vgg.features.children()), list(vgg.classifier.children())

        features[0].padding = (100, 100)

        for f in features:
            if 'MaxPool' in f.__class__.__name__:
                f.ceil_mode = True
            elif 'ReLU' in f.__class__.__name__:
                f.inplace = True

        self.features5 = nn.Sequential(*features)

        fc6 = nn.Conv2d(512, 4096, kernel_size=7)
        fc6.weight.data.copy_(classifier[0].weight.data.view(4096, 512, 7, 7))
        fc6.bias.data.copy_(classifier[0].bias.data)
        fc7 = nn.Conv2d(4096, 4096, kernel_size=1)
        fc7.weight.data.copy_(classifier[3].weight.data.view(4096, 4096, 1, 1))
        fc7.bias.data.copy_(classifier[3].bias.data)
        score_fr = nn.Conv2d(4096, num_classes, kernel_size=1)
        score_fr.weight.data.zero_()
        score_fr.bias.data.zero_()
        self.score_fr = nn.Sequential(
            fc6, nn.ReLU(inplace=True), nn.Dropout(), fc7, nn.ReLU(inplace=True), nn.Dropout(), score_fr
        )

        self.upscore = nn.ConvTranspose2d(num_classes, num_classes, kernel_size=64, stride=32, bias=False)
        self.upscore.weight.data.copy_(get_upsampling_weight(num_classes, num_classes, 64))
        ########################################################################
        #                             END OF YOUR CODE                         #
        ########################################################################

    def forward(self, x):
        """
        Forward pass of the convolutional neural network. Should not be called
        manually but by calling a model instance directly.

        Inputs:
        - x: PyTorch input Variable
        """
        ########################################################################
        #                             YOUR CODE                                #
        ########################################################################
        x_size = x.size()
        pool5 = self.features5(x)
        score_fr = self.score_fr(pool5)
        upscore = self.upscore(score_fr)
        return upscore[:, :, 19: (19 + x_size[2]), 19: (19 + x_size[3])].contiguous()
        ########################################################################
        #                             END OF YOUR CODE                         #
        ########################################################################

        return x

    @property
    def is_cuda(self):
        """
        Check if model parameters are allocated on the GPU.
        """
        return next(self.parameters()).is_cuda

    def save(self, path):
        """
        Save model with its parameters to the given path. Conventionally the
        path should end with "*.model".

        Inputs:
        - path: path string
        """
        print('Saving model... %s' % path)
        torch.save(self, path)
