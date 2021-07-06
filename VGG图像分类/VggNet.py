import torch.nn as nn
import torch.nn.init
import math
class VGG(nn.Module):
    '''
    VGG model
    '''
    def __init__(self, features):
        super(VGG, self).__init__()
        self.features = features
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(512,512),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512,512),
            nn.ReLU(True),
            nn.Linear(512,10),
        )

        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. /n))
                m.bias.data.zero_()

        def forward(self, x):
            x = self.features(x)
            x = x.view(x.size(0),-1)
            x = self.classifier(x)
            return x

def make_layers(cfg, batch_norm = False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers.append(nn.MaxPool2d(kernel_size=2,stride=2))
        else:
            conv2d = nn.Conv2d(in_channels, out_channels=v, kernel_size=3, padding=1)
            if batch_norm:
                layers.append(conv2d)
                layers.append(nn.BatchNorm2d(v))
                layers.append(nn.ReLU(inplace=True))
            else:
                layers.append(conv2d)
                layers.append(nn.ReLU(inplace=True))
            in_channels = v
    return nn.Sequential(*layers)


cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M',
          512, 512, 512, 512, 'M'],
}

def vgg11():
    return VGG(make_layers(cfg['VGG11']))


def vgg11_bn():
    return VGG(make_layers(cfg['VGG11'], batch_norm=True))

def vgg13():
    return VGG(make_layers(cfg['VGG13']))

def vgg13_bn():
    return VGG(make_layers(cfg['VGG13'],batch_norm=True))

def vgg16():
    return VGG(make_layers(cfg['VGG16']))

def vgg16_bn():
    return VGG(make_layers(cfg['VGG16'], batch_norm=True))

def vgg19():
    return VGG(make_layers(cfg['VGG19']))

def vgg19():
    return VGG(make_layers(cfg['VGG19'],batch_norm=True))