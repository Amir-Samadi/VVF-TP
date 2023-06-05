import torchvision.models as models
import torch
import torch.nn as nn


class Identity(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return x

class DenseNet121(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        self.feat_extract = models.densenet121(pretrained=pretrained)
        self.feat_extract.classifier = Identity()
        self.output_size = 1024

    def forward(self, x):
        return self.feat_extract(x)


class DecisionDensenetModel(nn.Module):

    def __init__(self, num_classes=40, pretrained=True):
        super().__init__()
        self.feat_extract = DenseNet121(pretrained=pretrained)
        self.classifier = nn.Linear(self.feat_extract.output_size, num_classes)

    def forward(self, input, before_sigmoid=False):

        feat = self.feat_extract(input)
        scores = self.classifier(feat)
        proba = torch.sigmoid(scores)
        if before_sigmoid:
            return scores
        return proba


class pretrainedResnetModel(nn.Module):

  def __init__(self, resnet=50, pretrained=True, layers_to_freeze=[]):
    super(pretrainedResnetModel,self).__init__()
    if resnet==18:
      self.resnet = models.resnet18(pretrained=pretrained)
    elif resnet==34:
      self.resnet = models.resnet34(pretrained=pretrained)
    elif resnet==50:
      self.resnet = models.resnet50(pretrained=pretrained)
    elif resnet==101:
      self.resnet = models.resnet101(pretrained=pretrained)
    elif resnet==152:
      self.resnet = models.resnet152(pretrained=pretrained)

    for layer in layers_to_freeze:

      if layer == 'conv1':
        for param in self.resnet.conv1.parameters():
          param.requires_grad = False
      if layer == 'bn1':
        for param in self.resnet.bn1.parameters():
          param.requires_grad = False
      if layer == 'layer1':
        for param in self.resnet.layer1.parameters():
          param.requires_grad = False
      if layer == 'layer2':
        for param in self.resnet.layer2.parameters():
          param.requires_grad = False
      if layer == 'layer3':
        for param in self.resnet.layer3.parameters():
          param.requires_grad = False
      if layer == 'layer4':
        for param in self.resnet.layer4.parameters():
          param.requires_grad = False

    num_features = self.resnet.fc.in_features

  def forward(self, input):

    scores = self.resnet(input)
    proba = torch.sigmoid(scores)

    return proba
