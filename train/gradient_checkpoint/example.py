import torch.nn as nn
import torchvision.models as models
from utils import checkpoint_segment
import torch
model = models.mobilenet_v2()
features = model.features
for i in range(1, 18):
    features[i] = checkpoint_segment.insert_checkpoint(features[i])
input = torch.rand([1, 3, 224, 224])
output = model(input)
output = output.sum()
output.backward()
