import argparse
import torchvision.models as models
model = models.mobilenet_v2(pretrained=True)
print(model)
