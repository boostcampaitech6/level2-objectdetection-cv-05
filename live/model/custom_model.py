from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torchvision
import torch.nn as nn


class CustomModel(nn.Module):
    def __init__(self, num_classes) -> None:
        super().__init__()
        self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
            pretrained=True
        )
        # get number of input features for the classifier
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    def forward(self, images, targets=None):
        if targets == None:
            return self.model(images)
        else:
            return self.model(images, targets)
