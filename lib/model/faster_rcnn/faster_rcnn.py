import random
import torch as tt
import torch.nn.functional as F
from torch import nn
import numpy as np
from torch.autograd import Variable
from torchvision import models

from lib.model.utils.config import cfg
from lib.model.rpn.rpn import _RPN
from lib.model.roi_pooling.modules.roi_pool import _RoIPooling
from lib.model.roi_crop.modules.roi_crop import _RoICrop
from lib.model.roi_align.modules.roi_align import RoIAlignAvg
from lib.model.rpn.proposal_target_layer_cascade import _ProposalTargetLayer

class _FasterRCNN(nn.Module):
    def __init__(self, classes, class_agnostic):
        super(_FasterRCNN, self).__init__()
        self.classes = classes
        self.n_classes = len(classes)
        self.class_agnostic = class_agnostic

        # loss
        self.RCNN_loss_cls = 0.0
        self.RCNN_loss_bbox = 0.0

        # RPN
        self.RCNN_rpn = _RPN(self.)

