import pytest
import torch
from timm.models import *

from src.models.modules.model_wrapper import ModelWrapper

def test_model_wrapper_timm_resnet():
    test_model = ModelWrapper()
    
    assert type(test_model.model) == ResNet
    assert test_model.num_classes == 1000
    assert test_model.pretrained == False