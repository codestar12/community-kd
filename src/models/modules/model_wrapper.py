import torch
from timm.models import ResNet, VGG
from collections import OrderedDict 
import timm

class ModelWrapper(torch.nn.Module):
    def __init__(self, model):
        super(ModelWrapper, self).__init__()        
        self.model = model
        self.f_hooks = []
        self.selected_out = OrderedDict()
    
    def forward_hook(self, layer_name):
        def hook(module, input, output):
            self.selected_out[layer_name] = output
        return hook
    
    def forward(self, x):
        intermediate_outputs = []
        x = self.model(x)
        for value in self.selected_out.values():
            intermediate_outputs.append(value)

        return intermediate_outputs, x

    # def forward(self, x):
    #     intermediate_outputs = []

    #     x = self.model.conv1(x)
    #     x = self.model.bn1(x)
    #     x = self.model.act1(x)
    #     x = self.model.maxpool(x)
        
    #     x = self.model.layer1(x)
    #     f1 = x
    #     x = self.model.layer2(x)
    #     f2 = x
    #     x = self.model.layer3(x)
    #     f3 = x
    #     x = self.model.layer4(x)
    #     f4 = x
        
    #     x = self.model.global_pool(x)
    #     x = self.model.fc(x)

    #     return [f1, f2, f3, f4], x

        