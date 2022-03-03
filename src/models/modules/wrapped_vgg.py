from .model_wrapper import ModelWrapper
import torch

class WrappedVGG(ModelWrapper):
    def __init__(self, model):
        super(WrappedVGG, self).__init__()
        self.set_hooks()

    def set_hooks(self):
        for i, layer in enumerate(self.model.features.children()):
            if isinstance(layer, torch.nn.MaxPool2d):
                self.model.features[i].register_forward_hook(self.forward_hook(i))
