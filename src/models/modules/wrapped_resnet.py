from .model_wrapper import ModelWrapper

class WrappedResnet(ModelWrapper):
    def __init__(self, model):
        super(WrappedResnet, self).__init__(model)
        self.set_hooks()

    def set_hooks(self):
        if hasattr(self.model, 'layer1'):
            self.model.layer1.register_forward_hook(self.forward_hook('layer1'))
        if hasattr(self.model, 'layer2'):
            self.model.layer1.register_forward_hook(self.forward_hook('layer2'))
        if hasattr(self.model, 'layer3'):
            self.model.layer1.register_forward_hook(self.forward_hook('layer3'))
        if hasattr(self.model, 'layer4'):
            self.model.layer1.register_forward_hook(self.forward_hook('layer4'))