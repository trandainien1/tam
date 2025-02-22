# Wrappers classes for comparison in benchmarks
import torch
import sys
import copy


# sys.path.append("comparison_methods/ViTCX/ViT_CX")

from vitcx.ViT_CX import ViT_CX, reshape_function_vit

from torchvision.models import VisionTransformer as VisionVIT
from timm.models.vision_transformer import VisionTransformer as TimmVIT

print('SUCCESS: ViT-CX loaded successfully')

class ViTCXWrapper:
    """
    Wrapper for ViT-CX: Wrap ViT-CX method to allow similar usage in scripts
    """
    def __init__(self, model, batch_size=2, **kwargs):
        """
        initialisation of the class
        :param model: model used for the maps computations
        """
        self.model = model
        self.batch_size = batch_size
        

    def exec_method(self, x, class_idx=None):
        """
        Call the saliency method
        :param x: input image tensor
        :param class_idx: index of the class to explain
        :return: a saliency map in shape (input_size, input_size)
        """
        with torch.enable_grad():
            model = copy.deepcopy(self.model)
            if isinstance(model, VisionVIT):
                target_layer = model.encoder.layers[-1].ln_1
            elif isinstance(model, TimmVIT):
                target_layer = model.blocks[-1].norm1
            else:
                raise NotImplementedError("Model not supported")
        
            prediction, saliency = ViT_CX(model,
                                           x,
                                           target_layer,
                                           class_idx,
                                           reshape_function=reshape_function_vit,
                                           gpu_batch=self.batch_size,
                                           )
                                    
            saliency = torch.Tensor(saliency).detach()
            del model
            return prediction, saliency
    def generate(self, x, target=None):
        target = target.to('cpu')
        with torch.enable_grad():
            prediction, saliency_map = self.exec_method(x, class_idx=target)
            return prediction, saliency_map.detach().cpu()
