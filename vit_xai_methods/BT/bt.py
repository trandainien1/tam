import torch

# try:
from bt.ViT.baselines.ViT.ViT_explanation_generator import Baselines as Exp
from bt.ViT.baselines.ViT.ViT_new import VisionTransformer, _conv_filter, _cfg
from bt.ViT.baselines.ViT.ViT_explanation_generator import Baselines as Exp
# Use load_pretrained from chefer since we use an url, the bt modifications does not allow this test
from chefer.baselines.ViT.helpers import load_pretrained
from timm.models.vision_transformer import default_cfgs as vit_cfgs
print('SUCCESS: bt was successfully imported.')
# except:
    # print('ERROR: bt was not found.')

def vit_base_patch16_224(pretrained=True, model_name="vit_base_patch16_224", pretrained_cfg='orig_in21k_ft_in1k', **kwargs):
    model = VisionTransformer(patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True, **kwargs)

    cfg = _cfg(url=vit_cfgs[model_name].cfgs[pretrained_cfg].url,mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))

    model.default_cfg = cfg

    if pretrained:
        load_pretrained(model, num_classes=model.num_classes, in_chans=kwargs.get('in_chans', 3), filter_fn=_conv_filter)

    return model.cuda()

class BTWrapperCommon:
    def __init__(self, model, start_layer=1, **kwargs):
        self.model = vit_base_patch16_224()
        self.method = Exp(self.model)
        self.start_layer = start_layer

class BTHWrapper(BTWrapperCommon):
    def __init__(self, model, start_layer=1, **kwargs):
        super().__init__(model, start_layer)

        print('[MODEL]')
        print('type:', type(self.model), end='\n\n')

        print('[METHOD]')
        print('type:', self.method.generate_ours, end='\n\n')

    def generate(self, x, target=None):
        with torch.enable_grad():
            prediction, saliency_map = self.method.generate_ours(x,
                                                     index=target,
                                                     start_layer=self.start_layer)
            return prediction[0], saliency_map.reshape(14, 14)

class BTTWrapper(BTWrapperCommon):
    def __init__(self, model, start_layer=1, **kwargs):
        super().__init__(model, start_layer)

        print('[MODEL]')
        print('type:', type(self.model), end='\n\n')

        print('[METHOD]')
        print('type:', self.method.generate_ours_c, end='\n\n')

    def generate(self, x, target=None):
        with torch.enable_grad():
            prediction, saliency_map = self.method.generate_ours_c(x,
                                                       index=target,
                                                       start_layer=self.start_layer)
            return prediction[0], saliency_map.reshape(14, 14)


