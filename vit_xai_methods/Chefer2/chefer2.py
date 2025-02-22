import torch
import numpy as np

# try:
from chefer.baselines.ViT.ViT_new import VisionTransformer, _conv_filter, _cfg
from chefer.baselines.ViT.helpers import load_pretrained
from timm.models.vision_transformer import default_cfgs as vit_cfgs
print('SUCCESS: chefer2 was successfully imported.')
# except:
    # print('ERROR: chefer2 was not found.')

def vit_base_patch16_224(pretrained=True, model_name="vit_base_patch16_224", pretrained_cfg='orig_in21k_ft_in1k', **kwargs):
    model = VisionTransformer(patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True, **kwargs)
    
    # print('[DEBUG]', vit_cfgs[model])
    cfg = _cfg(url=vit_cfgs[model_name].cfgs[pretrained_cfg].url,nmean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))

    model.default_cfg = cfg

    if pretrained:
        load_pretrained(model, num_classes=model.num_classes, in_chans=kwargs.get('in_chans', 3), filter_fn=_conv_filter)

    return model.cuda()


"""
Method computation
The functions for Chefer2 method applied to ViT are defined in a notebook at
https://github.com/hila-chefer/Transformer-MM-Explainability/blob/main/Transformer_MM_explainability_ViT.ipynb
We have copied them here for lack of being able to import them
"""

# rule 5 from paper
def avg_heads(cam, grad):
    cam = cam.reshape(-1, cam.shape[-2], cam.shape[-1])
    grad = grad.reshape(-1, grad.shape[-2], grad.shape[-1])
    cam = grad * cam
    cam = cam.clamp(min=0).mean(dim=0)
    return cam

# rule 6 from paper
def apply_self_attention_rules(R_ss, cam_ss):
    R_ss_addition = torch.matmul(cam_ss, R_ss)
    return R_ss_addition

def generate_relevance(model, input, index=None):
    output = model(input, register_hook=True)
    if index == None:
        index = np.argmax(output.cpu().data.numpy(), axis=-1)

    one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
    one_hot[0, index] = 1
    one_hot_vector = one_hot
    one_hot = torch.from_numpy(one_hot).requires_grad_(True)
    one_hot = torch.sum(one_hot.cuda() * output)
    model.zero_grad()
    one_hot.backward(retain_graph=True)

    num_tokens = model.blocks[0].attn.get_attention_map().shape[-1]
    R = torch.eye(num_tokens, num_tokens).cuda()

    for blk in model.blocks:
        grad = blk.attn.get_attn_gradients()
        cam = blk.attn.get_attention_map()
        cam = avg_heads(cam, grad)
        R += apply_self_attention_rules(R.cuda(), cam.cuda()).detach()

    return index, R[0, 1:]

class Chefer2Wrapper():
    def __init__(self, model, **kwargs):
        self.model = vit_base_patch16_224()
        self.model.eval()
        assert isinstance(self.model, VisionTransformer), '[ASSERT] Transformer architecture not recognised.'

        print('[MODEL]')
        print('type:', type(self.model), end='\n\n')

        print('[METHOD]')
        print('type:', generate_relevance, end='\n\n')

    def generate(self, x, target=None):
        with torch.enable_grad():
            prediction, saliency_map = generate_relevance(self.model, x, index=target)
            for block in self.model.blocks:
                block.attn.attn_gradients = None
                block.attn.attention_maps = None
            return prediction[0], saliency_map.reshape(14, 14)