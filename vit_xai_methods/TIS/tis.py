import torch

try:
    # from .tis.tis.tis import TIS
    from tis.tis import TIS
    print('SUCCESS: tis was successfully imported.')
except:
    print('ERROR: tis was not found.')

class TISWrapper:
    def __init__(self, model, n_masks=1024, batch_size=128, tokens_ratio=0.5, **kwargs):
        self.model = model
        self.method = TIS(self.model, n_masks=n_masks, batch_size=batch_size, tokens_ratio=tokens_ratio)

        # print('[MODEL]')
        # print('type:', type(self.model), end='\n\n')

        # print('[METHOD]')
        # print('type:', type(self.method), end='\n\n')

    def generate(self, x, target=None):
        with torch.enable_grad():
            prediction, saliency_map = self.method(x, class_idx=target)
            return prediction, saliency_map.detach().cpu() # origin
            