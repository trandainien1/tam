import sys
import os
import torch
from torch import nn
from tqdm import tqdm
from scipy.ndimage.filters import gaussian_filter
import numpy as np
import torch.nn.functional as F
from matplotlib import pyplot as plt
from torch.utils.data.sampler import RandomSampler, SubsetRandomSampler
from torchvision import transforms, datasets
from PIL import Image

import argparse


import pandas as pd
import torch

from os import path

from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.datasets.imagenet import ImageNet

import timm
import AGC_methods.AGCAM.ViT_for_AGCAM as ViT_Ours

# -------------------- datasets ---------------------
datasets_dict = {
    'imagenet': {
        'class_fn': ImageNet,
        'n_output': 1000,
        'split': 'val',
        'indices_csv': 'datasets/2000idx_ILSVRC2012.csv',
        'transform': transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((224, 224)),
            transforms.Normalize( (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
    }
}

def get_dataset(name, root):
    cur_dict = datasets_dict[name]
    if name=='imagenet':
        dataset = ImageNet(path.join(root, name), split=cur_dict['split'], transform=cur_dict['transform'])
    try:
        file_name = cur_dict['indices_csv']
        subset_indices = pd.read_csv(file_name, header=None)[0].to_numpy()
        subset = torch.utils.data.Subset(dataset, subset_indices)
        print(f'[DATASET] load dataset from files csv {file_name}')
        return subset, cur_dict["n_output"]
    except:
        print(f'[DATASET] load WHOLE dataset')
        return dataset, cur_dict["n_output"]

class XAIDataset(Dataset):
    def __init__(self, dataset, xai):
        self.dataset = dataset
        self.xai = xai
    def __len__(self):
        return len(self.dataset)
    def __getitem__(self, idx):
        return self.dataset[idx], self.xai[idx]
# -------------------- datasets ---------------------

# blur
def gkern(klen, nsig):
    """Returns a Gaussian kernel array.
    Convolution with it results in image blurring."""
    # create nxn zeros
    inp = np.zeros((klen, klen))
    # set element at the middle to one, a dirac delta
    inp[klen//2, klen//2] = 1
    # gaussian-smooth the dirac, resulting in a gaussian filter mask
    k = gaussian_filter(inp, nsig)
    kern = np.zeros((3, 3, klen, klen))
    kern[0, 0] = k
    kern[1, 1] = k
    kern[2, 2] = k
    return torch.from_numpy(kern.astype('float32'))

klen = 11
ksig = 5
kern = gkern(klen, ksig)

# Function that blurs input image
blur = lambda x: nn.functional.conv2d(x, kern, padding=klen//2)



# Given label number returns class name
def get_class_name(c):
    labels = np.loadtxt('synset_words.txt', str, delimiter='\t')
    return ' '.join(labels[c].split(',')[0].split()[1:])

def auc(arr):
    """Returns normalized Area Under Curve of the array."""
    return (arr.sum() - arr[0] / 2 - arr[-1] / 2) / (arr.shape[0] - 1)

class CausalMetric():

    def __init__(self, model, mode, step, substrate_fn):
        r"""Create deletion/insertion metric instance.
        Args:
            model (nn.Module): Black-box model being explained.
            mode (str): 'del' or 'ins'.
            step (int): number of pixels modified per one iteration.
            substrate_fn (func): a mapping from old pixels to new pixels.
        """
        assert mode in ['del', 'ins']
        self.model = model
        self.mode = mode
        self.step = step
        self.substrate_fn = substrate_fn
        
    def evaluate(self, img_batch, exp_batch):
        r"""Efficiently evaluate big batch of images.Z
        Args:
            img_batch (Tensor): batch of images.
            exp_batch (np.ndarray): batch of explanations.
            batch_size (int): number of images for one small batch.
        Returns:
            scores (nd.array): Array containing scores at every step for every image.
        """
        n_samples = img_batch.shape[0]
        predictions = torch.FloatTensor(n_samples, n_classes)
        preds = self.model(img_batch.cuda())
        preds = F.softmax(preds, dim=1).cpu().detach()
        predictions = preds
        top = np.argmax(predictions, -1)
        n_steps = (HW + self.step - 1) // self.step
        scores = np.empty((n_steps + 1, n_samples))
        salient_order = np.flip(np.argsort(exp_batch.reshape(-1, HW), axis=1), axis=-1)
        r = np.arange(n_samples).reshape(n_samples, 1)

        substrate = torch.zeros_like(img_batch)
        substrate = self.substrate_fn(img_batch)

        if self.mode == 'del':
            caption = 'Deleting  '
            start = img_batch.clone()
            finish = substrate
        elif self.mode == 'ins':
            caption = 'Inserting '
            start = substrate
            finish = img_batch.clone()

        # While not all pixels are changed
        for i in range(n_steps+1):
            # Compute new scores
            preds = self.model(start.cuda())
            preds = F.softmax(preds, dim=1).cpu().detach().numpy()
            scores[i] = preds[range(n_samples), top]
            # Change specified number of most salient pixels to substrate pixels
            coords = salient_order[:, self.step * i:self.step * (i + 1)]
            start.cpu().detach().numpy().reshape(n_samples, 3, HW)[r, :, coords] = finish.cpu().detach().numpy().reshape(n_samples, 3, HW)[r, :, coords]
#         print('AUC: {}'.format(auc(scores.mean(1))))
        return scores

class InterpretTransformer(object):
    def __init__(self, model, img_size=224):
        self.model = model
        self.model.eval()
        self.img_size=img_size
    
    def transition_attention_maps(self, input, index=None, start_layer=0, steps=20, with_integral=True, first_state=False):
        b = input.shape[0]
        output = self.model(input, register_hook=True)
        if index == None:
            index = np.argmax(output.cpu().data.numpy(), axis=-1)

        one_hot = np.zeros((b, output.size()[-1]), dtype=np.float32)
        one_hot[np.arange(b), index] = 1
        one_hot_vector = one_hot
        one_hot = torch.from_numpy(one_hot).requires_grad_(True)
        one_hot = torch.sum(one_hot.cuda() * output)

        self.model.zero_grad()
        one_hot.backward(retain_graph=True)

        b, h, s, _ = self.model.blocks[-1].attn.get_attention_map().shape

        num_blocks = len(self.model.blocks)

        states = self.model.blocks[-1].attn.get_attention_map().mean(1)[:, 0, :].reshape(b, 1, s)
        for i in range(start_layer, num_blocks-1)[::-1]:
            attn = self.model.blocks[i].attn.get_attention_map().mean(1)

            states_ = states
            states = states.bmm(attn)
            states += states_

        total_gradients = torch.zeros(b, h, s, s).cuda()
        for alpha in np.linspace(0, 1, steps):        
            # forward propagation
            data_scaled = input * alpha

            # backward propagation
            output = self.model(data_scaled, register_hook=True)
            one_hot = np.zeros((b, output.size()[-1]), dtype=np.float32)
            one_hot[np.arange(b), index] = 1
            one_hot_vector = one_hot
            one_hot = torch.from_numpy(one_hot).requires_grad_(True)
            one_hot = torch.sum(one_hot.cuda() * output)

            self.model.zero_grad()
            one_hot.backward(retain_graph=True)

            # cal grad
            gradients = self.model.blocks[-1].attn.get_attn_gradients()
            total_gradients += gradients

        if with_integral:
            W_state = (total_gradients / steps).clamp(min=0).mean(1)[:, 0, :].reshape(b, 1, s)
        else:
            W_state = self.model.blocks[-1].attn.get_attn_gradients().clamp(min=0).mean(1)[:, 0, :].reshape(b, 1, s)
        
        if first_state:
            states = self.model.blocks[-1].attn.get_attention_map().mean(1)[:, 0, :].reshape(b, 1, s)
        
        states = states * W_state
    
        sal = F.interpolate(states[:, 0, 1:].reshape(-1, 1, self.img_size//16, self.img_size//16), scale_factor=16, mode='bilinear').cuda()
        return sal.reshape(-1, self.img_size, self.img_size).cpu().detach().numpy()
    
    
    def attribution(self, input, index=None, start_layer=0):
        b = input.shape[0]
        output = self.model(input)
        kwargs = {"alpha": 1}
        if index == None:
            index = np.argmax(output.cpu().data.numpy(), axis=-1)

        one_hot = np.zeros((b, output.size()[-1]), dtype=np.float32)
        one_hot[np.arange(b), index] = 1
        one_hot_vector = one_hot
        one_hot = torch.from_numpy(one_hot).requires_grad_(True)
        one_hot = torch.sum(one_hot.cuda() * output)

        self.model.zero_grad()
        one_hot.backward(retain_graph=True)

        self.model.relprop(torch.tensor(one_hot_vector).cuda(), **kwargs)

        b, h, s, _ = self.model.blocks[-1].attn.get_attn_gradients().shape

        num_blocks = len(self.model.blocks)
        # first_block
        attn = self.model.blocks[start_layer].attn.get_attn_cam()
        grad = self.model.blocks[start_layer].attn.get_attn_gradients()
        attr = (grad * attn).clamp(min=0).mean(1)
        # add residual
        eye = torch.eye(s).expand(b, s, s).cuda()
        attr = attr + eye
        attr = attr / attr.sum(dim=-1, keepdim=True)
        
        attrs = attr
        for i in range(start_layer+1, num_blocks):
            attn = self.model.blocks[i].attn.get_attn_cam()
            grad = self.model.blocks[i].attn.get_attn_gradients()
            attr = (grad * attn).clamp(min=0).mean(1)
            # add residual
            eye = torch.eye(s).expand(b, s, s).cuda()
            attr = attr + eye
            attr = attr / attr.sum(dim=-1, keepdim=True)
            
            attrs = attr.bmm(attrs)
            
        sal = F.interpolate(attrs[:, 0, 1:].reshape(-1, 1, self.img_size//16, self.img_size//16), scale_factor=16, mode='bilinear').cuda()
        return sal.reshape(-1, self.img_size, self.img_size).cpu().detach().numpy()
    
    
    def raw_attn(self, input, index=None):
        b = input.shape[0]
        output = self.model(input, register_hook=True)

        attrs = self.model.blocks[-1].attn.get_attention_map().mean(dim=1)
    
        sal = F.interpolate(attrs[:, 0, 1:].reshape(-1, 1, self.img_size//16, self.img_size//16), scale_factor=16, mode='bilinear').cuda()
        return sal.reshape(-1, self.img_size, self.img_size).cpu().detach().numpy()
    
    def rollout(self, input, index=None, start_layer=0):
        b = input.shape[0]
        output = self.model(input, register_hook=True)
        if index == None:
            index = np.argmax(output.cpu().data.numpy(), axis=-1)

        one_hot = np.zeros((b, output.size()[-1]), dtype=np.float32)
        one_hot[np.arange(b), index] = 1
        one_hot_vector = one_hot
        one_hot = torch.from_numpy(one_hot).requires_grad_(True)
        one_hot = torch.sum(one_hot.cuda() * output)

        self.model.zero_grad()
        one_hot.backward(retain_graph=True)

        b, h, s, _ = self.model.blocks[-1].attn.get_attn_gradients().shape

        num_blocks = len(self.model.blocks)
        attrs = torch.eye(s).expand(b, h, s, s).cuda()
        for i in range(start_layer, num_blocks):
            attr = self.model.blocks[i].attn.get_attention_map()

            eye = torch.eye(s).expand(b, h, s, s).cuda()
            attr = attr + eye
            attr = attr / attr.sum(dim=-1, keepdim=True)

            attrs = (attr @ attrs)

        attrs = attrs.mean(1)
        
        sal = F.interpolate(attrs[:, 0, 1:].reshape(-1, 1, self.img_size//16, self.img_size//16), scale_factor=16, mode='bilinear').cuda()
        return sal.reshape(-1, self.img_size, self.img_size).cpu().detach().numpy()


class BetterAGC:
    def __init__(self, model, attention_matrix_layer = 'before_softmax', attention_grad_layer = 'after_softmax', head_fusion='sum', layer_fusion='sum'):
        """
        Args:
            model (nn.Module): the Vision Transformer model to be explained
            attention_matrix_layer (str): the name of the layer to set a forward hook to get the self-attention matrices
            attention_grad_layer (str): the name of the layer to set a backward hook to get the gradients
            head_fusion (str): type of head-wise aggregation (default: 'sum')
            layer_fusion (str): type of layer-wise aggregation (default: 'sum')
        """
        self.model = model
        self.head = None
        self.width = None
        self.head_fusion = head_fusion
        self.layer_fusion = layer_fusion
        self.attn_matrix = []
        self.grad_attn = []

        for layer_num, (name, module) in enumerate(self.model.named_modules()):
            if attention_matrix_layer in name:
                module.register_forward_hook(self.get_attn_matrix)
            if attention_grad_layer in name:
                module.register_full_backward_hook(self.get_grad_attn)

    def get_attn_matrix(self, module, input, output):
        # As stated in Methodology part, in ViT with [class] token, only the first row of the attention matrix is directly connected with the MLP head.
        self.attn_matrix.append(output[:, :, 0:1, :]) # shape: [batch, num_heads, 1, num_patches]


    def get_grad_attn(self, module, grad_input, grad_output):
        # As stated in Methodology part, in ViT with [class] token, only the first row of the attention matrix is directly connected with the MLP head.
        self.grad_attn.append(grad_output[0][:, :, 0:1, :]) # shape: [batch, num_heads, 1, num_patches]


    def generate_cams_of_heads(self, input_tensor, cls_idx=None):
        self.attn_matrix = []
        self.grad_attn = []

        # backpropagate the model from the classification output
        self.model.zero_grad()
        output = self.model(input_tensor)
        _, prediction = torch.max(output, 1)
        self.prediction = prediction
        if cls_idx==None:                               # generate CAM for a certain class label
            loss = output[0, prediction[0]]
        else:                                           # generate CAM for the predicted class
            loss = output[0, cls_idx]
        loss.backward()

        b, h, n, d = self.attn_matrix[0].shape
        # b, h, n, d = self.attn_matrix.shape
        self.head=h
        self.width = int((d-1)**0.5)

        # put all matrices from each layer into one tensor
        self.attn_matrix.reverse()
        attn = self.attn_matrix[0]
        # attn = self.attn_matrix
        gradient = self.grad_attn[0]
        # gradient = self.grad_attn
        # layer_index = 2
        for i in range(1, len(self.attn_matrix)):
        # for i in range(layer_index, layer_index+1):
            # print('hia')
            attn = torch.concat((attn, self.attn_matrix[i]), dim=0)
            gradient = torch.concat((gradient, self.grad_attn[i]), dim=0)

        # As stated in Methodology, only positive gradients are used to reflect the positive contributions of each patch.
        # The self-attention score matrices are normalized with sigmoid and combined with the gradients.
        gradient = torch.nn.functional.relu(gradient) # Here, the variable gradient is the gradients alpha^{k,c}_h in Equation 7 in the methodology part.
        attn = torch.sigmoid(attn) # Here, the variable attn is the attention score matrices newly normalized with sigmoid, which are eqaul to the feature maps F^k_h in Equation 2 in the methodology part.
        mask = gradient * attn

        # aggregation of CAM of all heads and all layers and reshape the final CAM.
        mask = mask[:, :, :, 1:].unsqueeze(0) # * niên: chỗ này thêm 1 ở đầu (ví dụ: (2) -> (1, 2)) và 1: là bỏ token class
        # print(mask.shape)

        # *Niên:Thay vì tính tổng theo blocks và theo head như công thức để ra 1 mask cuối cùng là CAM thì niên sẽ giữ lại tất cả các mask của các head ở mỗi block
        mask = Rearrange('b l hd z (h w)  -> b l hd z h w', h=self.width, w=self.width)(mask) # *Niên: chỗ này tách từng token (1, 196) thành từng patch (1, 14, 14)

        return prediction, mask, output

    def generate_scores(self, head_cams, prediction, output_truth, image):
        with torch.no_grad():
            tensor_heatmaps = head_cams[0]
            tensor_heatmaps = tensor_heatmaps.reshape(144, 1, 14, 14)
            tensor_heatmaps = transforms.Resize((224, 224))(tensor_heatmaps)
    
            # Compute min and max along each image
            min_vals = tensor_heatmaps.amin(dim=(2, 3), keepdim=True)  # Min across width and height
            max_vals = tensor_heatmaps.amax(dim=(2, 3), keepdim=True)  # Max across width and height
            # Normalize using min-max scaling
            tensor_heatmaps = (tensor_heatmaps - min_vals) / (max_vals - min_vals + 1e-7)  # Add small value to avoid division by zero
            # print("before multiply img with mask: ")
            # print(torch.cuda.memory_allocated()/1024**2)
            m = torch.mul(tensor_heatmaps, image)
            # print("After multiply img with mask scores: ")
            # print(torch.cuda.memory_allocated()/1024**2)

            with torch.no_grad():
                output_mask = self.model(m)
            
            # print("After get output from model: ")
            # print(torch.cuda.memory_allocated()/1024**2)
    
            agc_scores = output_mask[:, prediction.item()] - output_truth[0, prediction.item()]
            agc_scores = torch.sigmoid(agc_scores)
    
            agc_scores = agc_scores.reshape(head_cams[0].shape[0], head_cams[0].shape[1])

            del output_mask  # Delete unnecessary variables that are no longer needed
            torch.cuda.empty_cache()  # Clean up cache if necessary
            # print("After deleted output from model: ")
            # print(torch.cuda.memory_allocated()/1024**2)
            
            return agc_scores

    def generate_saliency(self, head_cams, agc_scores):
        mask = (agc_scores.view(12, 12, 1, 1, 1) * head_cams[0]).sum(axis=(0, 1))

        mask = mask.squeeze()
        return mask

    def __call__(self, x, class_idx=None):

        print('[DEBUG]', x.shape)
        # Check that we get only one image
        assert x.dim() == 3 or (x.dim() == 4 and x.shape[0] == 1), "Only one image can be processed at a time"

        # Unsqueeze to get 4 dimensions if needed
        if x.dim() == 3:
            x = x.unsqueeze(dim=0)

        with torch.enable_grad():
            predicted_class, head_cams, output_truth = self.generate_cams_of_heads(x)

        # print("After generate cams: ")
        # print(torch.cuda.memory_allocated()/1024**2)
        # print()
        
        # Define the class to explain. If not explicit, use the class predicted by the model
        if class_idx is None:
            class_idx = predicted_class
            print("class idx", class_idx)

        # Generate the saliency map for image x and class_idx
        scores = self.generate_scores(
            image=x,
            head_cams=head_cams,
            prediction=predicted_class, output_truth=output_truth
        )
        # print("After generate scores: ")
        # print(torch.cuda.memory_allocated()/1024**2)
        # print()
        
        saliency_map = self.generate_saliency(head_cams=head_cams, agc_scores=scores)
        # print("After generate saliency maps: ")
        # print(torch.cuda.memory_allocated()/1024**2)
        # print()
        return saliency_map.reshape(-1, 224, 224).cpu().detach().numpy()

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='insertion and deletion evaluation')
    parser.add_argument('--method', type=str,
            default='tam',
            choices=['tam',
                     'rollout',
                     'raw_attn',
                     'attribution', 'better_agc'],
            help='')
    parser.add_argument('--batch_size', type=int,
                        default=16,
                        help='')
    parser.add_argument('--num_samples', type=int,
                        default=2000,
                        help='')
    parser.add_argument('--blur', action='store_true',
                        default=False,
                        help='')
    
    parser.add_argument('--arch', type=str,
            default='vit_base_patch16_224',
            choices=['vit_base_patch16_224',
                     'vit_base_patch16_384',
                     'vit_large_patch16_224',
                     'deit_base_patch16_224'],
            help='')
    
    args = parser.parse_args()

     # ---------------------------------   Load model  ------------------------------------    
    MODEL = 'vit_base_patch16_224'

    if args.method in [
        'tam', 'raw_attn', 'rollout'
    ]:
        from baselines.ViT.ViT_new import vit_base_patch16_224, vit_large_patch16_224, deit_base_patch16_224, vit_base_patch16_384
        model = eval(args.arch)(pretrained=True).cuda()
    elif 'agc' in args.method:
        timm_model = timm.create_model(model_name='vit_base_patch16_224', pretrained=True, pretrained_cfg='orig_in21k_ft_in1k')
        state_dict = timm_model.state_dict()
        model = ViT_Ours.create_model(MODEL, pretrained=True, num_classes=1000)
        model.load_state_dict(state_dict, strict=True)
    else:
        from baselines.ViT.ViT_LRP import vit_base_patch16_224, vit_large_patch16_224, deit_base_patch16_224, vit_base_patch16_384
        model = eval(args.arch)(pretrained=True).cuda()

    # ---------------------------------   Load model  ------------------------------------

    if args.arch == 'vit_base_patch16_384':
        img_size = 384
    else:
        img_size = 224
        
    HW = img_size * img_size 

    n_classes = 1000

    if args.method == 'better_agc':
        it = BetterAGC(model)
    else:
        it = InterpretTransformer(model, img_size)
    
    preprocess = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                             std=[0.5, 0.5, 0.5]),
    ])
    
    batch_size = args.batch_size
    num_samples = args.num_samples

    # blur
    if args.blur:
        print("use blur insertion")
        insertion = CausalMetric(model, 'ins', img_size * 8, substrate_fn=blur)
    else:
        print("use zero insertion")
        insertion = CausalMetric(model, 'ins', img_size * 8, substrate_fn=torch.zeros_like)
    
    deletion = CausalMetric(model, 'del', img_size * 8, substrate_fn=torch.zeros_like)

    scores = {'del': [], 'ins': []}

    # ----------- get dataset -----------
    # dataset = datasets.ImageFolder('/root/datasets/ImageNet/val', preprocess)

    dataset, n_output = get_dataset(name='imagenet', root='.')

    np.random.seed(0)
    # max_index = np.random.randint(num_samples, len(dataset))
    # print("subset indices: ", [max_index-num_samples, max_index])
    # sub_dataset = torch.utils.data.Subset(dataset, indices=range(max_index-num_samples, max_index))
    sub_dataset = dataset
    
    # Load batch of images
    data_loader = torch.utils.data.DataLoader(
        sub_dataset, batch_size=batch_size, shuffle=True,
        num_workers=4, pin_memory=True)

    images = np.empty((len(data_loader), batch_size, 3, img_size, img_size))
    iterator = tqdm(data_loader, total=len(data_loader))

    for j, (img, _) in enumerate(iterator):
        if args.method == 'tam':
            exp = it.transition_attention_maps(img.cuda())
        elif args.method == 'raw_attn':
            exp = it.raw_attn(img.cuda()) 
        elif args.method == 'rollout':
            exp = it.rollout(img.cuda()) 
        elif args.method == 'attribution':
            exp = it.attribution(img.cuda())
        elif args.method == 'better_agc':
            exp = it(img.cuda())

        # Evaluate deletion
        h = deletion.evaluate(img, exp)
        scores['del'].append(auc(h.mean(1)))

        # Evaluate insertion
        h = insertion.evaluate(img, exp)
        scores['ins'].append(auc(h.mean(1)))
        iterator.set_description('del: %.4f, ins: %.4f' % (np.mean(scores['del']), np.mean(scores['ins'])))
        
    np.save(os.path.join('del_ins_results', args.method + '.npy'), scores)
    print('----------------------------------------------------------------')
    print('Final - {}:\nDeletion - {:.5f}\nInsertion - {:.5f}'.format(args.method, np.mean(scores['del']), np.mean(scores['ins'])))