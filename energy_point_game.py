'''
Implementation of Energy-based Pointing Game proposed in Score-CAM.
'''

import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torch.utils.data.sampler import RandomSampler, SubsetRandomSampler

import argparse
import numpy as np
import os
from glob import glob
import xml.etree.ElementTree as ET

from tqdm import tqdm
from PIL import Image
from matplotlib import pyplot as plt
import pandas as pd

from torchvision.datasets import ImageFolder
from bs4 import BeautifulSoup
import PIL
from torch.utils.data import Subset

import timm
import AGC_methods.AGCAM.ViT_for_AGCAM as ViT_Ours
from einops.layers.torch import Reduce, Rearrange
from torchvision.transforms import Resize

class ImageNetBboxDataset(Dataset):
    def __init__(self, img_path, anno_path, transform, num_samples=1, seed=0):
        print(f'ramdon seed: {seed}, num_samples: {num_samples}')
        np.random.seed(seed)
        
        imgs = glob.glob(os.path.join(img_path, '*.JPEG'))
        
        file_name = 'datasets/2000idx_ILSVRC2012.csv'
        indices = pd.read_csv(file_name, header=None)[0].to_numpy()
        # indices = np.random.randint(len(imgs), size=num_samples)

        self.imgs = np.array(imgs)[indices]
        self.annos = []
        for img in self.imgs:
            anno = os.path.join(anno_path, os.path.basename(img).replace('JPEG', 'xml'))
            self.annos.append(anno)
#         print(self.imgs)
#         print(self.annos)
        assert len(self.imgs) == len(self.annos), "length error"
        
        self.transform = transform

    def __getitem__(self, index):
        image = Image.open(self.imgs[index])
        image = image.convert('RGB')
        data = self.transform(image)
        
        return data, self.annos[index]
        
    def __len__(self):
        return len(self.imgs)
    
    
def parseXML(anno):
    tree = ET.parse(anno)
    root = tree.getroot()
    fileName = root.find("filename").text
    
    size = root.find('size')
    width = float(size.find('width').text)
    height = float(size.find('height').text)
    
    bboxs = []
    for obj in root.findall("object"):
        name = obj.find("name").text
        bndbox = obj.find('bndbox')
        xmin = float(bndbox.find('xmin').text)/width
        ymin = float(bndbox.find('ymin').text)/height
        xmax = float(bndbox.find('xmax').text)/width
        ymax = float(bndbox.find('ymax').text)/height
        bboxs.append([xmin, ymin, xmax, ymax])
    return bboxs
    

class InterpretTransformer(object):
    def __init__(self, model):
        self.model = model
        self.model.eval()
        
    def transition_attention_maps(self, input, index=None, start_layer=4, steps=20, with_integral=True, first_state=False):
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
    
        sal = F.interpolate(states[:, 0, 1:].reshape(-1, 1, 14, 14), scale_factor=16, mode='bilinear').cuda()
        return sal.reshape(-1, 224, 224)

    
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

        num_blocks = 12
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
            
        sal = F.interpolate(attrs[:, 0, 1:].reshape(-1, 1, 14, 14), scale_factor=16, mode='bilinear').cuda()
        return sal.reshape(-1, 224, 224)
    
    
    def rollout(self, input, index=None, start_layer=0, add_residual=True):
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

        num_blocks = 12
        attrs = torch.eye(s).expand(b, h, s, s).cuda()
        for i in range(start_layer, num_blocks):
            attr = self.model.blocks[i].attn.get_attention_map()

            # add residual
            if add_residual:
                eye = torch.eye(s).expand(b, h, s, s).cuda()
                attr = attr + eye
                attr = attr / attr.sum(dim=-1, keepdim=True)

            attrs = (attr @ attrs)

        attrs = attrs.mean(1)
        
        sal = F.interpolate(attrs[:, 0, 1:].reshape(-1, 1, 14, 14), scale_factor=16, mode='bilinear').cuda()
        return sal.reshape(-1, 224, 224)
    
    
    def raw_attn(self, input, index=None):
        output = self.model(input, register_hook=True)

        attr = self.model.blocks[-1].attn.get_attention_map().mean(dim=1) 
    
        sal = F.interpolate(attr[:, 0, 1:].reshape(-1, 1, 14, 14), scale_factor=16, mode='bilinear').cuda()
        return sal.reshape(-1, 224, 224)

'''
bbox (list): upper left and lower right coordinates of object bounding box
saliency_map (array): explanation map, ignore the channel
'''
def box_to_seg(box_cor):
    segmask = torch.zeros((224, 224))
    if box_cor.dim()!=1:
        n, _ = box_cor.shape
        for i in range(n):
            xmin = box_cor[i][0]
            ymin = box_cor[i][1]
            xmax = box_cor[i][2]
            ymax = box_cor[i][3]
            segmask[ymin:ymax+1, xmin:xmax+1]=1
    
    return segmask

def energy_point_game(bboxes_batch, saliency_map):
  
    print('[DEBUG]', saliency_map.shape)
    b, w, h = saliency_map.shape

    gt = torch.zeros(b, h, w)
    
    precisions = []
    recalls = []
    f1_scores = []
    


    # print('[DEBUG] bboxes patch shape: ', bboxes_batch.shape)
    for i in range(b):
        # for bboxes in bboxes_batch:
        #     for bbox in bboxes:
        #         print('[DEBUG] bbox shape: ', bbox.shape)
        #         x1, y1, x2, y2 = map(lambda x: int(x * 224), bbox)
        #         gt[i, y1:y2, x1:x2] = 1
        gt = box_to_seg(bboxes_batch)

        TP = (saliency_map * gt).sum()  

        predict_pos = saliency_map.sum()
        actual_pos = gt.sum()
        
        precision = float((TP / predict_pos).detach().numpy())
        recall = float((TP / actual_pos).detach().numpy())
        f1_score = (2*precision*recall) / (precision + recall + 1e-6)
        
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1_score)

    return precisions, recalls, f1_scores

class ImageNetDataset_val(ImageFolder):
    def __init__(self, root_dir, transforms=None):
        self.img_dir = os.path.join(root_dir, "Data", "CLS-LOC", "val")
        self.annotation_dir = os.path.join(root_dir, "Annotations", "CLS-LOC", "val")
        self.classes = sorted(os.listdir(self.img_dir))
        
        self.transforms = transforms
        self.img_data = []
        self.img_labels = []

        for idx, cls in enumerate(self.classes):
            # self.class_name.append(cls)
            img_cls_dir = os.path.join(self.img_dir, cls)
            for img in glob(os.path.join(img_cls_dir, '*.JPEG')):
                self.img_data.append(img)
                self.img_labels.append(idx)


    def __getitem__(self, idx):
        img_path, label = self.img_data[idx], self.img_labels[idx]
        # print('[DEBUG]', img_path)
        img = PIL.Image.open(img_path).convert('RGB')
        # img.show()
        width, height = img.size
        img_name = img_path.split('/')[-1].split('.')[0]
        anno_path = os.path.join(self.annotation_dir, img_name+".xml")
        with open(anno_path, 'r') as f:
            file = f.read()
        soup = BeautifulSoup(file, 'html.parser')
        if self.transforms:
            img = self.transforms(img)
        objects = soup.findAll('object')
        
        bnd_box = torch.tensor([])

        for object in objects:
            xmin = int(object.bndbox.xmin.text)
            ymin = int(object.bndbox.ymin.text)
            xmax = int(object.bndbox.xmax.text)
            ymax = int(object.bndbox.ymax.text)
            xmin = int(xmin/width*224)
            ymin = int(ymin/height*224)
            xmax = int(xmax/width*224)
            ymax = int(ymax/height*224)
            if bnd_box.dim()==1:
                bnd_box = torch.tensor((xmin, ymin, xmax, ymax)).unsqueeze(0)
            else:
                bnd_box = torch.cat((bnd_box, torch.tensor((xmin, ymin, xmax, ymax)).unsqueeze(0)), dim=0)
        # print(bnd_box.shape)
        sample = {
            'image': img, 
            'label': label, 
            'filename': img_name, 
            'num_objects': len(objects), 
            'bnd_box': bnd_box, 
            'img_path': img_path
            }
        return sample

    def __len__(self):
        return len(self.img_data)
    
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
            # print("class idx", class_idx)

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
        
        return saliency_map

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='insertion and deletion evaluation')
    parser.add_argument('--method', type=str,
            default='tam',
            choices=[
                'tam', 
                'attribution', 
                'raw_attn', 
                'rollout',
                'better_agc'
            ],
            help='')
    parser.add_argument('--batch_size', type=int,
                        default=8,
                        help='')
    
    parser.add_argument('--num_samples', type=int,
                        default=2000,
                        help='')
    parser.add_argument('--seed', type=int,
                    default=0,
                    help='random seed')
    
    args = parser.parse_args()
    
    if args.method in [
        'tam',
        'raw_attn', 
        'rollout'
    ]:
        from baselines.ViT.ViT_new import vit_base_patch16_224

        model = vit_base_patch16_224(pretrained=True).cuda()
    elif 'agc' in args.method:
        MODEL = 'vit_base_patch16_224'
        timm_model = timm.create_model(model_name='vit_base_patch16_224', pretrained=True, pretrained_cfg='orig_in21k_ft_in1k')
        state_dict = timm_model.state_dict()
        model = ViT_Ours.create_model(MODEL, pretrained=True, num_classes=1000)
        model.load_state_dict(state_dict, strict=True)
        model = model.eval()
        model = model.cuda()
    else:
        from baselines.ViT.ViT_LRP import vit_base_patch16_224 as vit_LRP
        
        model = vit_LRP(pretrained=True).cuda()
    
    if args.method == 'better_agc':
        it = BetterAGC(model)
    else:
        it = InterpretTransformer(model)
    print(f'explanation method: {args.method}')
    
    # Image preprocessing function
    # preprocess = transforms.Compose([
    #     transforms.Resize((224, 224)),
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=[0.5, 0.5, 0.5],
    #                          std=[0.5, 0.5, 0.5]),
    # ])
    transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    batch_size = args.batch_size
    num_samples = args.num_samples
    
    # dataset = ImageNetBboxDataset(
    #     img_path='/kaggle/input/ilsvrc/ILSVRC/Data',
    #     anno_path='/kaggle/input/ilsvrc/ILSVRC/Annotations',
    #     transform=preprocess,
    #     num_samples=args.num_samples,
    #     seed=args.seed
    # )

    validset = ImageNetDataset_val(
    # root_dir='./ILSVRC',
    root_dir='/kaggle/input/ilsvrc/ILSVRC',
    transforms=transform,
    )

    validloader = DataLoader(
        dataset = validset,
        batch_size=1,
        shuffle = False,
    )

    subset_indices = pd.read_csv('datasets/2000idx_ILSVRC2012.csv', header=None)[0].to_numpy()
    subset = Subset(validloader.dataset, subset_indices)
    subset_loader = torch.utils.data.DataLoader(subset, batch_size=1, shuffle=False)
    
    # Load batch of images
    # data_loader = torch.utils.data.DataLoader(
    #     dataset, 
    #     batch_size=batch_size, 
    #     shuffle=False,
    #     num_workers=8
    # )
    
    scores = []
    p, r, f1 = [], [], []
    # iterator = tqdm(data_loader, total=len(data_loader))
    iterator = tqdm(subset_loader, total=len(subset_loader))
    # for j, (img, annos) in enumerate(iterator):
    for data in iterator:
        # bboxes = []
        # for anno in annos:
        #     bboxes.append(parseXML(anno))
        img = data['image'].to('cuda')
        bboxes = data['bnd_box'].to('cuda').squeeze(0)

        if args.method == 'tam':
            Res = it.transition_attention_maps(img.cuda(), start_layer=4)
        elif args.method == 'raw_attn':
            Res = it.raw_attn(img.cuda())
        elif args.method == 'rollout':
            Res = it.rollout(img.cuda())
        elif args.method == 'attribution':
            Res = it.attribution(img.cuda())
        elif args.method == 'better_agc':
            saliency_map = it(img.cuda()) #saliency_map.shape = [14, 14]
            saliency_map = saliency_map.reshape((1, *saliency_map.shape)) #saliency_map.shape = [1, 14, 14]
            saliency_map = saliency_map.reshape((1, *saliency_map.shape)) #saliency_map.shape = [1, 1, 14, 14]
            
            # Reshape the mask to have the same size with the original input image (224 x 224)
            upsample = torch.nn.Upsample(224, mode = 'bilinear', align_corners=False)
            saliency_map = upsample(saliency_map)
            
            # saliency_map = saliency_map.cpu().detach().numpy()
            Res = saliency_map
        
            
        # threshold between FG and BG is the mean    
        Res = (Res - Res.min()) / (Res.max() - Res.min())

        ret = Res.max() * 0.3

        # greater than: Computes input > other element-wise.
        Res_1 = Res.gt(ret).type(Res.type())
        # less than
        Res_0 = Res.le(ret).type(Res.type())

        Res_1_AP = Res
        Res_0_AP = 1 - Res

        Res_1[Res_1 != Res_1] = 0
        Res_0[Res_0 != Res_0] = 0
        Res_1_AP[Res_1_AP != Res_1_AP] = 0
        Res_0_AP[Res_0_AP != Res_0_AP] = 0

        output = torch.cat((Res_0, Res_1), 1)
        output_AP = torch.cat((Res_0_AP, Res_1_AP), 1)
        
        _1, _2, _3 = energy_point_game(bboxes, Res_1.cpu().detach())
        p += _1
        r += _2
        f1 += _3
        
        iterator.set_description('P: %.4f, R: %4f, F1: %4f' % (np.mean(p), np.mean(r), np.mean(f1)))
        
    print('----------------------------------------------------------------')
    print('mean: P: %.5f, R: %5f, F1: %5f' % (np.mean(p), np.mean(r), np.mean(f1)))
    print('std: P: %.5f, R: %5f, F1: %5f' % (np.std(p), np.std(r), np.std(f1)))