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
  
    b, w, h = saliency_map.shape

    gt = torch.zeros(b, h, w)
    
    precisions = []
    recalls = []
    f1_scores = []
    


    print('[DEBUG] bboxes patch shape: ', bboxes_batch.shape)
    for i in range(b):
        # for bboxes in bboxes_batch:
        #     for bbox in bboxes:
        #         print('[DEBUG] bbox shape: ', bbox.shape)
        #         x1, y1, x2, y2 = map(lambda x: int(x * 224), bbox)
        #         gt[i, y1:y2, x1:x2] = 1
        gt = box_to_seg(bboxes_batch).to('cuda')

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
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='insertion and deletion evaluation')
    parser.add_argument('--method', type=str,
            default='tam',
            choices=[
                'tam', 
                'attribution', 
                'raw_attn', 
                'rollout'
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
        pass
    else:
        from baselines.ViT.ViT_LRP import vit_base_patch16_224 as vit_LRP
        
        model = vit_LRP(pretrained=True).cuda()
    
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
    # for j, (img, annos) in enumerate(iterator):
    for data in tqdm(subset_loader):
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