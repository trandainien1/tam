import torch
from torch.autograd.variable import Variable
from einops.layers.torch import Reduce, Rearrange
import torch.optim as optim
import torch.nn.functional as F

class AGCAM:
    """ Implementation of our method."""
    def __init__(self, model, attention_matrix_layer = 'before_softmax', attention_grad_layer = 'after_softmax', head_fusion='sum', layer_fusion='sum', 
                 learning_rate=0.1, 
                 max_iter = 100, device='cpu'):
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
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.device = device

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
        
    def generate(self, input_tensor, cls_idx=None):
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
        self.head=h
        self.width = int((d-1)**0.5)

        # put all matrices from each layer into one tensor
        self.attn_matrix.reverse()
        attn = self.attn_matrix[0]
        gradient = self.grad_attn[0]
        for i in range(1, len(self.attn_matrix)):
            attn = torch.concat((attn, self.attn_matrix[i]), dim=0)
            gradient = torch.concat((gradient, self.grad_attn[i]), dim=0)

        # As stated in Methodology, only positive gradients are used to reflect the positive contributions of each patch.
        # The self-attention score matrices are normalized with sigmoid and combined with the gradients.
        gradient = torch.nn.functional.relu(gradient) # Here, the variable gradient is the gradients alpha^{k,c}_h in Equation 7 in the methodology part.
        attn = torch.sigmoid(attn) # Here, the variable attn is the attention score matrices newly normalized with sigmoid, which are eqaul to the feature maps F^k_h in Equation 2 in the methodology part.
        mask = gradient * attn

        # aggregation of CAM of all heads and all layers and reshape the final CAM.
        mask = mask[:, :, :, 1:].unsqueeze(0)
        mask = Reduce('b l h z p -> b l z p', reduction=self.head_fusion)(mask)
        mask = Reduce('b l z p -> b z p', reduction=self.layer_fusion)(mask)
        mask = Rearrange('b z (h w) -> b z h w', h=self.width, w=self.width)(mask)
        return prediction, mask
    
    def nien_generate(self, input_tensor, cls_idx=None):
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

        print(f'Masks result: {mask.shape}')

        # aggregation of CAM of all heads and all layers and reshape the final CAM.
        mask = mask[:, :, :, 1:].unsqueeze(0) # * niên: chỗ này thêm 1 ở đầu (ví dụ: (2) -> (1, 2)) và 1: là bỏ token class


        # mask = mask[:, :, :, 1:] # * đang làm tính lỗi
        # print(mask.shape)

        # ------------- original -----------------------
        # mask = Reduce('b l h z p -> b l z p', reduction=self.head_fusion)(mask)
        # print(mask.shape)
        # mask = Reduce('b l z p -> b z p', reduction=self.layer_fusion)(mask)
        # print(mask.shape)
        # mask = Rerrange('b z (h w) -> b z h w', h=self.width, w=self.width)(mask)
        # print(mask.shape)
        
        # *Niên:Thay vì tính tổng theo blocks và theo head như công thức để ra 1 mask cuối cùng là CAM thì niên sẽ giữ lại tất cả các mask của các head ở mỗi block
        mask = Rearrange('b l hd z (h w)  -> b l hd z h w', h=self.width, w=self.width)(mask) # *Niên: chỗ này tách từng token (1, 196) thành từng patch (1, 14, 14)
        
        # mask = Rearrange('b l z (h w)  -> b l z h w', h=self.width, w=self.width)(mask) # *Niên: (12, 12, 1, 14, 14)
        # mask = mask.squeeze(2) # * Niên: (12, 12, 14, 14)
        print(f'kết quả cuối cùng trong generate: {mask.shape} ')

        return prediction, mask, output

    def __call__(self,
                images,
                labels,
                feature
                ):
        return self.forward(images, labels, feature)
    
    def __enter__(self):
        return self

    def combine_activations(self, feature, w, images):
        # softmax
        alpha = torch.nn.functional.softmax(w, dim=1).to(self.device)
        # sum (combination of feature)
        saliency_map = (alpha.repeat((1,1,feature.shape[2],feature.shape[3]))*feature).sum(axis=1).reshape((feature.shape[0],1,feature.shape[2],feature.shape[3]))
        # upsampling
        saliency_map = F.interpolate(saliency_map,size=(images.shape[2],images.shape[3]),mode='bilinear',align_corners=False)
        # normalize to 0-1
        norm_saliency_map = self.normalization(saliency_map)
        
        new_images = norm_saliency_map.repeat((1,images.shape[1],1,1)) * images
        return norm_saliency_map, new_images
    
    def f_logit_predict(self, model, device, x, predict_labels):
        outputs = model(x).to(device)
        one_hot_labels = torch.eye(len(outputs[0]))[predict_labels].to(device)
        j = torch.masked_select(outputs, one_hot_labels.bool())
        return j

    def get_f(self, x, y):
        return self.f_logit_predict(self.model, self.device, x, y)


    def get_loss(self, new_images, predict_labels, f_images):
        loss = torch.sum(f_images - self.get_f(new_images, predict_labels))
        return loss
        
    def forward(self, images, labels, feature):
        images = images.to(self.device)
        labels = labels.to(self.device)

        # w là alpha trong công thức tìm CAM
        w = Variable(0.5*torch.ones((feature.shape[0],feature.shape[1],1,1),dtype=torch.float), requires_grad=True)
        optimizer = optim.Adam([w], lr=self.learning_rate)
        prev = 1e10
        predict_labels = labels 
        # logit của hình ảnh ban đầu và dự đoán
        f_images = self.get_f(images, predict_labels)

        for step in range(self.max_iter):
            # CAM và các hình ảnh mới sau khi nhân CAM cho hình ảnh ban đầu
            norm_saliency_map, new_images = self.combine_activations(feature, w, images)
            loss = self.get_loss(new_images, predict_labels, f_images)
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()
            if step % (self.max_iter//10) == 0:
                if loss > prev:
                    print('Optimization stopped due to convergence...')
                    return norm_saliency_map, new_images
                prev = loss

            print('Learning Progress: %2.2f %%   ' %((step+1)/self.max_iter*100),end='\r')

        norm_saliency_map, new_images = self.combine_activations(feature, w, images)
        return norm_saliency_map, new_images
