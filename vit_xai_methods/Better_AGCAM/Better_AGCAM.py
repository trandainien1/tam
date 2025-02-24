import torch
from einops.layers.torch import Reduce, Rearrange
import torchvision.transforms as transforms
import numpy as np
import timm 
import torch.nn.functional as F
from sklearn.cluster import AgglomerativeClustering
from fast_pytorch_kmeans import KMeans


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

        return predicted_class, saliency_map


class BetterAGC_softmax:
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
            # agc_scores = torch.sigmoid(agc_scores)
            agc_scores = F.softmax(agc_scores)

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

        return predicted_class, saliency_map



class ScoreAGC:
    def __init__(self, model, attention_matrix_layer = 'before_softmax', attention_grad_layer = 'after_softmax', head_fusion='sum', layer_fusion='sum', normalize_cam_heads=True, score_minmax_norm=True, add_noise=False, plus=1, vitcx_score_formula=False, is_head_fuse=False):
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
        self.normalize_cam_heads = normalize_cam_heads
        self.score_minmax_norm = score_minmax_norm
        self.add_noise = add_noise
        self.plus = plus
        self.vitcx_score_formula = vitcx_score_formula
        self.is_head_fuse = is_head_fuse

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
        if self.is_head_fuse:
            mask = Reduce('b l h z p -> b l z p', reduction=self.head_fusion)(mask)
            mask = Rearrange('b l z (h w)  -> b l z h w', h=self.width, w=self.width)(mask)
        else:
            mask = Rearrange('b l hd z (h w)  -> b l hd z h w', h=self.width, w=self.width)(mask) # *Niên: chỗ này tách từng token (1, 196) thành từng patch (1, 14, 14)

        return prediction, mask, output

    def generate_scores(self, head_cams, prediction, output_truth, image):
        with torch.no_grad():
            tensor_heatmaps = head_cams[0]
            if self.is_head_fuse:
                tensor_heatmaps = tensor_heatmaps.reshape(12, 1, 14, 14)
            else:
                tensor_heatmaps = tensor_heatmaps.reshape(144, 1, 14, 14)
            tensor_heatmaps = transforms.Resize((224, 224))(tensor_heatmaps)
    
            if self.normalize_cam_heads:
                # Compute min and max along each image
                min_vals = tensor_heatmaps.amin(dim=(2, 3), keepdim=True)  # Min across width and height
                max_vals = tensor_heatmaps.amax(dim=(2, 3), keepdim=True)  # Max across width and height
                # Normalize using min-max scaling
                tensor_heatmaps = (tensor_heatmaps - min_vals) / (max_vals - min_vals + 1e-7)  # Add small value to avoid division by zero
            
            if self.add_noise:
                # -------------- add noise ------------
                N  = tensor_heatmaps.shape[0]
                H = tensor_heatmaps.shape[2]
                W = tensor_heatmaps.shape[3]
                # Generate the inverse of masks, i.e., 1-M_i
                masks_inverse=torch.from_numpy(np.repeat((1-tensor_heatmaps.cpu().numpy())[:, :, np.newaxis,:], 3, axis=2)).cuda()
                masks_inverse = masks_inverse.squeeze(1)

                random_whole=torch.randn([N]+list((3,H,W))).cuda()* 0.1
                noise_to_add = random_whole * masks_inverse

                mask = torch.mul(tensor_heatmaps, image)
                m = mask + noise_to_add
            else:
                m = torch.mul(tensor_heatmaps, image)

            with torch.no_grad():
                output_mask = self.model(m)

            if self.vitcx_score_formula:
                p_mask_with_noise = output_mask[:, prediction.item()]
                p_x_with_noise = self.model(image + noise_to_add)[0, prediction.item()]
                class_p = output_truth[0, prediction.item()]
                agc_scores = p_mask_with_noise - p_x_with_noise + class_p
            else:
            # print("After get output from model: ")
            # print(torch.cuda.memory_allocated()/1024**2)
                agc_scores = output_mask[:, prediction.item()] - output_truth[0, prediction.item()]
            
            if self.score_minmax_norm:   
                agc_scores = (agc_scores - agc_scores.min() ) / (agc_scores.max() - agc_scores.min())
            else:
                agc_scores = torch.sigmoid(agc_scores)
            
            agc_scores += self.plus

            agc_scores = agc_scores.reshape(head_cams[0].shape[0], head_cams[0].shape[1])

            del output_mask  # Delete unnecessary variables that are no longer needed
            torch.cuda.empty_cache()  # Clean up cache if necessary
            # print("After deleted output from model: ")
            # print(torch.cuda.memory_allocated()/1024**2)
            
            return agc_scores

    def generate_saliency(self, head_cams, agc_scores):
        if self.is_head_fuse:
            mask = (agc_scores.view(1, 12, 1, 1, 1) * head_cams[0]).sum(axis=(0, 1))
        else:
            mask = (agc_scores.view(12, 12, 1, 1, 1) * head_cams[0]).sum(axis=(0, 1))

        mask = mask.squeeze()
        return mask

    def generate(self, x, class_idx=None):
        return self(x, class_idx)
        
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

        return predicted_class, saliency_map


class ScoreAGC_head_fusion:
    def __init__(self, model, attention_matrix_layer = 'before_softmax', attention_grad_layer = 'after_softmax', head_fusion='sum', layer_fusion='sum', normalize_cam_heads=True, score_minmax_norm=False):
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
        self.normalize_cam_heads = normalize_cam_heads
        self.score_minmax_norm = score_minmax_norm

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
        mask = Reduce('b l h z p -> b l z p', reduction=self.head_fusion)(mask)
        mask = Rearrange('b l z (h w)  -> b l z h w', h=self.width, w=self.width)(mask) # *Niên: chỗ này tách từng token (1, 196) thành từng patch (1, 14, 14)

        return prediction, mask, output

    def generate_scores(self, head_cams, prediction, output_truth, image):
        with torch.no_grad():
            tensor_heatmaps = head_cams[0]
            tensor_heatmaps = tensor_heatmaps.reshape(12, 1, 14, 14)
            tensor_heatmaps = transforms.Resize((224, 224))(tensor_heatmaps)
    
            if self.normalize_cam_heads:
                # Compute min and max along each image
                min_vals = tensor_heatmaps.amin(dim=(2, 3), keepdim=True)  # Min across width and height
                max_vals = tensor_heatmaps.amax(dim=(2, 3), keepdim=True)  # Max across width and height
                # Normalize using min-max scaling
                tensor_heatmaps = (tensor_heatmaps - min_vals) / (max_vals - min_vals + 1e-7)  # Add small value to avoid division by zero
            
            # -------------- add noise ------------
            N  = tensor_heatmaps.shape[0]
            H = tensor_heatmaps.shape[2]
            W = tensor_heatmaps.shape[3]
            # Generate the inverse of masks, i.e., 1-M_i
            masks_inverse=torch.from_numpy(np.repeat((1-tensor_heatmaps.cpu().numpy())[:, :, np.newaxis,:], 3, axis=2)).cuda()
            masks_inverse = masks_inverse.squeeze(1)

            random_whole=torch.randn([N]+list((3,H,W))).cuda()* 0.1
            noise_to_add = random_whole * masks_inverse

            mask = torch.mul(tensor_heatmaps, image)
            m = mask + noise_to_add

            with torch.no_grad():
                output_mask = self.model(m)
            
            agc_scores = output_mask[:, prediction.item()] - output_truth[0, prediction.item()]
            
            if self.score_minmax_norm:
                
                agc_scores = (agc_scores - agc_scores.min() ) / (agc_scores.max() - agc_scores.min())
            else:
                agc_scores = torch.sigmoid(agc_scores)
            
            agc_scores += 1

            agc_scores = agc_scores.reshape(head_cams[0].shape[0], head_cams[0].shape[1])

            del output_mask  # Delete unnecessary variables that are no longer needed
            torch.cuda.empty_cache()  # Clean up cache if necessary
            # print("After deleted output from model: ")
            # print(torch.cuda.memory_allocated()/1024**2)
            
            return agc_scores

    def generate_saliency(self, head_cams, agc_scores):
        mask = (agc_scores.view(1, 12, 1, 1, 1) * head_cams[0]).sum(axis=(0, 1))

        mask = mask.squeeze()
        return mask

    def generate(self, x, class_idx=None):

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

        return predicted_class, saliency_map


class BetterAGC_ver2:
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

        self.gradient = gradient
        self.attn = attn
        # print('[DEBUG] gradient shape: ', self.gradient.shape)
        # print('[DEBUG] attn shape: ', self.attn.shape)

        # aggregation of CAM of all heads and all layers and reshape the final CAM.
        mask = mask[:, :, :, 1:].unsqueeze(0) # * niên: chỗ này thêm 1 ở đầu (ví dụ: (2) -> (1, 2)) và 1: là bỏ token class
        self.gradient = self.gradient[:, :, :, 1:].unsqueeze(0) # * niên: chỗ này thêm 1 ở đầu (ví dụ: (2) -> (1, 2)) và 1: là bỏ token class
        self.attn = self.attn[:, :, :, 1:].unsqueeze(0) # * niên: chỗ này thêm 1 ở đầu (ví dụ: (2) -> (1, 2)) và 1: là bỏ token class
        # print(mask.shape)

        # *Niên:Thay vì tính tổng theo blocks và theo head như công thức để ra 1 mask cuối cùng là CAM thì niên sẽ giữ lại tất cả các mask của các head ở mỗi block
        mask = Rearrange('b l hd z (h w)  -> b l hd z h w', h=self.width, w=self.width)(mask) # *Niên: chỗ này tách từng token (1, 196) thành từng patch (1, 14, 14)
        self.gradient = Rearrange('b l hd z (h w)  -> b l hd z h w', h=self.width, w=self.width)(self.gradient) # *Niên: chỗ này tách từng token (1, 196) thành từng patch (1, 14, 14)
        self.attn = Rearrange('b l hd z (h w)  -> b l hd z h w', h=self.width, w=self.width)(self.attn) # *Niên: chỗ này tách từng token (1, 196) thành từng patch (1, 14, 14)

        # return prediction, mask, output
        return prediction, self.attn, output

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
            # agc_scores = F.softmax(agc_scores)
    
            agc_scores = agc_scores.reshape(head_cams[0].shape[0], head_cams[0].shape[1])

            del output_mask  # Delete unnecessary variables that are no longer needed
            torch.cuda.empty_cache()  # Clean up cache if necessary
            # print("After deleted output from model: ")
            # print(torch.cuda.memory_allocated()/1024**2)
            
            return agc_scores

    def generate_saliency(self, head_cams, agc_scores):
        # mask = (agc_scores.view(12, 12, 1, 1, 1) * head_cams[0]).sum(axis=(0, 1))
        mask = ((agc_scores.view(12, 12, 1, 1, 1) + self.gradient[0]) * head_cams[0]).sum(axis=(0, 1))

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
        # print('[DEBUG] head_cams shape: ', head_cams.shape)
        # Generate the saliency map for image x and class_idx
        scores = self.generate_scores(
            image=x,
            head_cams=head_cams,
            prediction=predicted_class, output_truth=output_truth
        )

        # print('[DEBUG] score shape: ', scores.shape)

        # print("After generate scores: ")
        # print(torch.cuda.memory_allocated()/1024**2)
        # print()
        
        saliency_map = self.generate_saliency(head_cams=head_cams, agc_scores=scores)
        # print("After generate saliency maps: ")
        # print(torch.cuda.memory_allocated()/1024**2)
        # print()

        # return saliency_map.detach().cpu(), scores.detach().cpu(), head_cams.detach().cpu()
        return predicted_class, saliency_map
    
class BetterAGC_cluster:
    def __init__(self, model, attention_matrix_layer = 'before_softmax', attention_grad_layer = 'after_softmax', head_fusion='sum', layer_fusion='sum', thresold=0.7, num_heatmaps=30):
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
        self.thresold = thresold
        self.num_heatmaps = num_heatmaps

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
            tensor_heatmaps = head_cams
            tensor_heatmaps = transforms.Resize((224, 224))(tensor_heatmaps)
            # print('[DEBUG3]', tensor_heatmaps.shape)
    
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
            agc_scores += 1
            # print('[DEBUG4]', agc_scores.shape)
    
            # agc_scores = agc_scores.reshape(head_cams.shape[0], head_cams.shape[1])

            del output_mask  # Delete unnecessary variables that are no longer needed
            torch.cuda.empty_cache()  # Clean up cache if necessary
            # print("After deleted output from model: ")
            # print(torch.cuda.memory_allocated()/1024**2)
            
            return agc_scores

    def generate_saliency(self, head_cams, agc_scores):
        # print('------------------- [DEBUG GENERATE SALIENCY] --------------')
        # print('Head cams shape: ', head_cams.shape)
        # print('scores shape', agc_scores.shape)

        mask = (agc_scores.view(self.num_heatmaps, 1, 1, 1) * head_cams).sum(axis=(0, 1))
        # print('mask shape: ', mask.shape)
        # mask = mask.squeeze()
        return mask

    def k_means(self, encoder_activations):
        # Squeeze to shape (n_tokens+1, n_activations)
        # encoder_activations = encoder_activations.squeeze(0)

        # remove CLS token and transpose, shape (n_activations, n_tokens)
        # encoder_activations = encoder_activations[1:].T
        # Reshape to (144, 196)

        encoder_activations = encoder_activations.view(-1, 14 * 14) # -------

        # print('[DEBUG] ', encoder_activations.shape)
        # encoder_activations = encoder_activations.T

        # Create clusters with kmeans
        kmeans = KMeans(n_clusters=self.num_heatmaps, mode='euclidean', verbose=False)
        kmeans.fit(encoder_activations)

        # Use kmeans centroids as basis for masks
        raw_masks = kmeans.centroids

        raw_masks = raw_masks.reshape(self.num_heatmaps, 14, 14)
        raw_masks = raw_masks.unsqueeze(1)

        return raw_masks

    def clustering(self, head_cams):
        def get_cos_similar_matrix(v1, v2):
            num = torch.mm(v1,torch.transpose(v2, 0, 1)) 
            denom = torch.linalg.norm(v1,  dim=1).reshape(-1, 1) * torch.linalg.norm(v2,  dim=1)
            res = num / denom
            res[torch.isnan(res)] = 0
            return res

        def norm_matrix(act):
            row_mins=torch.min(act,1).values[:, None]
            row_maxs=torch.max(act,1).values[:, None] 
            act=(act-row_mins)/(row_maxs-row_mins + 0.0000000000001)
        
            return act
        
        # Compute the pairwise cosine similarity and distance of the ViT masks
         # Step 1: Reshape to (144, 196)
        head_cams = norm_matrix(head_cams.reshape(144, -1)).detach()
        
        similarity = get_cos_similar_matrix(head_cams, head_cams)
        distance = 1 - similarity

        # Apply the  AgglomerativeClustering with a given distance_threshold
        cluster = AgglomerativeClustering(n_clusters = None, distance_threshold=self.thresold,metric='precomputed', linkage='complete') 
        cluster.fit(distance.cpu())
        cluster_num=len(set(cluster.labels_))
        # print('number of masks after the clustering:'+str(cluster_num))

        # Use the sum of a clustering as a representation of the cluster
        cluster_labels=cluster.labels_
        cluster_labels_set=set(cluster_labels)
        # print('[DEBUG] cluster labels: ', cluster_labels_set)
        mask_clustering=torch.zeros((len(cluster_labels_set), 14*14)).cuda()
        num_mask_clustering =torch.zeros((len(cluster_labels_set), 1)).cuda()
        for i in range(len(head_cams)):
            mask_clustering[cluster_labels[i]]+=head_cams[i]
            num_mask_clustering[cluster_labels[i]] += 1


        # print('[DEBUG]', num_mask_clustering)   
        # print('[BEFORE]', mask_clustering)   

        # old = mask_clustering

        # for i in cluster_labels_set:
        #     mask_clustering[i] /= num_mask_clustering[i]

        # print('[AFTER]', mask_clustering)

        # normalize the masks
        mask_clustering_norm=norm_matrix(mask_clustering).reshape((len(cluster_labels_set), 14, 14))
        old =norm_matrix(old).reshape((len(cluster_labels_set), 14, 14))
        # print('[SAME] ', mask_clustering_norm == old)

        mask_clustering_norm = mask_clustering_norm.unsqueeze(1)
        # print('[FINISH CLUSTERING], new masks shape: ', mask_clustering_norm.shape)
        return mask_clustering_norm

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

        
        # head_cams = self.clustering(head_cams)
        # print('[[DEBUG]', len(head_cams))

        # print('---------- [BEFORE CLUSTERING] ---------------')
        # print('Number of heatmaps: ', len(head_cams))
        # print('Shape of heatmaps: ', head_cams.shape)
        # print()

        head_cams = self.k_means(head_cams)
        # print('---------- [AFTER CLUSTERING] ---------------')
        # print('Number of heatmaps: ', len(head_cams))
        # print('Shape of heatmaps: ', head_cams.shape)
        # print()

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

        return predicted_class, saliency_map
    
class BetterAGC_cluster_add_noise:
    def __init__(self, model, attention_matrix_layer = 'before_softmax', attention_grad_layer = 'after_softmax', head_fusion='sum', layer_fusion='sum', thresold=0.7, num_heatmaps=30):
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
        self.thresold = thresold
        self.num_heatmaps = num_heatmaps

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
            tensor_heatmaps = head_cams
            tensor_heatmaps = transforms.Resize((224, 224))(tensor_heatmaps)
            # print('[DEBUG3]', tensor_heatmaps.shape)
    
            # Compute min and max along each image
            min_vals = tensor_heatmaps.amin(dim=(2, 3), keepdim=True)  # Min across width and height
            max_vals = tensor_heatmaps.amax(dim=(2, 3), keepdim=True)  # Max across width and height
            # Normalize using min-max scaling
            tensor_heatmaps = (tensor_heatmaps - min_vals) / (max_vals - min_vals + 1e-7)  # Add small value to avoid division by zero
            # print("before multiply img with mask: ")
            # print(torch.cuda.memory_allocated()/1024**2)

            # print('[DEBUG1]', tensor_heatmaps.shape)
            # -------------- add noise ------------
            N  = tensor_heatmaps.shape[0]
            H = tensor_heatmaps.shape[2]
            W = tensor_heatmaps.shape[3]
            # Generate the inverse of masks, i.e., 1-M_i
            masks_inverse=torch.from_numpy(np.repeat((1-tensor_heatmaps.cpu().numpy())[:, :, np.newaxis,:], 3, axis=2)).cuda()
            masks_inverse = masks_inverse.squeeze(1)

            # print('[DEBUG INVERSE MASK]', masks_inverse.shape)
            # Generate the random Gaussian noise
            random_whole=torch.randn([N]+list((3,H,W))).cuda()* 0.1
            # print('[DEBUG RANDOM]', random_whole.shape)
            noise_to_add = random_whole * masks_inverse

            mask = torch.mul(tensor_heatmaps, image)
            # print('[DEBUG2]', mask.shape)
            # print('[DEBUG3]', noise_to_add.shape)
            m = mask + noise_to_add
            # print("After multiply img with mask scores: ")
            # print(torch.cuda.memory_allocated()/1024**2)

            with torch.no_grad():
                output_mask = self.model(m)
            

            # print("After get output from model: ")
            # print(torch.cuda.memory_allocated()/1024**2)
    
            # agc_scores = output_mask[:, prediction.item()] - output_truth[0, prediction.item()]
            p_mask_with_noise = output_mask[:, prediction.item()]
            p_x_with_noise = self.model(image + noise_to_add)[0, prediction.item()]
            class_p = output_truth[0, prediction.item()]
            agc_scores = p_mask_with_noise - p_x_with_noise + class_p
            agc_scores = torch.sigmoid(agc_scores)
            agc_scores += 1
            # print('[DEBUG4]', agc_scores.shape)
    
            # agc_scores = agc_scores.reshape(head_cams.shape[0], head_cams.shape[1])

            del output_mask  # Delete unnecessary variables that are no longer needed
            torch.cuda.empty_cache()  # Clean up cache if necessary
            # print("After deleted output from model: ")
            # print(torch.cuda.memory_allocated()/1024**2)
            
            return agc_scores

    def generate_saliency(self, head_cams, agc_scores):
        # print('------------------- [DEBUG GENERATE SALIENCY] --------------')
        # print('Head cams shape: ', head_cams.shape)
        # print('scores shape', agc_scores.shape)

        mask = (agc_scores.view(self.num_heatmaps, 1, 1, 1) * head_cams).sum(axis=(0, 1))
        # print('mask shape: ', mask.shape)
        # mask = mask.squeeze()
        return mask

    def k_means(self, encoder_activations):
        # Squeeze to shape (n_tokens+1, n_activations)
        # encoder_activations = encoder_activations.squeeze(0)

        # remove CLS token and transpose, shape (n_activations, n_tokens)
        # encoder_activations = encoder_activations[1:].T
        # Reshape to (144, 196)

        encoder_activations = encoder_activations.view(-1, 14 * 14) # -------

        # print('[DEBUG] ', encoder_activations.shape)
        # encoder_activations = encoder_activations.T

        # Create clusters with kmeans
        kmeans = KMeans(n_clusters=self.num_heatmaps, mode='euclidean', verbose=False)
        kmeans.fit(encoder_activations)

        # Use kmeans centroids as basis for masks
        raw_masks = kmeans.centroids

        raw_masks = raw_masks.reshape(self.num_heatmaps, 14, 14)
        raw_masks = raw_masks.unsqueeze(1)

        return raw_masks

    def clustering(self, head_cams):
        def get_cos_similar_matrix(v1, v2):
            num = torch.mm(v1,torch.transpose(v2, 0, 1)) 
            denom = torch.linalg.norm(v1,  dim=1).reshape(-1, 1) * torch.linalg.norm(v2,  dim=1)
            res = num / denom
            res[torch.isnan(res)] = 0
            return res

        def norm_matrix(act):
            row_mins=torch.min(act,1).values[:, None]
            row_maxs=torch.max(act,1).values[:, None] 
            act=(act-row_mins)/(row_maxs-row_mins + 0.0000000000001)
        
            return act
        
        # Compute the pairwise cosine similarity and distance of the ViT masks
         # Step 1: Reshape to (144, 196)
        head_cams = norm_matrix(head_cams.reshape(144, -1)).detach()
        
        similarity = get_cos_similar_matrix(head_cams, head_cams)
        distance = 1 - similarity

        # Apply the  AgglomerativeClustering with a given distance_threshold
        cluster = AgglomerativeClustering(n_clusters = None, distance_threshold=self.thresold,metric='precomputed', linkage='complete') 
        cluster.fit(distance.cpu())
        cluster_num=len(set(cluster.labels_))
        # print('number of masks after the clustering:'+str(cluster_num))

        # Use the sum of a clustering as a representation of the cluster
        cluster_labels=cluster.labels_
        cluster_labels_set=set(cluster_labels)
        # print('[DEBUG] cluster labels: ', cluster_labels_set)
        mask_clustering=torch.zeros((len(cluster_labels_set), 14*14)).cuda()
        num_mask_clustering =torch.zeros((len(cluster_labels_set), 1)).cuda()
        for i in range(len(head_cams)):
            mask_clustering[cluster_labels[i]]+=head_cams[i]
            num_mask_clustering[cluster_labels[i]] += 1


        # print('[DEBUG]', num_mask_clustering)   
        # print('[BEFORE]', mask_clustering)   

        # old = mask_clustering

        # for i in cluster_labels_set:
        #     mask_clustering[i] /= num_mask_clustering[i]

        # print('[AFTER]', mask_clustering)

        # normalize the masks
        mask_clustering_norm=norm_matrix(mask_clustering).reshape((len(cluster_labels_set), 14, 14))
        old =norm_matrix(old).reshape((len(cluster_labels_set), 14, 14))
        # print('[SAME] ', mask_clustering_norm == old)

        mask_clustering_norm = mask_clustering_norm.unsqueeze(1)
        # print('[FINISH CLUSTERING], new masks shape: ', mask_clustering_norm.shape)
        return mask_clustering_norm

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

        
        # head_cams = self.clustering(head_cams)
        # print('[[DEBUG]', len(head_cams))

        # print('---------- [BEFORE CLUSTERING] ---------------')
        # print('Number of heatmaps: ', len(head_cams))
        # print('Shape of heatmaps: ', head_cams.shape)
        # print()

        head_cams = self.k_means(head_cams)
        # print('---------- [AFTER CLUSTERING] ---------------')
        # print('Number of heatmaps: ', len(head_cams))
        # print('Shape of heatmaps: ', head_cams.shape)
        # print()

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

        return predicted_class, saliency_map

class ScoreAGC_no_grad:
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

        self.gradient = gradient
        self.attn = attn
        # print('[DEBUG] gradient shape: ', self.gradient.shape)
        # print('[DEBUG] attn shape: ', self.attn.shape)

        # aggregation of CAM of all heads and all layers and reshape the final CAM.
        mask = mask[:, :, :, 1:].unsqueeze(0) # * niên: chỗ này thêm 1 ở đầu (ví dụ: (2) -> (1, 2)) và 1: là bỏ token class
        self.gradient = self.gradient[:, :, :, 1:].unsqueeze(0) # * niên: chỗ này thêm 1 ở đầu (ví dụ: (2) -> (1, 2)) và 1: là bỏ token class
        self.attn = self.attn[:, :, :, 1:].unsqueeze(0) # * niên: chỗ này thêm 1 ở đầu (ví dụ: (2) -> (1, 2)) và 1: là bỏ token class
        # print(mask.shape)

        # *Niên:Thay vì tính tổng theo blocks và theo head như công thức để ra 1 mask cuối cùng là CAM thì niên sẽ giữ lại tất cả các mask của các head ở mỗi block
        mask = Rearrange('b l hd z (h w)  -> b l hd z h w', h=self.width, w=self.width)(mask) # *Niên: chỗ này tách từng token (1, 196) thành từng patch (1, 14, 14)
        self.gradient = Rearrange('b l hd z (h w)  -> b l hd z h w', h=self.width, w=self.width)(self.gradient) # *Niên: chỗ này tách từng token (1, 196) thành từng patch (1, 14, 14)
        self.attn = Rearrange('b l hd z (h w)  -> b l hd z h w', h=self.width, w=self.width)(self.attn) # *Niên: chỗ này tách từng token (1, 196) thành từng patch (1, 14, 14)

        # return prediction, mask, output
        return prediction, self.attn, output

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
            # agc_scores = F.softmax(agc_scores)
    
            agc_scores = agc_scores.reshape(head_cams[0].shape[0], head_cams[0].shape[1])

            del output_mask  # Delete unnecessary variables that are no longer needed
            torch.cuda.empty_cache()  # Clean up cache if necessary
            # print("After deleted output from model: ")
            # print(torch.cuda.memory_allocated()/1024**2)
            
            return agc_scores

    def generate_saliency(self, head_cams, agc_scores):
        # mask = (agc_scores.view(12, 12, 1, 1, 1) * head_cams[0]).sum(axis=(0, 1))
        mask = ((agc_scores.view(12, 12, 1, 1, 1)) * head_cams[0]).sum(axis=(0, 1))

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
        # print('[DEBUG] head_cams shape: ', head_cams.shape)
        # Generate the saliency map for image x and class_idx
        scores = self.generate_scores(
            image=x,
            head_cams=head_cams,
            prediction=predicted_class, output_truth=output_truth
        )

        # print('[DEBUG] score shape: ', scores.shape)

        # print("After generate scores: ")
        # print(torch.cuda.memory_allocated()/1024**2)
        # print()
        
        saliency_map = self.generate_saliency(head_cams=head_cams, agc_scores=scores)
        # print("After generate saliency maps: ")
        # print(torch.cuda.memory_allocated()/1024**2)
        # print()

        # return saliency_map.detach().cpu(), scores.detach().cpu(), head_cams.detach().cpu()
        return predicted_class, saliency_map
 