import numpy as np
import torch
from torch.nn.modules.module import Module


class PruningModule(Module):
    def prune_by_percentile(self, q={'conv1': 16, 'conv2': 62, 'conv3': 65, 'conv4': 63, 'conv5': 63, 'fc1': 91, 'fc2': 91, 'fc3': 75}):
        
        # 	For each layer of weights W (including fc and conv layers) in the model, obtain the qth percentile of W as
        # 	the threshold, and then set the nodes with weight W less than threshold to 0, and the rest remain unchanged.
        
        # Calculate percentile value
        # Calculate percentile value
        conv1_alive_parameters = []
        conv2_alive_parameters = []
        conv3_alive_parameters = []
        conv4_alive_parameters = []
        conv5_alive_parameters = []
        fc1_alive_parameters = []
        fc2_alive_parameters = []
        fc3_alive_parameters = []

        for name, p in self.named_parameters():
            # We do not prune bias term
            tensor = p.data.cpu().numpy()
            alive = tensor[np.nonzero(tensor)] 
            if 'bias' in name or 'mask' in name:
                continue 
            if 'conv1' in name:
                conv1_alive_parameters.append(alive)
            if 'conv2' in name:
                conv2_alive_parameters.append(alive)
            if 'conv3' in name:
                conv3_alive_parameters.append(alive)
            if 'conv4' in name:
                conv4_alive_parameters.append(alive)
            if 'conv5' in name:
                conv5_alive_parameters.append(alive)
            if 'fc1' in name:
                fc1_alive_parameters.append(alive)
            if 'fc2' in name:
                fc2_alive_parameters.append(alive)
            if 'fc3' in name:
                fc3_alive_parameters.append(alive)

        conv1_percentile_value = np.percentile(abs(np.concatenate(conv1_alive_parameters)), q['conv1'])
        conv2_percentile_value = np.percentile(abs(np.concatenate(conv2_alive_parameters)), q['conv2'])
        conv3_percentile_value = np.percentile(abs(np.concatenate(conv3_alive_parameters)), q['conv3'])
        conv4_percentile_value = np.percentile(abs(np.concatenate(conv4_alive_parameters)), q['conv4'])
        conv5_percentile_value = np.percentile(abs(np.concatenate(conv5_alive_parameters)), q['conv5'])
        fc1_percentile_value = np.percentile(abs(np.concatenate(fc1_alive_parameters)), q['fc1'])
        fc2_percentile_value = np.percentile(abs(np.concatenate(fc2_alive_parameters)), q['fc2'])
        fc3_percentile_value = np.percentile(abs(np.concatenate(fc3_alive_parameters)), q['fc3'])

        percentile = {'conv1': conv1_percentile_value,
                      'conv2': conv2_percentile_value, 
                      'conv3': conv3_percentile_value, 
                      'conv4': conv4_percentile_value, 
                      'conv5': conv5_percentile_value, 
                      'fc1': fc1_percentile_value, 
                      'fc2': fc2_percentile_value, 
                      'fc3': fc3_percentile_value}

        # Prune the weights and mask
        for name, module in self.named_modules():
            if name in ['conv1','conv2','conv3','conv4','conv5','fc1', 'fc2', 'fc3']:
                print(name)
                self.prune(module=module,threshold=percentile[name])

    def prune_by_std(self, s=0.25):

        for name, module in self.named_modules():   
            #    Only fully connected layers were considered, but convolution layers also needed

            if name in ['conv1', 'conv2', 'conv3', 'conv4', 'conv5', 'fc1', 'fc2', 'fc3']:
                threshold = np.std(module.weight.data.cpu().numpy()) * s
                print(f'Pruning with threshold : {threshold} for layer {name}')
                self.prune(module, threshold)

    def prune(self, module, threshold):

        #    1. Use "module.weight.data" to get the weights of a certain layer of the model
        #    2. Set weights whose absolute value is less than threshold to 0, and keep the rest unchanged
        #    3. Save the results of the step 2 back to "module.weight.data"
        #    --------------------------------------------------------
        #    In addition, there is no need to return in this function ("module" can be considered as call by
        #    reference)
        
        #weight_dev = module.weight.device
        # Convert Tensors to numpy and calculate
        tensor = module.weight.data.cpu().numpy()
        new_weights = np.where(abs(tensor) < threshold, 0, 1)
        # Apply new weight and mask
        module.weight.data = torch.from_numpy(tensor * new_weights).cuda() 
        




