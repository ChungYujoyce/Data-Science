import torch
import numpy as np
from sklearn.cluster import KMeans
from scipy.sparse import csc_matrix, csr_matrix


def apply_weight_sharing(model, bits=5):
    """
    Applies weight sharing to the given model
    """
    for name, module in model.named_children():
        dev = module.weight.device
        weight = module.weight.data.cpu().numpy()
        shape = weight.shape
        quan_range = 2 ** bits
        if len(shape) == 2:  # fully connected layers
            print(f'{name:20} | {str(module.weight.size()):35} | => quantize to {quan_range} indices')
            mat = csr_matrix(weight) if shape[0] < shape[1] else csc_matrix(weight)

            # weight sharing by kmeans
            space = np.linspace(min(mat.data), max(mat.data), num=quan_range)
            kmeans = KMeans(n_clusters=len(space), init=space.reshape(-1, 1), n_init=1, precompute_distances=True, algorithm="full")
            kmeans.fit(mat.data.reshape(-1, 1))
            new_weight = kmeans.cluster_centers_[kmeans.labels_].reshape(-1)
            mat.data = new_weight

            # Insert to model
            module.weight.data = torch.from_numpy(mat.toarray()).to(dev)
        elif len(shape) == 4: 
            nzw = weight[np.where(weight!=0)]
            space = np.linspace(np.amin(nzw), np.amax(nzw), num=quan_range)
            kmeans = KMeans(n_clusters=len(space), init=space.reshape(-1, 1), n_init=1, precompute_distances=True, algorithm="full")
            kmeans.fit(nzw.reshape(-1, 1))
            
            all_mat = np.zeros(shape)
            for i in range(shape[0]):
                for j in range(shape[1]):
                    mat = csr_matrix(weight[i,j,:,:]) if shape[2] < shape[3] else csc_matrix(weight[i,j,:,:])
                    mat_reshape = mat.data.reshape(-1, 1)
                    if mat_reshape.shape[0] == 0:
                        continue
                    center_index = kmeans.predict(mat_reshape)
                    mat.data = kmeans.cluster_centers_[center_index].reshape(-1)
                    module.weight.data[i,j,:,:] = torch.from_numpy(mat.toarray()).to(dev)

            print(f'{name:20} | {str(module.weight.size()):35} | => quantize to {quan_range} indices')
            pass
