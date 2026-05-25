import numpy as np
import torch
from sklearn.cluster import KMeans
from pruned_layers import *
import torch.nn as nn

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def quantize_whole_model(net, bits=8):
    """
    Quantize the whole model.
    :param net: (object) network model.
    :return: centroids of each weight layer, used in the quantization codebook.
    """
    cluster_centers = []
    assert isinstance(net, nn.Module)
    layer_ind = 0
    for n, m in net.named_modules():
        if isinstance(m, PrunedConv):
            # Extract the weight parameters and the pruning mask
            weight = m.conv.weight.data.cpu().numpy()
            mask = m.mask
            
            # Cluster the non-zero weight parameters using KMeans
            active_weights = weight[mask == 1].reshape(-1, 1)  # Reshape for KMeans
            if len(active_weights) > 0:
                n_clusters = min(len(active_weights), 2**bits)
                kmeans= KMeans(n_clusters=n_clusters, n_init=10, random_state=0).fit(active_weights)
                
                # Replace the non-zero weight parameters with their corresponding cluster centers
                centroids = kmeans.cluster_centers_
                labels = kmeans.labels_
                weight[mask == 1] = centroids[labels].flatten()
                
                # Update the model weights and store centroids for the codebook
                m.conv.weight.data = torch.from_numpy(weight).to(device)
                cluster_centers.append(centroids)           
            
            layer_ind += 1
            print("Complete %d layers quantization..." %layer_ind)
        elif isinstance(m, PruneLinear):
            # Extract the weight parameters and the pruning mask
            weight = m.linear.weight.data.cpu().numpy()
            mask = m.mask
            
            #Cluster the non-zero weight parameters using KMeans
            active_weights = weight[mask == 1].reshape(-1, 1)
            
            if len(active_weights) > 0:
                n_clusters = min(len(active_weights), 2**bits)
                kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=0).fit(active_weights)
                
                # Replace the non-zero weight parameters with their corresponding cluster centers
                centroids = kmeans.cluster_centers_
                labels = kmeans.labels_
                weight[mask == 1] = centroids[labels].flatten()
                
                # Update the model weights and store centroids for the codebook
                m.linear.weight.data = torch.from_numpy(weight).to(device)
                cluster_centers.append(centroids)
                
            layer_ind += 1
            print("Complete %d layers quantization..." %layer_ind)
    torch.save(net.state_dict(), "net_after_quantization.pt")
    return np.array(cluster_centers)
