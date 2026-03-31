import numpy as np
from sklearn.cluster import KMeans
from pruned_layers import *
import torch.nn as nn
import heapq
from collections import defaultdict

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def _huffman_coding_per_layer(weight, centers):
    """
    Huffman coding at each layer
    :param weight: weight parameter of the current layer.
    :param centers: KMeans centroids in the quantization codebook of the current weight layer.
    :return: 
            'encodings': Encoding map mapping each weight parameter to its Huffman coding.
            'frequency': Frequency map mapping each weight parameter to the total number of its appearance.
    """
    # If there is no weight parameter in the current layer, return empty encoding and frequency map.
    if len(centers) == 0:
        return {}, {}

    # Flatten the wieghts and filter out the non-zero weights
    weight = weight.flatten()
    weight = weight[weight != 0]
    
    # Generate frequency map for the non-zero weights
    frequency = defaultdict(int)
    for w in weight:
        frequency[w] += 1
        
    # Build a min-heap based on the frequency map
    heap = [[freq, [sym, ""]] for sym, freq in frequency.items()]
    heapq.heapify(heap)
    
    # Handle the case when there is only one unique weight
    if len(heap) == 1:
        encodings = {heap[0][1][0]: "0"}
        return encodings, frequency
    
    # Iteratively combine the two least frequent nodes until only one node (the root of the Huffman tree) remains
    while len(heap) > 1:
        lo = heapq.heappop(heap)   # Node with the smallest frequency
        hi = heapq.heappop(heap)   # Node with the second smallest frequency
        # Assign '0' to the left node and '1' to the right node, and combine them into a new node
        for pair in lo[1:]:
            pair[1] = '0' + pair[1]
        for pair in hi[1:]:
            pair[1] = '1' + pair[1]
        
        # Combine the two nodes and push the new node back into the heap
        heapq.heappush(heap, [lo[0] + hi[0]] + lo[1:] + hi[1:])
    
    # Extract the encodings from the final node in the heap
    encodings = {}
    if heap:
        for pair in heap[0][1:]:
            encodings[pair[0]] = pair[1]
            
    return encodings, frequency


def compute_average_bits(encodings, frequency):
    """
    Compute the average storage bits of the current layer after Huffman Coding.
    :param 'encodings': Encoding map mapping each weight parameter to its Huffman coding.
    :param 'frequency': Frequency map mapping each weight parameter to the total number of its appearance.
    :return (float) a floating value represents the average bits.
    """
    total = 0
    total_bits = 0
    for key in frequency.keys():
        total += frequency[key]
        total_bits += frequency[key] * len(encodings[key])
    return total_bits / total

def huffman_coding(net, centers):
    """
    Apply huffman coding on a 'quantized' model to save further computation cost.
    :param net: a 'nn.Module' network object.
    :param centers: KMeans centroids in the quantization codebook for Huffman coding.
    :return: frequency map and encoding map of the whole 'net' object.
    """
    assert isinstance(net, nn.Module)
    layer_ind = 0
    freq_map = []
    encodings_map = []
    for n, m in net.named_modules():
        if isinstance(m, PrunedConv):
            weight = m.conv.weight.data.cpu().numpy()
            center = centers[layer_ind]
            orginal_avg_bits = round(np.log2(len(center)))
            print("Original storage for each parameter: %.4f bits" %orginal_avg_bits)
            encodings, frequency = _huffman_coding_per_layer(weight, center)
            freq_map.append(frequency)
            encodings_map.append(encodings)
            huffman_avg_bits = compute_average_bits(encodings, frequency)
            print("Average storage for each parameter after Huffman Coding: %.4f bits" %huffman_avg_bits)
            layer_ind += 1
            print("Complete %d layers for Huffman Coding..." %layer_ind)
        elif isinstance(m, PruneLinear):
            weight = m.linear.weight.data.cpu().numpy()
            center = centers[layer_ind]
            orginal_avg_bits = round(np.log2(len(center)))
            print("Original storage for each parameter: %.4f bits" %orginal_avg_bits)
            encodings, frequency = _huffman_coding_per_layer(weight, center)
            freq_map.append(frequency)
            encodings_map.append(encodings)
            huffman_avg_bits = compute_average_bits(encodings, frequency)
            print("Average storage for each parameter after Huffman Coding: %.4f bits" %huffman_avg_bits)
            layer_ind += 1
            print("Complete %d layers for Huffman Coding..." %layer_ind)

    return freq_map, encodings_map
