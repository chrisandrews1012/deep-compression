# Deep Compression

An implementation of the deep compression pipeline applied to a VGG-16 neural network trained on CIFAR-10.

## Overview

This was a lab assignment for my course in Software-Hardware Co-design for Intelligent Systems. It explored how to make neural networks small enough to deploy on resource-constrained hardware. Deep compression is a technique that dramatically shrinks a trained model's storage footprint without meaningfully hurting its accuracy. This project implements all three stages of the pipeline from [Han et al. (2015)](https://arxiv.org/abs/1510.00149): removing weak connections, clustering the remaining weights, and compressing the result with entropy coding. Applied to a VGG-16 model on CIFAR-10, the pipeline achieves over **40x compression** while maintaining **90%+ test accuracy**.

## Implementation

- **Pruning**: Implemented two strategies for removing insignificant weights from the network, one based on a percentage threshold and one based on the distribution of each layer's weights. Pruned weights are masked out and stay zeroed during fine-tuning.
- **Fine-tuning**: Extended the training loop to enforce the pruning masks during backpropagation, ensuring pruned connections do not recover across training steps.
- **Quantization**: Reduced the number of unique weight values in each layer by grouping them into clusters, so weights can be stored as small indices into a codebook rather than full 32-bit floats.
- **Huffman Coding**: Applied entropy coding to the quantized weight indices, taking advantage of the uneven distribution of values to further reduce the average bits needed per weight.
- **Experiments**: Analyzed the trade-off between sparsity and accuracy across a range of pruning thresholds, compared two pruning methods at equivalent sparsity levels, and swept quantization bit-widths to find where accuracy starts to degrade.

## Results

| Stage | Sparsity | Accuracy |
|---|---|---|
| Baseline | 0% | 92.30% |
| After Pruning | 83.07% | 90.94% |
| After Quantization | 83.07% | 90.84% |
| After Huffman Coding | 83.07% | 90.84% |

**Compression Ratio: ~40x**

## Notebooks

- **DeepCompression.ipynb**: The main experimental notebook. Covers the full compression pipeline with a moderate pruning threshold, and includes a sensitivity analysis sweeping multiple thresholds, a side-by-side comparison of the two pruning methods, and a quantization bit-width sweep to explore the accuracy trade-off.
- **DeepCompression_AggressivePruning.ipynb**: A focused run of the full pipeline using a more aggressive pruning threshold, pushing model sparsity to ~83%. Skips the sensitivity experiments and instead zeroes in on computing the final end-to-end compression ratio.

## Project Structure

```
deep-compression/
├── pruned_layers.py        # Custom layer wrappers with pruning and masking
├── prune.py                # Applies pruning across all layers of a network
├── train_util.py           # Training, fine-tuning, and evaluation
├── quantize.py             # K-means weight quantization per layer
├── huffman_coding.py       # Huffman encoding and average bit-width computation
├── summary.py              # Per-layer sparsity and parameter count reporting
├── vgg16.py                # VGG16 and VGG16_half model definitions
├── DeepCompression.ipynb
└── DeepCompression_AggressivePruning.ipynb
```

## Getting Started

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended)

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/deep-compression.git
   cd deep-compression
   ```
2. Open either notebook and run the cells in order.

## License

This project is licensed under the [MIT License](LICENSE).
