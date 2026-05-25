# Deep Compression

Neural network compression project implementing the three-stage pipeline from [Han et al. (2015)](https://arxiv.org/abs/1510.00149) on VGG-16 trained on CIFAR-10. Achieves over **40x compression** while maintaining **90%+ test accuracy**.

## Problem Statement

Trained neural networks are too large to deploy on resource-constrained hardware like embedded systems and mobile devices. A full-precision VGG-16 model stores every weight as a 32-bit float, making it impractical to fit in limited memory or ship over a network. The challenge is reducing model size significantly without meaningfully hurting accuracy.

## Approach

Implements the three-stage pipeline from Han et al. (2015) applied to a VGG-16 trained on CIFAR-10.

**Pruning** removes low-magnitude weights from every layer using a threshold derived from each layer's weight distribution. Pruned connections are masked out and stay zeroed during backpropagation, so the network is then fine-tuned at the same sparsity level to recover accuracy.

**Quantization** reduces the number of unique weight values per layer using k-means clustering. Weights are replaced by cluster indices, so instead of storing a full 32-bit float per weight you store a small integer pointing into a shared codebook. The number of bits controls how many clusters are used.

**Huffman Coding** applies entropy coding to the quantized indices. Because the distribution of index values is uneven after quantization, Huffman coding reduces the average bits per weight below what fixed-width encoding would require.

The notebooks also include a sensitivity analysis sweeping pruning thresholds, a comparison of std-based vs percentage pruning at equivalent sparsity, and a quantization bit-width sweep to find where accuracy starts to degrade.

## Results

| Stage | Sparsity | Accuracy |
|---|---|---|
| Baseline | 0% | 92.30% |
| After Pruning | 83.07% | 90.94% |
| After Quantization | 83.07% | 90.84% |
| After Huffman Coding | 83.07% | 90.84% |

**Compression Ratio: ~40x**

The pipeline removes 83% of weights with less than 1.5% accuracy loss. Quantization and Huffman coding add further compression on top with no additional accuracy cost, bringing the total compression ratio to approximately 40x over the original full-precision model.

## Project Structure

```
deep-compression/
├── notebooks/
│   ├── 01_compression_pipeline.ipynb
│   └── 02_aggressive_pruning.ipynb
├── src/
│   └── deep_compression/
│       ├── vgg16.py
│       ├── prune.py
│       ├── pruned_layers.py
│       ├── train_util.py
│       ├── quantize.py
│       ├── huffman_coding.py
│       └── summary.py
├── models/
└── reports/
    └── figures/
```

**Notebooks**

`01_compression_pipeline.ipynb` runs the full pipeline at a moderate pruning threshold. Includes a sensitivity analysis sweeping multiple thresholds, a comparison of std-based vs percentage pruning at equivalent sparsity, and a quantization bit-width sweep showing where accuracy starts to degrade.

`02_aggressive_pruning.ipynb` runs the same pipeline at a more aggressive threshold, pushing sparsity to ~83%, and computes the final end-to-end compression ratio.

**Source**

`vgg16.py` defines the VGG-16 and VGG-16 half-width model architectures. 
`prune.py` and `pruned_layers.py` implement the pruning logic and custom layer wrappers that enforce binary masks during training. 
`train_util.py` handles training, fine-tuning and evaluation.
`quantize.py` runs k-means per layer. 
`huffman_coding.py` builds the encoding and computes average bits per weight. 
`summary.py` reports per-layer sparsity and parameter counts.

## Getting Started

Requirements: Python 3.8+, PyTorch, a CUDA-capable GPU is recommended.

```bash
git clone https://github.com/chrisandrews1012/deep-compression.git
cd deep-compression
pip install torch torchvision numpy matplotlib
```

Place pretrained model weights in `models/` then open either notebook and run cells in order. CIFAR-10 will download automatically on first run.

## License

This project is licensed under the [MIT License](LICENSE).
