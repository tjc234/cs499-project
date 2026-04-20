# Project: PathMNIST Image Classification

# Setup and configuration
# - Import required libraries
# - Set random seeds for reproducibility
# - Select training device (CPU, GPU, or HPC cluster)
# - Configure global training parameters


# Dataset
# - Load the PathMNIST dataset
# - Inspect dataset structure and labels
# - Create training, validation, and test splits


# Data preprocessing
# - Normalize image pixel values
# - Apply optional data augmentation
#   - Random flips
#   - Random rotations
# - Ensure identical preprocessing for all models


# Model definitions
# - Simple CNN
#   - Basic convolution and pooling layers
#   - Fully connected classifier
#
# - ResNet-18
#   - Shallow residual architecture
#   - Fewer layers and parameters
#
# - ResNet-50
#   - Deeper residual architecture
#   - Higher model capacity


# Training strategy
# - Simple CNN
#   - Train from random initialization
#
# - ResNet-18
#   - Train from scratch
#   - Train with ImageNet pretrained weights
#
# - ResNet-50
#   - Train from scratch
#   - Train with ImageNet pretrained weights


# Training loop
# - Iterate over training data
# - Perform forward pass
# - Compute loss
# - Backpropagate gradients
# - Update model weights
# - Evaluate on validation set each epoch


# Evaluation
# - Load best-performing model checkpoints
# - Evaluate on held-out test set
# - Compute classification accuracy
# - Store results for comparison


# Analysis
# - Compare Simple CNN vs ResNet models
# - Compare ResNet-18 vs ResNet-50
# - Analyze effect of pretrained weights
# - Identify performance vs complexity trade-offs


# Conclusions
# - Summarize main findings
# - Discuss relevance to heal
