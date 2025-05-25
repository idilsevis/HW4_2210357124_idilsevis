# HW4_2210357124_idilsevis
Comprehensive CNN analysis for MNIST digit recognition: architectural variants, hyperparameter optimization, and failure case studies achieving 99.4% accuracy with PyTorch
CNN MNIST Handwritten Digit Recognition
A comprehensive study of Convolutional Neural Networks for handwritten digit recognition using the MNIST dataset in PyTorch. This project explores various CNN architectures, hyperparameter optimization, and failure analysis to understand deep learning model performance.
-Project Overview
This repository contains the complete implementation and analysis of CNN models for MNIST digit classification, including:

Baseline CNN Implementation: A foundational model achieving 99.20% test accuracy
Architectural Variants: Exploration of depth, kernel sizes, activation functions, and advanced components
Hyperparameter Analysis: Systematic testing of learning rates, optimizers, batch sizes, dropout, and weight initialization
Failure Case Studies: Intentional experiments demonstrating common pitfalls and failure modes
Architecture Details
Baseline CNN
Conv(1→32, 3x3) → ReLU → MaxPool(2x2) →
Conv(32→64, 3x3) → ReLU → MaxPool(2x2) →
Flatten → Dense(3136→128) → ReLU → Dropout(0.5) →
Dense(128→10) → LogSoftmax
Architectural Variants Tested

Deeper Networks: Additional convolutional layers
Variable Kernel Sizes: 5x5, 1x1, and 3x3 combinations
Alternative Activations: LeakyReLU vs ReLU
Modern Components: BatchNorm + Global Average Pooling

-Experimental Analysis
Hyperparameter Studies

Learning Rates: [0.0001, 0.001, 0.01, 0.1]
Optimizers: Adam vs SGD comparison
Batch Sizes: [32, 64, 128, 256]
Dropout Rates: [0.0, 0.25, 0.5, 0.75]
Weight Initialization: Default, Xavier, Kaiming, Normal
-Quick Start
Prerequisites
bashpip install torch torchvision matplotlib numpy
-Basic Usage
python# Run baseline and other experiment
python q123.py

# Failure case studies
python Q4.py
