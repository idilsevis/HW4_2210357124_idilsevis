import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import seaborn as sns

# Set device
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# Data loading
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST('./data', train=False, transform=transform)

# Baseline CNN for comparison
class BaselineCNN(nn.Module):
    def __init__(self):
        super(BaselineCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 10)
    
    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

# Failed Experiment 1: Extreme Learning Rate
class FailedCNN1(nn.Module):
    """Same architecture as baseline but will use extreme learning rate"""
    def __init__(self):
        super(FailedCNN1, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 10)
    
    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

# Failed Experiment 2: Massive Over-parameterization with Tiny Dataset
class OverparameterizedCNN(nn.Module):
    """Extremely deep and wide network that will overfit horribly"""
    def __init__(self):
        super(OverparameterizedCNN, self).__init__()
        # Massive convolutional layers
        self.conv1 = nn.Conv2d(1, 256, 3, padding=1)
        self.conv2 = nn.Conv2d(256, 512, 3, padding=1)
        self.conv3 = nn.Conv2d(512, 512, 3, padding=1)
        self.conv4 = nn.Conv2d(512, 1024, 3, padding=1)
        self.conv5 = nn.Conv2d(1024, 1024, 3, padding=1)
        self.conv6 = nn.Conv2d(1024, 2048, 3, padding=1)
        
        self.pool = nn.MaxPool2d(2, 2)
        self.flatten = nn.Flatten()
        
        # Calculate the size after convolutions and pooling
        # After 3 pooling operations: 28 -> 14 -> 7 -> 3 (with padding)
        self.fc1 = nn.Linear(2048 * 3 * 3, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, 2048)
        self.fc4 = nn.Linear(2048, 1024)
        self.fc5 = nn.Linear(1024, 512)
        self.fc6 = nn.Linear(512, 10)
        
        # No dropout - intentionally allowing overfitting
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.pool(x)
        
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = self.pool(x)
        
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = self.fc6(x)
        return F.log_softmax(x, dim=1)

# Failed Experiment 3: Terrible Architecture Decisions
class TerribleCNN(nn.Module):
    """Intentionally bad architectural choices"""
    def __init__(self):
        super(TerribleCNN, self).__init__()
        # Huge kernels that destroy spatial information
        self.conv1 = nn.Conv2d(1, 8, kernel_size=15, stride=1, padding=7)  # Huge kernel, same size output
        self.conv2 = nn.Conv2d(8, 4, kernel_size=13, stride=1, padding=6)   # Another huge kernel
        
        # Aggressive pooling
        self.pool = nn.MaxPool2d(5, 5)  # Huge pooling that destroys information
        
        self.flatten = nn.Flatten()
        
        # Calculate correct output size:
        # Input: 28x28
        # After conv1 (with padding): 28x28
        # After first pool (5x5): floor(28/5) = 5, so 5x5
        # After conv2 (with padding): 5x5  
        # After second pool (5x5): floor(5/5) = 1, so 1x1
        # So final size is: 4 channels * 1 * 1 = 4
        
        self.fc1 = nn.Linear(4, 2)  # Extreme bottleneck - from 4 to 2 features
        self.fc2 = nn.Linear(2, 10)
        
        # Extreme dropout
        self.dropout = nn.Dropout(0.95)
    
    def forward(self, x):
        x = F.tanh(self.conv1(x))  # Tanh can saturate and kill gradients
        x = self.pool(x)  # First aggressive pooling: 28x28 -> 5x5
        x = F.tanh(self.conv2(x))  # Another tanh activation
        x = self.pool(x)  # Second aggressive pooling: 5x5 -> 1x1
        x = self.flatten(x)  # Should be batch_size x 4
        x = self.dropout(x)  # 95% dropout kills most information
        x = F.tanh(self.fc1(x))  # Bottleneck to just 2 features
        x = self.dropout(x)  # More extreme dropout
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

def train_model(model, train_loader, test_loader, optimizer, epochs=10, experiment_name=""):
    """Train model and return training history"""
    model.to(device)
    train_losses = []
    train_accuracies = []
    test_accuracies = []
    
    for epoch in range(epochs):
        # Training
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
            
            if batch_idx % 200 == 0:
                print(f'{experiment_name} Epoch {epoch+1}, Batch {batch_idx}, Loss: {loss.item():.6f}')
        
        train_accuracy = 100. * correct / total
        avg_loss = total_loss / len(train_loader)
        
        # Testing
        model.eval()
        test_correct = 0
        test_total = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                pred = output.argmax(dim=1, keepdim=True)
                test_correct += pred.eq(target.view_as(pred)).sum().item()
                test_total += target.size(0)
        
        test_accuracy = 100. * test_correct / test_total
        
        train_losses.append(avg_loss)
        train_accuracies.append(train_accuracy)
        test_accuracies.append(test_accuracy)
        
        print(f'{experiment_name} Epoch {epoch+1}: Train Acc: {train_accuracy:.2f}%, Test Acc: {test_accuracy:.2f}%, Loss: {avg_loss:.6f}')
    
    return train_losses, train_accuracies, test_accuracies

def run_experiments():
    """Run all three failed experiments plus baseline"""
    batch_size = 64
    epochs = 10
    
    # Standard data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Small dataset for overfitting experiment
    small_dataset = torch.utils.data.Subset(train_dataset, range(500))  # Only 500 samples
    small_train_loader = DataLoader(small_dataset, batch_size=batch_size, shuffle=True)
    
    results = {}
    
    # Baseline (for comparison)
    print("="*50)
    print("BASELINE EXPERIMENT")
    print("="*50)
    baseline_model = BaselineCNN()
    baseline_optimizer = optim.Adam(baseline_model.parameters(), lr=0.001)
    baseline_results = train_model(baseline_model, train_loader, test_loader, 
                                 baseline_optimizer, epochs, "Baseline")
    results['Baseline'] = baseline_results
    
    # Failed Experiment 1: Extreme Learning Rate
    print("="*50)
    print("FAILED EXPERIMENT 1: EXTREME LEARNING RATE")
    print("="*50)
    print("Using learning rate of 10.0 - this will cause gradient explosion")
    failed_model1 = FailedCNN1()
    failed_optimizer1 = optim.Adam(failed_model1.parameters(), lr=10.0)  # Extreme LR
    failed_results1 = train_model(failed_model1, train_loader, test_loader, 
                                failed_optimizer1, epochs, "Failed-1 (Extreme LR)")
    results['Failed Experiment 1'] = failed_results1
    
    # Failed Experiment 2: Overparameterization with tiny dataset
    print("="*50)
    print("FAILED EXPERIMENT 2: MASSIVE OVERPARAMETERIZATION")
    print("="*50)
    print("Using huge network with tiny dataset (500 samples) - severe overfitting expected")
    failed_model2 = OverparameterizedCNN()
    failed_optimizer2 = optim.Adam(failed_model2.parameters(), lr=0.001)
    failed_results2 = train_model(failed_model2, small_train_loader, test_loader, 
                                failed_optimizer2, epochs, "Failed-2 (Overfit)")
    results['Failed Experiment 2'] = failed_results2
    
    # Failed Experiment 3: Terrible Architecture
    print("="*50)
    print("FAILED EXPERIMENT 3: TERRIBLE ARCHITECTURE")
    print("="*50)
    print("Using huge kernels, aggressive pooling, extreme dropout, and bottleneck layers")
    failed_model3 = TerribleCNN()
    failed_optimizer3 = optim.Adam(failed_model3.parameters(), lr=0.001)
    failed_results3 = train_model(failed_model3, train_loader, test_loader, 
                                failed_optimizer3, epochs, "Failed-3 (Bad Arch)")
    results['Failed Experiment 3'] = failed_results3
    
    return results

def plot_results(results):
    """Plot comparison of all experiments"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Training Loss
    axes[0, 0].set_title('Training Loss Comparison')
    for name, (losses, _, _) in results.items():
        axes[0, 0].plot(losses, label=name, linewidth=2)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Training Accuracy
    axes[0, 1].set_title('Training Accuracy Comparison')
    for name, (_, train_acc, _) in results.items():
        axes[0, 1].plot(train_acc, label=name, linewidth=2)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy (%)')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Test Accuracy
    axes[1, 0].set_title('Test Accuracy Comparison')
    for name, (_, _, test_acc) in results.items():
        axes[1, 0].plot(test_acc, label=name, linewidth=2)
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Accuracy (%)')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # Final Performance Summary
    final_test_acc = {name: acc[-1] for name, (_, _, acc) in results.items()}
    names = list(final_test_acc.keys())
    accuracies = list(final_test_acc.values())
    
    bars = axes[1, 1].bar(range(len(names)), accuracies, 
                         color=['green', 'red', 'orange', 'purple'])
    axes[1, 1].set_title('Final Test Accuracy')
    axes[1, 1].set_xlabel('Experiment')
    axes[1, 1].set_ylabel('Test Accuracy (%)')
    axes[1, 1].set_xticks(range(len(names)))
    axes[1, 1].set_xticklabels(names, rotation=45, ha='right')
    
    # Add value labels on bars
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        axes[1, 1].text(bar.get_x() + bar.get_width()/2., height + 0.5,
                       f'{acc:.1f}%', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()

def analyze_failures():
    """Detailed analysis of why each experiment failed"""
    print("\n" + "="*80)
    print("DETAILED FAILURE ANALYSIS")
    print("="*80)
    
    print("\n1. FAILED EXPERIMENT 1: EXTREME LEARNING RATE (LR = 10.0)")
    print("-" * 60)
    print("WHY IT FAILED:")
    print("• Learning rate of 10.0 is extremely high for neural networks")
    print("• Causes gradient explosion - weights update too aggressively")
    print("• Network parameters oscillate wildly and never converge")
    print("• Loss may increase instead of decrease")
    print("• Accuracy remains at random guessing level (~10% for 10-class problem)")
    print("\nEXPECTED BEHAVIOR:")
    print("• Training loss will be erratic and high")
    print("• Accuracy will hover around 10% (random guessing)")
    print("• Gradients will explode, causing numerical instability")
    
    print("\n2. FAILED EXPERIMENT 2: MASSIVE OVERPARAMETERIZATION")
    print("-" * 60)
    print("WHY IT FAILED:")
    print("• Network has millions of parameters but only 500 training samples")
    print("• Severe overfitting - memorizes training data perfectly")
    print("• No generalization to test data")
    print("• Model complexity >> data complexity")
    print("• No regularization (dropout removed intentionally)")
    print("\nEXPECTED BEHAVIOR:")
    print("• Training accuracy reaches near 100%")
    print("• Test accuracy remains very low (poor generalization)")
    print("• Huge gap between train and test performance")
    
    print("\n3. FAILED EXPERIMENT 3: TERRIBLE ARCHITECTURE")
    print("-" * 60)
    print("WHY IT FAILED:")
    print("• Huge kernels (15x15, 13x13) destroy spatial relationships")
    print("• Aggressive pooling (5x5) loses critical information")
    print("• Extreme dropout (95%) randomly zeros most neurons")
    print("• Bottleneck layer (2 neurons) creates information bottleneck")
    print("• Tanh activation can saturate and kill gradients")
    print("\nEXPECTED BEHAVIOR:")
    print("• Poor feature extraction due to bad conv layers")
    print("• Information loss from aggressive pooling")
    print("• Training instability from extreme dropout")
    print("• Low accuracy due to insufficient representational capacity")

if __name__ == "__main__":
    # Run all experiments
    results = run_experiments()
    
    # Plot results
    plot_results(results)
    
    # Analyze failures
    analyze_failures()
    
    # Print final summary
    print("\n" + "="*80)
    print("FINAL PERFORMANCE SUMMARY")
    print("="*80)
    for name, (losses, train_acc, test_acc) in results.items():
        print(f"{name:25} | Final Test Accuracy: {test_acc[-1]:6.2f}% | Final Loss: {losses[-1]:8.6f}")