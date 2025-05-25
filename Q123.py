import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import time
import os

# Set device
device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Create results directory
os.makedirs('results', exist_ok=True)

# Data loading
transform = transforms.ToTensor()

train_dataset = torchvision.datasets.MNIST('data', train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.MNIST('data', train=False, transform=transform)

# Q1: Baseline CNN
class BaselineCNN(nn.Module):
    def __init__(self):
        super(BaselineCNN, self).__init__()
        # Conv → ReLU → MaxPool → Conv → ReLU → MaxPool → Flatten → Dense → ReLU → Dropout(0.5) → Dense(10) → Softmax
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 10)
        
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

# Q2: Architectural Variants
class DeeperCNN(nn.Module):
    """Deeper network with more conv blocks"""
    def __init__(self):
        super(DeeperCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv4 = nn.Conv2d(64, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 10)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = self.pool(F.relu(self.conv4(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

class DifferentKernelCNN(nn.Module):
    """Using different kernel sizes (1x1, 5x5)"""
    def __init__(self):
        super(DifferentKernelCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, padding=2)  # 5x5 kernel
        self.conv2 = nn.Conv2d(32, 64, kernel_size=1)  # 1x1 kernel
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)  # 3x3 kernel
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 10)
        
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = F.relu(self.conv2(x))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

class LeakyReLUCNN(nn.Module):
    """Using LeakyReLU activation"""
    def __init__(self):
        super(LeakyReLUCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 10)
        
    def forward(self, x):
        x = self.pool(F.leaky_relu(self.conv1(x)))
        x = self.pool(F.leaky_relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = F.leaky_relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

class BatchNormCNN(nn.Module):
    """Using BatchNorm and GlobalAveragePooling"""
    def __init__(self):
        super(BatchNormCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(2, 2)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(64, 10)
        
    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.global_pool(x)
        x = x.view(-1, 64)
        x = self.fc(x)
        return F.log_softmax(x, dim=1)

# Training function
def train_model(model, train_loader, test_loader, epochs=10, lr=0.001, optimizer_type='adam'):
    model = model.to(device)
    
    if optimizer_type == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=lr)
    else:
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    
    criterion = nn.NLLLoss()
    
    train_losses = []
    train_accuracies = []
    test_losses = []
    test_accuracies = []
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
        
        train_loss /= len(train_loader)
        train_acc = 100. * correct / total
        
        # Testing
        model.eval()
        test_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                test_loss += criterion(output, target).item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)
        
        test_loss /= len(test_loader)
        test_acc = 100. * correct / total
        
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        test_losses.append(test_loss)
        test_accuracies.append(test_acc)
        
        print(f'Epoch {epoch+1}/{epochs}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%')
    
    return {
        'train_losses': train_losses,
        'train_accuracies': train_accuracies,
        'test_losses': test_losses,
        'test_accuracies': test_accuracies,
        'final_test_acc': test_acc
    }

# Improved plotting function that saves figures
def plot_results(results_dict, title="Training Results", filename=None):
    # Set matplotlib backend to Agg for non-interactive plotting
    plt.switch_backend('Agg')
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(title, fontsize=16)
    
    for name, results in results_dict.items():
        epochs = range(1, len(results['train_losses']) + 1)
        
        ax1.plot(epochs, results['train_losses'], label=f'{name} Train', marker='o', markersize=3)
        ax2.plot(epochs, results['test_losses'], label=f'{name} Test', marker='o', markersize=3)
        ax3.plot(epochs, results['train_accuracies'], label=f'{name} Train', marker='o', markersize=3)
        ax4.plot(epochs, results['test_accuracies'], label=f'{name} Test', marker='o', markersize=3)
    
    ax1.set_title('Training Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.set_title('Test Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    ax3.set_title('Training Accuracy')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Accuracy (%)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    ax4.set_title('Test Accuracy')
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Accuracy (%)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save the figure
    if filename is None:
        filename = title.replace(" ", "_").replace(":", "").lower()
    
    save_path = f'results/{filename}.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {save_path}")
    
    # Also try to display if running in interactive mode
    try:
        plt.show()
    except:
        pass
    
    plt.close()  # Important: close the figure to free memory

# Function to create a summary table
def create_summary_table(results_dict, title):
    print(f"\n{title}")
    print("-" * 60)
    print(f"{'Model/Config':<20} {'Final Test Acc (%)':<15} {'Best Test Acc (%)':<15}")
    print("-" * 60)
    
    for name, results in results_dict.items():
        final_acc = results['final_test_acc']
        best_acc = max(results['test_accuracies'])
        print(f"{name:<20} {final_acc:<15.2f} {best_acc:<15.2f}")
    print("-" * 60)

# Main execution
def main():
    # Standard data loaders
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)
    
    print("="*60)
    print("Q1: BASELINE CNN TRAINING")
    print("="*60)
    
    # Q1: Train baseline CNN
    baseline_model = BaselineCNN()
    baseline_results = train_model(baseline_model, train_loader, test_loader, epochs=10)
    
    # Save baseline results
    plot_results({'Baseline': baseline_results}, "Q1: Baseline CNN Training", "q1_baseline")
    
    print("\n" + "="*60)
    print("Q2: ARCHITECTURAL VARIANTS")
    print("="*60)
    
    # Q2: Train architectural variants
    models = {
        'Baseline': BaselineCNN(),
        'Deeper': DeeperCNN(),
        'Different Kernels': DifferentKernelCNN(),
        'LeakyReLU': LeakyReLUCNN(),
        'BatchNorm+GAP': BatchNormCNN()
    }
    
    architectural_results = {}
    for name, model in models.items():
        print(f"\nTraining {name} CNN:")
        results = train_model(model, train_loader, test_loader, epochs=10)
        architectural_results[name] = results
        print(f"Final Test Accuracy: {results['final_test_acc']:.2f}%")
    
    # Plot architectural comparison
    plot_results(architectural_results, "Q2: Architectural Variants Comparison", "q2_architectural_variants")
    create_summary_table(architectural_results, "ARCHITECTURAL VARIANTS SUMMARY")
    
    print("\n" + "="*60)
    print("Q3: HYPERPARAMETER ANALYSIS")
    print("="*60)
    
    # Q3: Hyperparameter analysis
    hyperparameter_results = {}
    
    # Learning rate analysis
    print("\n--- Learning Rate Analysis ---")
    learning_rates = [0.0001, 0.001, 0.01, 0.1]
    lr_results = {}
    
    for lr in learning_rates:
        print(f"\nTesting Learning Rate: {lr}")
        model = BaselineCNN()
        results = train_model(model, train_loader, test_loader, epochs=10, lr=lr)
        lr_results[f'LR_{lr}'] = results
        print(f"Final Test Accuracy: {results['final_test_acc']:.2f}%")
    
    plot_results(lr_results, "Q3: Learning Rate Comparison", "q3_learning_rates")
    create_summary_table(lr_results, "LEARNING RATE ANALYSIS SUMMARY")
    
    # Optimizer comparison
    print("\n--- Optimizer Analysis ---")
    optimizer_results = {}
    
    for opt in ['adam', 'sgd']:
        print(f"\nTesting Optimizer: {opt.upper()}")
        model = BaselineCNN()
        results = train_model(model, train_loader, test_loader, epochs=10, optimizer_type=opt)
        optimizer_results[opt.upper()] = results
        print(f"Final Test Accuracy: {results['final_test_acc']:.2f}%")
    
    plot_results(optimizer_results, "Q3: Optimizer Comparison", "q3_optimizers")
    create_summary_table(optimizer_results, "OPTIMIZER ANALYSIS SUMMARY")
    
    # Batch size analysis
    print("\n--- Batch Size Analysis ---")
    batch_sizes = [32, 64, 128, 256]
    batch_results = {}
    
    for batch_size in batch_sizes:
        print(f"\nTesting Batch Size: {batch_size}")
        train_loader_bs = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        model = BaselineCNN()
        results = train_model(model, train_loader_bs, test_loader, epochs=10)
        batch_results[f'BS_{batch_size}'] = results
        print(f"Final Test Accuracy: {results['final_test_acc']:.2f}%")
    
    plot_results(batch_results, "Q3: Batch Size Comparison", "q3_batch_sizes")
    create_summary_table(batch_results, "BATCH SIZE ANALYSIS SUMMARY")
    
    # Dropout rate analysis
    print("\n--- Dropout Rate Analysis ---")
    
    class VariableDropoutCNN(nn.Module):
        def __init__(self, dropout_rate=0.5):
            super(VariableDropoutCNN, self).__init__()
            self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
            self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
            self.pool = nn.MaxPool2d(2, 2)
            self.fc1 = nn.Linear(64 * 7 * 7, 128)
            self.dropout = nn.Dropout(dropout_rate)
            self.fc2 = nn.Linear(128, 10)
            
        def forward(self, x):
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            x = x.view(-1, 64 * 7 * 7)
            x = F.relu(self.fc1(x))
            x = self.dropout(x)
            x = self.fc2(x)
            return F.log_softmax(x, dim=1)
    
    dropout_rates = [0.0, 0.25, 0.5, 0.75]
    dropout_results = {}
    
    for dropout_rate in dropout_rates:
        print(f"\nTesting Dropout Rate: {dropout_rate}")
        model = VariableDropoutCNN(dropout_rate)
        results = train_model(model, train_loader, test_loader, epochs=10)
        dropout_results[f'Dropout_{dropout_rate}'] = results
        print(f"Final Test Accuracy: {results['final_test_acc']:.2f}%")
    
    plot_results(dropout_results, "Q3: Dropout Rate Comparison", "q3_dropout_rates")
    create_summary_table(dropout_results, "DROPOUT RATE ANALYSIS SUMMARY")
    
    # Weight initialization analysis
    print("\n--- Weight Initialization Analysis ---")
    
    def init_weights(m, init_type='xavier'):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            if init_type == 'xavier':
                nn.init.xavier_uniform_(m.weight)
            elif init_type == 'kaiming':
                nn.init.kaiming_uniform_(m.weight)
            elif init_type == 'normal':
                nn.init.normal_(m.weight, 0, 0.01)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
    
    init_types = ['default', 'xavier', 'kaiming', 'normal']
    init_results = {}
    
    for init_type in init_types:
        print(f"\nTesting Initialization: {init_type}")
        model = BaselineCNN()
        if init_type != 'default':
            model.apply(lambda m: init_weights(m, init_type))
        results = train_model(model, train_loader, test_loader, epochs=10)
        init_results[f'Init_{init_type}'] = results
        print(f"Final Test Accuracy: {results['final_test_acc']:.2f}%")
    
    plot_results(init_results, "Q3: Weight Initialization Comparison", "q3_weight_init")
    create_summary_table(init_results, "WEIGHT INITIALIZATION ANALYSIS SUMMARY")
    
    print("\n" + "="*60)
    print("FINAL SUMMARY OF ALL RESULTS")
    print("="*60)
    
    # Comprehensive summary
    all_results = [
        ("Architectural Variants", architectural_results),
        ("Learning Rates", lr_results),
        ("Optimizers", optimizer_results),
        ("Batch Sizes", batch_results),
        ("Dropout Rates", dropout_results),
        ("Weight Initialization", init_results)
    ]
    
    print("\nBest performing configurations:")
    for category, results in all_results:
        best_name = max(results.keys(), key=lambda k: results[k]['final_test_acc'])
        best_acc = results[best_name]['final_test_acc']
        print(f"{category:<25}: {best_name:<20} ({best_acc:.2f}%)")
    
    print(f"\nAll plots and results saved to 'results/' directory")
    print("="*60)

if __name__ == "__main__":
    main()