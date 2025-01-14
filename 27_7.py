import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import time
import numpy as np

# --------------------------------------------------------------------------------
# 1. Introduction to Gradient Debugging
# --------------------------------------------------------------------------------

print("="*60)
print("1. Introduction to Gradient Debugging")
print("="*60)

print("""
Gradient Debugging in PyTorch:
1. Common Issues:
   - Vanishing/Exploding gradients
   - Incorrect gradient flow
   - NaN/Inf values
   - Memory leaks

2. Debugging Tools:
   - Hooks
   - Gradient clipping
   - Anomaly detection
   - Visualization tools
""")

# --------------------------------------------------------------------------------
# 2. Forward and Backward Hooks
# --------------------------------------------------------------------------------

class GradientTracker:
    """
    Tracks gradients and activations using forward and backward hooks.
    
    Purpose:
    - Monitor gradient norms for vanishing or exploding gradients.
    - Inspect activations for potential issues (e.g., dead neurons).
    """

    def __init__(self):
        self.gradients: Dict[str, List[float]] = {}
        self.activations: Dict[str, List[float]] = {}
    
    def hook_fn(self, name: str) -> callable:
        """
        Hook function to track gradients.
        - Tracks the norm of gradients for each layer.
        """
        def hook(module, grad_input, grad_output):
            if name not in self.gradients:
                self.gradients[name] = []
            self.gradients[name].append(torch.norm(grad_output[0]).item())
        return hook
    
    def activation_hook(self, name: str) -> callable:
        """
        Hook function to track activations.
        - Tracks the norm of activations for each layer.
        """
        def hook(module, input, output):
            if name not in self.activations:
                self.activations[name] = []
            self.activations[name].append(torch.norm(output).item())
        return hook

def demonstrate_hooks():
    print("\nDemonstrating Gradient and Activation Hooks:")
    
    # Simple Model
    model = nn.Sequential(
        nn.Linear(10, 20),
        nn.ReLU(),
        nn.Linear(20, 1)
    )
    
    # Attach Hooks
    tracker = GradientTracker()
    hooks = []
    for name, layer in model.named_modules():
        if isinstance(layer, nn.Linear):
            hooks.append(layer.register_full_backward_hook(tracker.hook_fn(f"{name}_grad")))
            hooks.append(layer.register_forward_hook(tracker.activation_hook(f"{name}_activ")))
    
    # Training Loop
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    for i in range(100):
        x = torch.randn(32, 10)
        y = torch.randn(32, 1)
        optimizer.zero_grad()
        output = model(x)
        loss = nn.MSELoss()(output, y)
        loss.backward()
        optimizer.step()
    
    # Visualize Gradients and Activations
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    for name, grads in tracker.gradients.items():
        ax1.plot(grads, label=name)
    ax1.set_title('Gradient Norms Over Time')
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Gradient Norm')
    ax1.legend()
    
    for name, activs in tracker.activations.items():
        ax2.plot(activs, label=name)
    ax2.set_title('Activation Norms Over Time')
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Activation Norm')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('gradient_tracking.png')
    plt.show()
    
    # Remove hooks
    for hook in hooks:
        hook.remove()

demonstrate_hooks()


# --------------------------------------------------------------------------------
# 3. In-Place Operations and Their Pitfalls
# --------------------------------------------------------------------------------

class InPlaceDemo:
    """
    Demonstrates the impact of in-place operations on gradient computation.
    
    Purpose:
    - Educate about avoiding in-place operations in training.
    - Highlight runtime errors caused by in-place operations.
    """

    def demonstrate_inplace_issues(self):
        print("\nDemonstrating In-Place Operation Issues:")
        
        # Correct Version
        x = torch.tensor([2.0], requires_grad=True)
        h = x * 2
        h = h + 2  # Not in-place
        h.backward()
        print(f"Correct gradient: {x.grad.item()}")
        
        # Incorrect Version
        x = torch.tensor([2.0], requires_grad=True)
        h = x * 2
        h += 2  # In-place operation
        try:
            h.backward()
        except RuntimeError as e:
            print(f"Error with in-place operation: {e}")

InPlaceDemo().demonstrate_inplace_issues()


# --------------------------------------------------------------------------------
# 4. Gradient Anomaly Detection
# --------------------------------------------------------------------------------

class GradientAnomalyDetector:
    """Detects and visualizes gradient anomalies."""
    def __init__(self):
        self.gradient_history = []
        self.anomalies = []
    
    def detect_anomalies(self, grad: torch.Tensor, threshold: float = 5.0) -> bool:
        """Detects if gradient is anomalous."""
        if grad is None:
            return False
        
        grad_norm = torch.norm(grad).item()
        self.gradient_history.append(grad_norm)
        
        if len(self.gradient_history) > 1:
            prev_norm = self.gradient_history[-2]
            if abs(grad_norm / prev_norm) > threshold:
                self.anomalies.append(len(self.gradient_history) - 1)
                return True
        return False
    
    def visualize_anomalies(self):
        plt.figure(figsize=(12, 6))
        plt.plot(self.gradient_history, label='Gradient Norm')
        plt.scatter(self.anomalies, 
                   [self.gradient_history[i] for i in self.anomalies],
                   color='red', label='Anomalies')
        plt.title('Gradient Anomaly Detection')
        plt.xlabel('Iteration')
        plt.ylabel('Gradient Norm')
        plt.legend()
        plt.savefig('gradient_anomalies.png')
        plt.close()

def demonstrate_anomaly_detection():
    print("\nDemonstrating Gradient Anomaly Detection:")
    
    model = nn.Sequential(
        nn.Linear(10, 10),
        nn.ReLU(),
        nn.Linear(10, 1)
    )
    
    detector = GradientAnomalyDetector()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    
    for i in range(100):
        x = torch.randn(32, 10)
        # Intentionally create some anomalies
        y = torch.randn(32, 1) * (10 if i % 20 == 0 else 1)
        
        optimizer.zero_grad()
        output = model(x)
        loss = nn.MSELoss()(output, y)
        loss.backward()
        
        # Check gradients
        for name, param in model.named_parameters():
            if detector.detect_anomalies(param.grad):
                print(f"Anomaly detected in {name} at iteration {i}")
        
        optimizer.step()
    
    detector.visualize_anomalies()

demonstrate_anomaly_detection()

# --------------------------------------------------------------------------------
# 5. Advanced Debugging Techniques (PhD Level)
# --------------------------------------------------------------------------------

class GradientDebugger:
    """Advanced gradient debugging tools."""
    
    @staticmethod
    def analyze_gradient_flow(model: nn.Module) -> Dict[str, float]:
        """Analyzes gradient flow through the model."""
        grad_stats = {}
        
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_stats[name] = {
                    'mean': param.grad.mean().item(),
                    'std': param.grad.std().item(),
                    'norm': param.grad.norm().item(),
                    'max': param.grad.max().item(),
                    'min': param.grad.min().item()
                }
        
        return grad_stats
    
    @staticmethod
    def visualize_gradient_flow(grad_stats: Dict[str, Dict[str, float]]):
        """Creates visualization of gradient statistics."""
        metrics = ['mean', 'std', 'norm', 'max', 'min']
        layers = list(grad_stats.keys())
        
        fig, axes = plt.subplots(len(metrics), 1, figsize=(12, 4*len(metrics)))
        fig.suptitle('Gradient Flow Analysis')
        
        for i, metric in enumerate(metrics):
            values = [grad_stats[layer][metric] for layer in layers]
            axes[i].bar(layers, values)
            axes[i].set_title(f'Gradient {metric.capitalize()}')
            axes[i].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig('gradient_flow_analysis.png')
        plt.close()

def demonstrate_advanced_debugging():
    print("\nDemonstrating Advanced Gradient Debugging:")
    
    # Create a complex model
    model = nn.Sequential(
        nn.Linear(20, 40),
        nn.ReLU(),
        nn.Linear(40, 20),
        nn.ReLU(),
        nn.Linear(20, 10),
        nn.ReLU(),
        nn.Linear(10, 1)
    )
    
    debugger = GradientDebugger()
    optimizer = torch.optim.Adam(model.parameters())
    
    # Training with gradient analysis
    x = torch.randn(64, 20)
    y = torch.randn(64, 1)
    
    optimizer.zero_grad()
    output = model(x)
    loss = nn.MSELoss()(output, y)
    loss.backward()
    
    # Analyze gradients
    grad_stats = debugger.analyze_gradient_flow(model)
    debugger.visualize_gradient_flow(grad_stats)
    
    print("\nGradient Statistics:")
    for layer, stats in grad_stats.items():
        print(f"\n{layer}:")
        for metric, value in stats.items():
            print(f"  {metric}: {value:.6f}")

demonstrate_advanced_debugging()

# --------------------------------------------------------------------------------
# 6. Best Practices and Guidelines
# --------------------------------------------------------------------------------

print("\n"+"="*60)
print("6. Best Practices and Guidelines")
print("="*60)

print("""
Best Practices for Gradient Debugging:

1. Hook Usage:
   - Use hooks sparingly (they add overhead)
   - Remove hooks when not needed
   - Monitor memory usage with hooks

2. In-Place Operations:
   - Avoid in-place ops during training
   - Use clone() when needed
   - Be careful with view operations

3. Gradient Checks:
   - Regular gradient norm monitoring
   - Use torch.autograd.gradcheck for validation
   - Implement gradient clipping

4. Memory Management:
   - Clear unused variables
   - Use del for large tensors
   - Monitor GPU memory usage

5. Debugging Tools:
   - PyTorch profiler
   - Tensorboard
   - Custom visualization tools

Common Issues:
1. Vanishing/Exploding gradients
2. Broken computational graphs
3. Memory leaks from hooks
4. NaN gradients
5. Incorrect backward passes
""") 