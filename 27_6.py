from typing import Tuple, Optional
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.autograd import Function, gradcheck
from dataclasses import dataclass
import time
import matplotlib.pyplot as plt

print("="*60)
print("1. Introduction to Custom Autograd Functions")
print("="*60)

print("""
Custom Autograd Functions allow you to define custom forward and backward computations in PyTorch.
They are useful when:
1. You need operations that aren't natively supported in PyTorch.
2. You want to optimize memory or computation by customizing the backward pass.
3. You're implementing cutting-edge research ideas that require novel gradients.

Key Components:
- `forward()`: Computes the output using the input tensors.
- `backward()`: Computes the gradients during the backward pass.
- `ctx`: Context object for saving intermediate values between forward and backward passes.

We'll explore several scenarios with practical examples and compare them to built-in PyTorch operations.
""")

# --------------------------------------------------------------------------------
# 2. Basic Custom Function Example: ReLU
# --------------------------------------------------------------------------------

print("="*60)
print("2. Basic Custom Function: Custom ReLU")
print("="*60)

class CustomReLU(Function):
    """
    Custom implementation of the ReLU activation function.
    Purpose:
    - Demonstrates the basics of forward and backward methods.
    - Compares custom and built-in PyTorch ReLU implementations.
    """
    
    @staticmethod
    def forward(ctx, input):
        """
        Forward pass computes y = max(0, x).
        Saves the input tensor for use during the backward pass.
        """
        ctx.save_for_backward(input)  # Save input for backward pass
        return input.clamp(min=0)
    
    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass computes gradients w.r.t input:
        dy/dx = 1 if x > 0, otherwise 0.
        """
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input < 0] = 0
        return grad_input

def demonstrate_basic_autograd():
    print("\nPurpose:")
    print("Demonstrate how to create and verify a custom ReLU activation function.")
    print("Compare it with PyTorch's built-in ReLU for correctness and performance.\n")
    
    custom_relu = CustomReLU.apply
    x = torch.randn(1000, dtype=torch.float64, requires_grad=True)  # Double precision for gradcheck

    # Forward Pass
    y1 = custom_relu(x)
    y2 = torch.relu(x)
    print(f"Forward Pass: Max difference: {(y1 - y2).abs().max().item()}")

    # Backward Pass
    loss1 = y1.sum()
    loss2 = y2.sum()
    loss1.backward()
    grad1 = x.grad.clone()
    x.grad.zero_()
    loss2.backward()
    grad2 = x.grad
    print(f"Backward Pass: Max gradient difference: {(grad1 - grad2).abs().max().item()}")

    # Gradient Check
    print("\nGradient Check:")
    try:
        gradcheck(custom_relu, (x,), eps=1e-3, atol=1e-4)
        print("Gradient check passed!")
    except Exception as e:
        print(f"Gradient check failed: {e}")

demonstrate_basic_autograd()

# --------------------------------------------------------------------------------
# 3. Advanced Custom Function Example: LayerNorm
# --------------------------------------------------------------------------------

print("="*60)
print("3. Advanced Custom Function: Custom LayerNorm")
print("="*60)

class CustomLayerNorm(Function):
    """
    Custom implementation of Layer Normalization.
    Purpose:
    - Optimizes memory usage by tailoring the backward pass.
    - Highlights flexibility in implementing specific behavior.
    """
    
    @staticmethod
    def forward(ctx, input, eps=1e-5):
        """
        Forward pass computes:
        y = (x - mean) / sqrt(var + eps).
        Saves intermediate values for backward computation.
        """
        mean = input.mean(dim=-1, keepdim=True)
        var = input.var(dim=-1, keepdim=True, unbiased=False)
        x_norm = (input - mean) / torch.sqrt(var + eps)
        ctx.eps = eps
        ctx.save_for_backward(input, x_norm, mean, var)
        return x_norm
    
    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass computes gradients w.r.t. input.
        Uses intermediate values (mean, var) saved during the forward pass.
        """
        input, x_norm, mean, var = ctx.saved_tensors
        eps = ctx.eps
        N = input.size(-1)
        std = torch.sqrt(var + eps)
        dx_norm = grad_output
        dvar = (-0.5 * dx_norm * (input - mean) / (var + eps) ** 1.5).sum(dim=-1, keepdim=True)
        dmean = (-dx_norm / std).sum(dim=-1, keepdim=True)
        dx = dx_norm / std + 2 * dvar * (input - mean) / N + dmean / N
        return dx, None

def demonstrate_advanced_autograd():
    print("\nPurpose:")
    print("Implement a custom LayerNorm and compare it with PyTorch's built-in LayerNorm.")
    print("Highlight memory savings and numerical stability.\n")

    custom_ln = CustomLayerNorm.apply
    torch_ln = nn.LayerNorm(10)

    x = torch.randn(100, 10, requires_grad=True)

    # Memory usage tracking
    @dataclass
    class MemoryTracker:
        peak: float = 0
        current: float = 0

        def update(self):
            if torch.cuda.is_available():
                self.current = torch.cuda.memory_allocated() / 1024**2
                self.peak = max(self.peak, self.current)

    tracker = MemoryTracker()

    # Test Custom Implementation
    def test_custom():
        y = custom_ln(x)
        loss = y.sum()
        loss.backward()
        tracker.update()
        return tracker.current

    # Test PyTorch Implementation
    def test_torch():
        y = torch_ln(x)
        loss = y.sum()
        loss.backward()
        tracker.update()
        return tracker.current

    mem_custom = test_custom()
    x.grad.zero_()
    mem_torch = test_torch()

    print(f"Memory Usage Comparison:")
    print(f"Custom LayerNorm: {mem_custom:.2f}MB")
    print(f"PyTorch LayerNorm: {mem_torch:.2f}MB")

demonstrate_advanced_autograd()


print("="*60)
print("4. Custom Functions with Multiple Outputs")
print("="*60)

class CustomSplitActivation(Function):
    """
    Custom Function: Splits input into positive and negative parts.
    Purpose:
    - Demonstrate handling of multiple outputs in autograd.
    - Useful in scenarios like advanced activation functions or branching networks.
    """
    
    @staticmethod
    def forward(ctx, input):
        """
        Forward pass splits the input:
        - Positive values remain as-is.
        - Negative values are inverted (multiplied by -1).
        Saves the input tensor for backward computation.
        """
        pos = input.clamp(min=0)  # Positive part
        neg = (-input).clamp(min=0)  # Negative part
        ctx.save_for_backward(input)  # Save input for use in backward
        return pos, neg  # Return both outputs
    
    @staticmethod
    def backward(ctx, grad_pos, grad_neg):
        """
        Backward pass computes the gradients for the input:
        - Gradients for positive part are directly passed.
        - Gradients for negative part are inverted.
        """
        input, = ctx.saved_tensors
        grad_input = grad_pos.clone()
        grad_input[input < 0] = -grad_neg[input < 0]  # Invert gradient for negative part
        return grad_input

def demonstrate_multiple_outputs():
    print("""
Purpose:
- Illustrate how to implement custom autograd functions with multiple outputs.
- Use case: Operations where outputs branch out, such as gated activations.
""")
    split_activate = CustomSplitActivation.apply
    x = torch.randn(100, requires_grad=True)  # Input tensor with gradients enabled
    
    # Forward pass
    pos, neg = split_activate(x)
    
    # Compute loss as a combination of both outputs
    loss = pos.sum() + 0.5 * neg.sum()
    loss.backward()  # Backward pass
    
    # Print results
    print("\nMultiple Output Gradients:")
    print(f"Input shape: {x.shape}")
    print(f"Positive output shape: {pos.shape}")
    print(f"Negative output shape: {neg.shape}")
    print(f"Gradient shape: {x.grad.shape}")
    print("Gradient computation complete. Check for correctness!")

demonstrate_multiple_outputs()


print("="*60)
print("5. Performance Optimization: Optimized Matrix Multiplication")
print("="*60)

class OptimizedMatMul(Function):
    """
    Optimized Matrix Multiplication.
    Purpose:
    - Save memory during backpropagation by chunking operations.
    - Useful for large matrices that can't fit into GPU memory.
    """
    
    @staticmethod
    def forward(ctx, a, b):
        """
        Forward pass computes the standard matrix multiplication: C = A * B.
        Saves inputs A and B for the backward pass.
        """
        ctx.save_for_backward(a, b)
        return torch.mm(a, b)
    
    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass computes gradients:
        - Grad A: dL/dA = dL/dC * B^T
        - Grad B: dL/dB = A^T * dL/dC
        Uses chunking to reduce memory usage during large operations.
        """
        a, b = ctx.saved_tensors
        
        # Chunked matrix multiplication to save memory
        def chunked_mm(mat1, mat2, chunk_size=1024):
            rows, cols = mat1.size(0), mat2.size(1)
            result = torch.zeros(rows, cols, device=mat1.device)
            for i in range(0, rows, chunk_size):
                end = min(i + chunk_size, rows)
                result[i:end] = torch.mm(mat1[i:end], mat2)
            return result
        
        grad_a = chunked_mm(grad_output, b.t())
        grad_b = chunked_mm(a.t(), grad_output)
        return grad_a, grad_b

def demonstrate_optimization():
    print("""
Purpose:
- Compare standard PyTorch matrix multiplication with an optimized implementation.
- Highlight trade-offs in memory usage and computation time.
""")
    def profile_matmul(func, *args):
        """
        Profiles a matrix multiplication function for execution time and memory usage.
        """
        start_time = time.time()
        start_mem = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        
        result = func(*args)
        
        end_mem = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        end_time = time.time()
        
        return result, end_time - start_time, end_mem - start_mem

    # Test matrices
    size = 2048
    a = torch.randn(size, size, requires_grad=True)
    b = torch.randn(size, size, requires_grad=True)

    # Standard PyTorch matrix multiplication
    _, time1, mem1 = profile_matmul(torch.mm, a, b)

    # Optimized matrix multiplication
    optimized_mm = OptimizedMatMul.apply
    _, time2, mem2 = profile_matmul(optimized_mm, a, b)

    print("\nPerformance Comparison:")
    print(f"Standard MatMul: {time1:.4f}s, {mem1 / 1024**2:.2f}MB")
    print(f"Optimized MatMul: {time2:.4f}s, {mem2 / 1024**2:.2f}MB")
    print("Optimization trade-offs analyzed!")
    return time1, time2, mem1, mem2

time1, time2, mem1, mem2 = demonstrate_optimization()

# --------------------------------------------------------------------------------
# 6. Best Practices and Guidelines
# --------------------------------------------------------------------------------

print("\n"+"="*60)
print("6. Best Practices and Guidelines")
print("="*60)

print("""
Best Practices for Custom Autograd Functions:

1. Memory Management:
   - Use ctx.save_for_backward() judiciously
   - Clear temporary variables
   - Consider chunked operations for large tensors

2. Numerical Stability:
   - Handle edge cases
   - Use stable algorithms
   - Add small epsilon values when dividing

3. Performance:
   - Profile memory usage
   - Optimize backward pass
   - Use in-place operations when safe

4. Testing:
   - Use gradcheck for verification
   - Test edge cases
   - Compare with existing implementations

5. Documentation:
   - Document mathematical formulation
   - Explain limitations
   - Provide usage examples

Common Pitfalls:
1. Not handling all input types
2. Memory leaks in backward pass
3. Numerical instability
4. Incorrect gradient computation
5. Poor performance with large inputs
""") 

import matplotlib.pyplot as plt

# Example visualization of memory and time trade-offs
labels = ['Standard MatMul', 'Optimized MatMul']
time_data = [time1, time2]
memory_data = [mem1 / 1024**2, mem2 / 1024**2]

fig, ax1 = plt.subplots()

ax1.bar(labels, time_data, alpha=0.7, label='Time (s)')
ax1.set_ylabel('Time (s)')
ax1.set_xlabel('Matrix Multiplication Methods')

ax2 = ax1.twinx()
ax2.plot(labels, memory_data, color='r', marker='o', label='Memory (MB)')
ax2.set_ylabel('Memory (MB)')

fig.suptitle('Performance Comparison: Standard vs Optimized MatMul')
plt.show()
