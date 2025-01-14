import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import Function, grad, gradcheck
import time
import graphviz
from torchviz import make_dot

# --------------------------------------------------------------------------------
# 1. Introduction to Automatic Differentiation
# --------------------------------------------------------------------------------

print("="*60)
print("1. Introduction to Automatic Differentiation")
print("="*60)

print("""
Automatic Differentiation (AutoGrad) is a fundamental technique in deep learning that allows us to automatically compute gradients of functions. Unlike numerical differentiation (finite differences) or symbolic differentiation, AutoGrad combines the best of both worlds.

Key Concepts:
- Forward Pass: Computing the output of a function
- Backward Pass: Computing gradients with respect to inputs
- Computational Graph: A directed graph representing computations
- Chain Rule: The fundamental principle behind backpropagation
""")

print("""
Why is AutoGrad important?
- Automatic computation of gradients for complex functions
- Enables efficient training of neural networks
- Handles dynamic computational graphs
- Supports higher-order derivatives
""")

# --------------------------------------------------------------------------------
# 2. Basic AutoGrad Examples
# --------------------------------------------------------------------------------

print("\n"+"="*60)
print("2. Basic AutoGrad Examples")
print("="*60)

print("Let's start with simple examples to understand AutoGrad mechanics:\n")

# Simple scalar example
x = torch.tensor([2.0], requires_grad=True)
y = x ** 2
y.backward()
print(f"d(x^2)/dx at x=2: {x.grad}")  # Should be 4

# Reset gradients
x.grad.zero_()

# More complex example
z = torch.sin(x) * torch.exp(x)
z.backward()
print(f"d(sin(x)*exp(x))/dx at x=2: {x.grad}")

print("""
Key Points:
- requires_grad=True enables gradient tracking
- backward() computes gradients
- grad holds the computed gradient
- zero_() resets gradients
""")

# --------------------------------------------------------------------------------
# 3. Computational Graphs
# --------------------------------------------------------------------------------

print("\n"+"="*60)
print("3. Computational Graphs")
print("="*60)

def visualize_comp_graph():
    x = torch.tensor(2.0, requires_grad=True)
    a = x ** 2
    b = torch.sin(a)
    c = torch.exp(b)
    y = c * x
    
    print("Computational Graph Example:")
    print(f"x = {x.item()}")
    print(f"a = x² = {a.item()}")
    print(f"b = sin(a) = {b.item()}")
    print(f"c = exp(b) = {c.item()}")
    print(f"y = c * x = {y.item()}")
    
    # Compute gradients
    y.backward()
    print(f"\nGradient at x: {x.grad}")

visualize_comp_graph()

# --------------------------------------------------------------------------------
# 3.1 Visualizing Computational Graphs
# --------------------------------------------------------------------------------

print("\n"+"="*60)
print("3.1 Visualizing Computational Graphs")
print("="*60)

def visualize_with_graphviz():
    x = torch.tensor(2.0, requires_grad=True)
    a = x ** 2
    b = torch.sin(a)
    c = torch.exp(b)
    y = c * x
    
    # Create visualization
    dot = make_dot(y, params={"x": x})
    dot.render("computational_graph", format="png")
    
    print("""
    Computational Graph Visualization:
    - Nodes represent operations (functions)
    - Edges represent data flow
    - Leaf nodes are inputs
    - Root node is the output
    - Gradient flows backward through this graph
    """)
    
    # Show intermediate values and gradients
    y.backward()
    print("\nForward Pass Values:")
    print(f"x = {x.item():.4f}")
    print(f"a = x² = {a.item():.4f}")
    print(f"b = sin(a) = {b.item():.4f}")
    print(f"c = exp(b) = {c.item():.4f}")
    print(f"y = c * x = {y.item():.4f}")
    
    print("\nBackward Pass Gradients:")
    print(f"dy/dx = {x.grad.item():.4f}")
    
visualize_with_graphviz()

# --------------------------------------------------------------------------------
# 4. Custom Autograd Functions
# --------------------------------------------------------------------------------

print("\n"+"="*60)
print("4. Custom Autograd Functions")
print("="*60)

class CustomReLU(Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input.clamp(min=0)
    
    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input < 0] = 0
        return grad_input

# Test custom ReLU
custom_relu = CustomReLU.apply
x = torch.randn(5, requires_grad=True)
y = custom_relu(x)
print(f"Input: {x.data}")
print(f"Output: {y.data}")

# Compute gradients
z = y.sum()
z.backward()
print(f"Gradients: {x.grad}")

# --------------------------------------------------------------------------------
# 5. Higher-Order Derivatives
# --------------------------------------------------------------------------------

print("\n"+"="*60)
print("5. Higher-Order Derivatives")
print("="*60)

def compute_higher_order():
    x = torch.tensor([2.0], requires_grad=True)
    
    # First derivative of x^3
    def first_derivative(x):
        return 3 * x**2
    
    # Second derivative of x^3
    def second_derivative(x):
        return 6 * x
    
    # Compute using autograd
    y = x**3
    first_grad = grad(y.sum(), x, create_graph=True)[0]
    second_grad = grad(first_grad.sum(), x)[0]
    
    print(f"Function: x^3")
    print(f"First derivative at x=2: {first_grad.item()} (Expected: {first_derivative(x).item()})")
    print(f"Second derivative at x=2: {second_grad.item()} (Expected: {second_derivative(x).item()})")

compute_higher_order()

# --------------------------------------------------------------------------------
# 5.5 Gradient Clipping and Scaling
# --------------------------------------------------------------------------------

print("\n"+"="*60)
print("5.5 Gradient Clipping and Scaling")
print("="*60)

def demonstrate_gradient_clipping():
    # Create a simple neural network
    model = torch.nn.Sequential(
        torch.nn.Linear(10, 10),
        torch.nn.ReLU(),
        torch.nn.Linear(10, 1)
    )
    
    # Generate random data
    x = torch.randn(32, 10)
    y = torch.randn(32, 1)
    
    # Function to print gradient norms
    def print_grad_norms():
        total_norm = 0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        return total_norm
    
    # Training loop with different clipping methods
    optimizers = {
        'No Clipping': torch.optim.SGD(model.parameters(), lr=1.0),
        'Norm Clipping': torch.optim.SGD(model.parameters(), lr=1.0),
        'Value Clipping': torch.optim.SGD(model.parameters(), lr=1.0)
    }
    
    for name, optimizer in optimizers.items():
        # Forward pass
        output = model(x)
        loss = (output - y).pow(2).mean()
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Print original gradients
        print(f"\n{name}:")
        print(f"Original gradient norm: {print_grad_norms():.4f}")
        
        if name == 'Norm Clipping':
            # Clip gradient norm to 1.0
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            print(f"After norm clipping: {print_grad_norms():.4f}")
        
        elif name == 'Value Clipping':
            # Clip gradient values to [-0.5, 0.5]
            torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=0.5)
            print(f"After value clipping: {print_grad_norms():.4f}")
        
        optimizer.step()

print("""
Gradient Clipping Methods:
1. Norm Clipping (clip_grad_norm_):
   - Scales gradients so their norm doesn't exceed a threshold
   - Preserves the direction of gradients
   - Useful for preventing exploding gradients in RNNs

2. Value Clipping (clip_grad_value_):
   - Clamps each gradient value to a range
   - Changes gradient direction
   - Simpler but less theoretically justified

Use Cases:
- Deep networks (especially RNNs/LSTMs)
- Unstable training
- Large learning rates
- Dealing with outliers in data
""")

demonstrate_gradient_clipping()

# --------------------------------------------------------------------------------
# 6. Advanced Topics (PhD Level)
# --------------------------------------------------------------------------------

print("\n"+"="*60)
print("6. Advanced Topics (PhD Level)")
print("="*60)

print("""
Advanced AutoGrad Concepts:
1. Implicit Gradients
2. Jacobian and Hessian Computation
3. Vector-Jacobian Products
4. Performance Optimization
""")

# Example of Jacobian computation
def compute_jacobian():
    x = torch.randn(3, requires_grad=True)
    
    # Create y without in-place operations
    y = torch.stack([
        x[0]**2,  # First function: f₁(x) = x₀²
        x[1]**3,  # Second function: f₂(x) = x₁³
        x[2]**4   # Third function: f₃(x) = x₂⁴
    ])
    
    jacobian = torch.zeros(3, 3)
    for i in range(3):
        if i > 0:  # Clear gradients from previous iteration
            x.grad.zero_()
        # Compute gradient of y[i] with respect to x
        y[i].backward(retain_graph=True)
        jacobian[i] = x.grad.clone()  # Clone to prevent modification
    
    print("\nInput x:", x.detach())
    print("Function y:", y.detach())
    print("Jacobian Matrix:")
    print(jacobian)
    print("\nJacobian Explanation:")
    print("Each row i shows ∂y_i/∂x_j for all j")
    print(f"Row 1: ∂(x₀²)/∂x = [{2*x[0].item():.2f}, 0, 0]")
    print(f"Row 2: ∂(x₁³)/∂x = [0, {3*(x[1]**2).item():.2f}, 0]")
    print(f"Row 3: ∂(x₂⁴)/∂x = [0, 0, {4*(x[2]**3).item():.2f}]")

    # Visualize the Jacobian matrix
    plt.figure(figsize=(6, 6))
    plt.imshow(jacobian.detach(), cmap='coolwarm', aspect='equal')
    plt.colorbar(label='Gradient Value')
    plt.title('Jacobian Matrix Visualization')
    plt.xlabel('Input dimension (j)')
    plt.ylabel('Output dimension (i)')
    plt.savefig('jacobian_matrix.png')
    plt.close()
    
    print("\nJacobian matrix visualization saved as 'jacobian_matrix.png'")
    print("Shows how each output depends on each input")

# Add visualization of Jacobian
def visualize_jacobian():
    plt.figure(figsize=(8, 6))
    x = torch.linspace(-2, 2, 100, requires_grad=True)
    
    # Compute Jacobian for different functions
    y1 = x**2
    y2 = x**3
    y3 = torch.sin(x)
    
    # Get gradients
    y1.backward(torch.ones_like(x), retain_graph=True)
    grad1 = x.grad.clone()
    x.grad.zero_()
    
    y2.backward(torch.ones_like(x), retain_graph=True)
    grad2 = x.grad.clone()
    x.grad.zero_()
    
    y3.backward(torch.ones_like(x))
    grad3 = x.grad
    
    # Plot derivatives
    plt.plot(x.detach(), grad1.detach(), label='d(x²)/dx = 2x')
    plt.plot(x.detach(), grad2.detach(), label='d(x³)/dx = 3x²')
    plt.plot(x.detach(), grad3.detach(), label='d(sin(x))/dx = cos(x)')
    
    plt.title('Visualization of Different Derivatives')
    plt.xlabel('x')
    plt.ylabel('dy/dx')
    plt.legend()
    plt.grid(True)
    plt.savefig('jacobian_visualization.png')
    plt.close()
    
    print("\nJacobian visualization saved as 'jacobian_visualization.png'")
    print("Shows how derivatives change with respect to input x")

print("\nJacobian Computation and Visualization:")
compute_jacobian()
visualize_jacobian()

# Example of gradient checkpointing
print("\nGradient Checkpointing Example:")
def memory_efficient_computation():
    class CheckpointedFunction(Function):
        @staticmethod
        def forward(ctx, x):
            ctx.save_for_backward(x)
            return x ** 3
        
        @staticmethod
        def backward(ctx, grad_output):
            x, = ctx.saved_tensors
            return 3 * x ** 2 * grad_output
    
    x = torch.randn(1000, requires_grad=True)
    
    # Standard computation
    start_time = time.time()
    y1 = x ** 3
    y1.sum().backward()
    standard_time = time.time() - start_time
    
    # Checkpointed computation
    x.grad.zero_()
    start_time = time.time()
    y2 = CheckpointedFunction.apply(x)
    y2.sum().backward()
    checkpointed_time = time.time() - start_time
    
    print(f"Standard computation time: {standard_time:.4f}s")
    print(f"Checkpointed computation time: {checkpointed_time:.4f}s")

memory_efficient_computation()

print("""
Research Directions:
- Efficient gradient computation for large-scale models
- Novel automatic differentiation techniques
- Memory-efficient backpropagation
- Applications in scientific computing and optimization
""")

# --------------------------------------------------------------------------------
# 7. Best Practices and Common Pitfalls
# --------------------------------------------------------------------------------

print("\n"+"="*60)
print("7. Best Practices and Common Pitfalls")
print("="*60)

print("""
Best Practices:
1. Use retain_graph=True when needed
2. Clear gradients with zero_() before new backward pass
3. Detach tensors when you don't need gradients
4. Use no_grad() for inference
""")

def demonstrate_pitfalls():
    # 1. Gradient accumulation
    x = torch.tensor([2.0], requires_grad=True)
    y = x ** 2
    y.backward(retain_graph=True)
    print("First gradient:", x.grad)
    y.backward()  # Gradients accumulate!
    print("Accumulated gradient:", x.grad)
    
    # 2. Memory leak prevention
    x = torch.tensor([2.0], requires_grad=True)
    for _ in range(3):
        y = x ** 2
        y.backward()
        print("Without zeroing grad:", x.grad)
        x.grad.zero_()

print("\nCommon Pitfalls Demonstration:")
demonstrate_pitfalls() 