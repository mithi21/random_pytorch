import numpy as np
import torch
import torch.nn as nn
import time

# --------------------------------------------------------------------------------
# 1. Introduction to Convolution
# --------------------------------------------------------------------------------

print("="*60)
print("1. Introduction to Convolution")
print("="*60)

print("""
Convolution is a fundamental operation in signal and image processing. It involves sliding a kernel (or filter) over an input, performing element-wise multiplication, and summing the results to produce an output.
""")

print("""
Why are convolutions useful?
- Feature extraction: Convolutions can detect patterns, edges, and textures in images.
- Noise reduction: Convolutions can be used to smooth out noise in signals.
- Feature maps: Convolutions can create feature maps that highlight important aspects of the input.
""")

print("""
Mathematically, a 1D convolution is defined as:
(f * g)[n] = sum_{m=-inf}^{inf} f[m] * g[n - m]
where f is the input signal and g is the kernel.

For 2D convolution, it's:
(f * g)[i, j] = sum_{m=-inf}^{inf} sum_{n=-inf}^{inf} f[m, n] * g[i - m, j - n]
where f is the input image and g is the kernel.
""")

print("""
Intuitive Explanation:
Imagine sliding a small window (the kernel) over a larger input. At each position, you multiply the values in the window with the corresponding values in the input and sum them up. This sum becomes one element in the output.
""")

# --------------------------------------------------------------------------------
# 2. 1D Convolution
# --------------------------------------------------------------------------------

print("\n"+"="*60)
print("2. 1D Convolution")
print("="*60)

print("""
Let's start with a 1D convolution. We'll implement it from scratch using Python and NumPy.
""")

def conv1d_naive(input_signal, kernel, padding=0, stride=1):
    input_len = len(input_signal)
    kernel_len = len(kernel)
    
    padded_input = np.pad(input_signal, (padding, padding), 'constant')
    
    output_len = (len(padded_input) - kernel_len) // stride + 1
    output = np.zeros(output_len)
    
    for i in range(output_len):
        start = i * stride
        end = start + kernel_len
        output[i] = np.sum(padded_input[start:end] * kernel)
    
    return output

# Example usage
input_signal = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
kernel_1d = np.array([1, 0, -1])

print("Input Signal:", input_signal)
print("Kernel:", kernel_1d)

output_no_pad = conv1d_naive(input_signal, kernel_1d)
print("Output (no padding, stride=1):", output_no_pad)

output_pad = conv1d_naive(input_signal, kernel_1d, padding=1)
print("Output (padding=1, stride=1):", output_pad)

output_stride = conv1d_naive(input_signal, kernel_1d, stride=2)
print("Output (no padding, stride=2):", output_stride)

print("""
Padding:
- Padding adds extra values (usually zeros) around the input.
- It helps control the output size and prevent information loss at the edges.

Stride:
- Stride determines how many steps the kernel moves at each iteration.
- A larger stride reduces the output size.
""")

# --------------------------------------------------------------------------------
# 3. 2D Convolution
# --------------------------------------------------------------------------------

print("\n"+"="*60)
print("3. 2D Convolution")
print("="*60)

print("""
Now, let's extend the concept to 2D convolution.
""")

def conv2d_naive(input_image, kernel, padding=0, stride=1):
    input_height, input_width = input_image.shape
    kernel_height, kernel_width = kernel.shape
    
    padded_input = np.pad(input_image, ((padding, padding), (padding, padding)), 'constant')
    
    output_height = (padded_input.shape[0] - kernel_height) // stride + 1
    output_width = (padded_input.shape[1] - kernel_width) // stride + 1
    output = np.zeros((output_height, output_width))
    
    for i in range(output_height):
        for j in range(output_width):
            start_h = i * stride
            end_h = start_h + kernel_height
            start_w = j * stride
            end_w = start_w + kernel_width
            output[i, j] = np.sum(padded_input[start_h:end_h, start_w:end_w] * kernel)
    
    return output

# Example usage
input_image = np.array([[1, 2, 3, 4],
                        [5, 6, 7, 8],
                        [9, 10, 11, 12],
                        [13, 14, 15, 16]])
kernel_2d = np.array([[1, 0, -1],
                     [1, 0, -1],
                     [1, 0, -1]])

print("Input Image:\n", input_image)
print("Kernel:\n", kernel_2d)

output_no_pad = conv2d_naive(input_image, kernel_2d)
print("Output (no padding, stride=1):\n", output_no_pad)

output_pad = conv2d_naive(input_image, kernel_2d, padding=1)
print("Output (padding=1, stride=1):\n", output_pad)

output_stride = conv2d_naive(input_image, kernel_2d, stride=2)
print("Output (no padding, stride=2):\n", output_stride)

print("""
In 2D convolution, the kernel slides over the input image both horizontally and vertically.
The kernel is also called a filter.
Input channels: If the input is a color image, it has multiple channels (e.g., RGB).
Output channels: Multiple kernels can be used to create multiple output channels (feature maps).
""")

# --------------------------------------------------------------------------------
# 4. Convolution in PyTorch
# --------------------------------------------------------------------------------

print("\n"+"="*60)
print("4. Convolution in PyTorch")
print("="*60)

print("""
Now, let's see how to perform convolution using PyTorch.
""")

# 1D Convolution in PyTorch
def conv1d_torch(input_signal):
    # Shape: [batch, channel, length]
    input_signal_torch = torch.tensor(input_signal, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    # Shape: [out_channels, in_channels, kernel_size]
    kernel_torch = torch.tensor(kernel_1d, dtype=torch.float32).view(1, 1, 3)
    
    # Apply convolution
    conv1d = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=3, padding=0, stride=1)
    with torch.no_grad():
        conv1d.weight.copy_(kernel_torch)
        conv1d.bias.zero_()
    
    output_torch = conv1d(input_signal_torch)
    return output_torch.squeeze()

# Example usage with different parameters
output_torch_1d = conv1d_torch(input_signal)  # No padding, stride=1
print("PyTorch 1D Convolution Output (no padding, stride=1):\n", output_torch_1d.detach().numpy())

# Create input tensor once for padding and stride examples
input_signal_torch = torch.tensor(input_signal, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
kernel_torch = torch.tensor(kernel_1d, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

# Add padding
conv1d_pad = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=3, padding=1, stride=1)
with torch.no_grad():
    conv1d_pad.weight.copy_(kernel_torch)
    conv1d_pad.bias.zero_()
output_torch_1d_pad = conv1d_pad(input_signal_torch).squeeze()
print("PyTorch 1D Convolution Output (padding=1, stride=1):\n", output_torch_1d_pad.detach().numpy())

# Add stride
conv1d_stride = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=3, padding=0, stride=2)
with torch.no_grad():
    conv1d_stride.weight.copy_(kernel_torch)
    conv1d_stride.bias.zero_()
output_torch_1d_stride = conv1d_stride(input_signal_torch).squeeze()
print("PyTorch 1D Convolution Output (no padding, stride=2):\n", output_torch_1d_stride.detach().numpy())

# 2D Convolution in PyTorch
def conv2d_torch(input_image):
    # Shape: [batch, channel, height, width]
    input_image_torch = torch.tensor(input_image, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    # Shape: [out_channels, in_channels, kernel_height, kernel_width]
    kernel_torch = torch.tensor(kernel_2d, dtype=torch.float32).view(1, 1, 3, 3)
    
    # Apply convolution
    conv2d = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, padding=0, stride=1)
    with torch.no_grad():
        conv2d.weight.copy_(kernel_torch)
        conv2d.bias.zero_()
    
    output_torch = conv2d(input_image_torch)
    return output_torch.squeeze()

# Example usage
output_torch_2d = conv2d_torch(input_image)
print("PyTorch 2D Convolution Output (no padding, stride=1):\n", output_torch_2d.detach().numpy())

output_torch_2d_pad = conv2d_torch(input_image)
print("PyTorch 2D Convolution Output (padding=1, stride=1):\n", output_torch_2d_pad.detach().numpy())

output_torch_2d_stride = conv2d_torch(input_image)
print("PyTorch 2D Convolution Output (no padding, stride=2):\n", output_torch_2d_stride.detach().numpy())

print("""
PyTorch provides optimized implementations of convolution.
- torch.nn.Conv1d for 1D convolution.
- torch.nn.Conv2d for 2D convolution.
- Parameters include in_channels, out_channels, kernel_size, padding, stride, bias.
""")

# --------------------------------------------------------------------------------
# 5. Internals of Convolution
# --------------------------------------------------------------------------------

print("\n"+"="*60)
print("5. Internals of Convolution")
print("="*60)

print("""
Under the hood, convolution is often implemented using optimized algorithms.
""")

print("""
Hardware Implementation:
- Convolution is often implemented using specialized hardware like GPUs.
- GPUs have parallel processing capabilities that make convolution very fast.
""")

print("""
Optimized Algorithms:
- im2col: This algorithm transforms the input into a matrix, allowing convolution to be performed as a matrix multiplication.
- This is more efficient than the naive sliding window approach.
""")

print("""
Memory Considerations:
- Large convolutions can require a lot of memory.
- Techniques like tiling and memory-efficient algorithms are used to handle large inputs.
""")

print("""
Relationship with Matrix Multiplication:
- Convolution can be expressed as a matrix multiplication using im2col.
- This allows us to leverage highly optimized matrix multiplication libraries.
""")

# --------------------------------------------------------------------------------
# 6. Advanced Topics (for PhD level)
# --------------------------------------------------------------------------------

print("\n"+"="*60)
print("6. Advanced Topics (for PhD level)")
print("="*60)

print("""
Advanced Convolution Techniques:
- Transposed Convolution (Deconvolution): Used for upsampling.
- Dilated Convolution: Allows for a larger receptive field without increasing the number of parameters.
- Depthwise Separable Convolution: Reduces the number of parameters and computations.
""")

print("""
Fast Convolution Algorithms:
- Winograd Algorithm: A fast algorithm for small convolutions.
- Convolution in the Frequency Domain: Can be faster for very large kernels.
""")

print("""
Research Directions:
- Optimizing convolution for different hardware architectures.
- Developing new convolution algorithms for specific applications.
- Exploring the use of convolution in non-Euclidean spaces.
""") 