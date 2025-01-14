import torch
import numpy as np
import psutil
import sys
import time
import matplotlib.pyplot as plt
from memory_profiler import profile
import pandas as pd
from torch.utils.benchmark import Timer
from typing import List, Tuple, Dict

import pandas as pd
from timeit import Timer

# --------------------------------------------------------------------------------
# 1. Introduction to Memory Management
# --------------------------------------------------------------------------------

print("="*60)
print("1. Introduction to Memory Management")
print("="*60)

print("""
Memory management in PyTorch involves understanding:
1. Storage Types: CPU vs GPU memory
2. Tensor Data Types: Impact on memory and speed
3. Memory Allocation Patterns
4. Memory Leaks and Prevention
5. Performance Optimization Techniques
""")


print("\n" + "=" * 60)
print("2. Data Type Analysis")
print("=" * 60)

import torch
import pandas as pd
import matplotlib.pyplot as plt
from time import perf_counter

def analyze_datatypes():
    """
    Analyze memory usage and computation speed for various data types in PyTorch.
    """
    size = (1000, 1000)  # Matrix size
    
    float_types = {
        'float32': torch.float32,
        'float64': torch.float64,
        'float16': torch.float16,
    }
    int_types = {
        'int32': torch.int32,
        'int64': torch.int64,
        'bool': torch.bool
    }
    
    results = []

    # Analyze floating-point types
    for name, dtype in float_types.items():
        tensor = torch.randn(size, dtype=dtype)  # Random tensor
        
        # Measure memory usage
        memory = tensor.element_size() * tensor.nelement() / (1024 * 1024)  # Memory in MB
        
        # Measure computation time
        start_time = perf_counter()
        for _ in range(100):  # Perform 100 iterations
            torch.matmul(tensor, tensor)
        end_time = perf_counter()
        
        speed = (end_time - start_time) / 100  # Average time per operation
        
        results.append({
            'dtype': name,
            'memory_mb': memory,
            'matmul_time': speed
        })

    # Analyze integer types
    for name, dtype in int_types.items():
        if dtype == torch.bool:
            tensor = torch.randint(0, 2, size, dtype=dtype)  # Boolean tensor
        else:
            tensor = torch.randint(0, 100, size, dtype=dtype)  # Integer tensor
        
        # Measure memory usage
        memory = tensor.element_size() * tensor.nelement() / (1024 * 1024)  # Memory in MB
        
        # Measure computation time (convert to float for matmul)
        start_time = perf_counter()
        for _ in range(100):  # Perform 100 iterations
            torch.matmul(tensor.float(), tensor.float())
        end_time = perf_counter()
        
        speed = (end_time - start_time) / 100  # Average time per operation
        
        results.append({
            'dtype': name,
            'memory_mb': memory,
            'matmul_time': speed
        })
    
    # Convert results to DataFrame for display
    df = pd.DataFrame(results)
    print("\nData Type Analysis Results:")
    print(df)
    
    # Visualize the results
    plt.figure(figsize=(12, 6))

    # Memory usage plot
    plt.subplot(1, 2, 1)
    plt.bar(df['dtype'], df['memory_mb'])
    plt.title('Memory Usage by Data Type')
    plt.ylabel('Memory (MB)')
    plt.xlabel('Data Type')
    plt.xticks(rotation=45)

    # Computation time plot
    plt.subplot(1, 2, 2)
    plt.bar(df['dtype'], df['matmul_time'])
    plt.title('Matrix Multiplication Time by Data Type')
    plt.ylabel('Time (seconds)')
    plt.xlabel('Data Type')
    plt.xticks(rotation=45)

    # Save and show plots
    plt.tight_layout()
    plt.savefig('datatype_analysis_fixed.png')
    print("\nVisualization saved as 'datatype_analysis_fixed.png'.")
    plt.show()

# Call the function
analyze_datatypes()



# --------------------------------------------------------------------------------
# 3. Memory Allocation Patterns
# --------------------------------------------------------------------------------

print("\n"+"="*60)
print("3. Memory Allocation Patterns")
print("="*60)

print("""
Memory Allocation Patterns:

1. Continuous vs Fragmented Memory:
   - Continuous: Single large block of memory
   - Fragmented: Multiple small blocks scattered in memory
   
   Benefits of Continuous:
   - Faster memory access (better cache utilization)
   - Less memory overhead
   - Better hardware optimization
   
   Drawbacks of Fragmented:
   - Cache misses
   - Memory overhead from bookkeeping
   - Slower access patterns
""")

@profile
def demonstrate_allocation_patterns():
    # 1. Continuous vs Fragmented Memory
    def continuous_allocation():
        # Allocates one large 1000x1000 tensor
        # Benefits: Single memory block, better cache utilization
        return torch.randn(1000, 1000)
    
    def fragmented_allocation():
        # Allocates 1000 small 32x32 tensors
        # Drawbacks: Memory fragmentation, multiple allocations, worse cache performance
        tensors = []
        for _ in range(1000):
            tensors.append(torch.randn(32, 32))
        return tensors
    
    # 2. In-place vs New Memory Operations
    def inplace_ops():
        # In-place operation: Modifies tensor directly
        # Benefits: No new memory allocation, memory efficient
        x = torch.randn(1000, 1000)
        x.add_(1)  # In-place addition (note the underscore)
        return x
    
    def new_memory_ops():
        # Creates new tensor for result
        # Drawbacks: Allocates new memory, keeps old tensor in memory
        x = torch.randn(1000, 1000)
        x = x + 1  # Creates new tensor
        return x
    
    # Execute and measure
    print("\nMemory Usage Comparison:")
    x1 = continuous_allocation()
    print(f"Continuous allocation size: {x1.element_size() * x1.nelement() / 1024**2:.2f} MB")
    
    x2 = fragmented_allocation()
    total_fragmented = sum(t.element_size() * t.nelement() for t in x2) / 1024**2
    print(f"Fragmented allocation size: {total_fragmented:.2f} MB")
    
    x3 = inplace_ops()
    x4 = new_memory_ops()
    
    return x1, x2, x3, x4

demonstrate_allocation_patterns()

# --------------------------------------------------------------------------------
# 4. Memory Leaks and Prevention
# --------------------------------------------------------------------------------

print("\n"+"="*60)
print("4. Memory Leaks and Prevention")
print("="*60)

print("""
Memory Leak Prevention:

1. Memory Leaks in PyTorch:
   - Happen when tensors are not properly deallocated
   - Common in loops and long-running processes
   - Can cause OOM (Out of Memory) errors
   
2. Prevention Techniques:
   - Explicit deletion with 'del'
   - Using context managers
   - Proper scope management
   - Regular memory monitoring
""")

class MemoryTracker:
    def __init__(self):
        self.memory_samples = []
    
    def sample(self):
        process = psutil.Process()
        memory = process.memory_info().rss / (1024 * 1024)  # MB
        self.memory_samples.append(memory)
        return memory
    
    def plot(self):
        plt.figure(figsize=(10, 5))
        plt.plot(self.memory_samples)
        plt.title('Memory Usage Over Time')
        plt.ylabel('Memory (MB)')
        plt.xlabel('Sample')
        plt.savefig('memory_tracking.png')
        plt.close()

def demonstrate_memory_leaks():
    tracker = MemoryTracker()
    
    # 1. Bad Practice: Memory Leak
    def leaky_function():
        # This function leaks memory because:
        # - Keeps appending tensors to list
        # - Never releases old tensors
        # - List keeps growing indefinitely
        tensors = []
        for _ in range(1000):
            tensors.append(torch.randn(100, 100))  # Each tensor is ~40KB
            tracker.sample()
            print(f"Current memory usage: {tracker.memory_samples[-1]:.2f} MB")
    
    # 2. Good Practice: Proper Cleanup
    def clean_function():
        # This function manages memory properly:
        # - Creates one tensor at a time
        # - Explicitly deletes old tensor
        # - Memory usage stays constant
        for _ in range(1000):
            x = torch.randn(100, 100)
            del x  # Explicit cleanup
            tracker.sample()
            print(f"Current memory usage: {tracker.memory_samples[-1]:.2f} MB")
    
    print("\nMemory Leak Demonstration:")
    print("Running leaky function...")
    leaky_function()
    
    print("Running clean function...")
    clean_function()
    
    tracker.plot()

demonstrate_memory_leaks()

# --------------------------------------------------------------------------------
# 5. Advanced Memory Optimization (PhD Level)
# --------------------------------------------------------------------------------

print("\n"+"="*60)
print("5. Advanced Memory Optimization")
print("="*60)

print("""
Pinned Memory in PyTorch:

What is Pinned Memory?
- Special CPU memory that can't be paged out to disk
- Locked in RAM for faster CPU-GPU transfers
- Direct memory access (DMA) capable

Benefits:
1. Faster CPU to GPU transfers:
   - Up to 2-3x speedup for large tensors
   - Enables asynchronous data transfer
   - Better GPU utilization

2. Use Cases:
   - DataLoader with GPU training
   - Real-time applications
   - High-performance computing

Drawbacks:
1. Limited by RAM size
2. Can't be swapped to disk
3. May impact system performance if overused
""")

class TensorPool:
    def __init__(self, max_size: int = 100):
        self.pool: Dict[Tuple[int, ...], List[torch.Tensor]] = {}
        self.max_size = max_size
    
    def get(self, shape: Tuple[int, ...]) -> torch.Tensor:
        if shape in self.pool and self.pool[shape]:
            return self.pool[shape].pop()
        return torch.empty(shape)
    
    def put(self, tensor: torch.Tensor):
        shape = tuple(tensor.shape)
        if shape not in self.pool:
            self.pool[shape] = []
        if len(self.pool[shape]) < self.max_size:
            self.pool[shape].append(tensor)

def demonstrate_advanced_optimization():
    # 1. Tensor Pooling
    pool = TensorPool()
    
    def with_pooling():
        for _ in range(1000):
            x = pool.get((100, 100))
            y = torch.randn(100, 100)
            z = x @ y
            pool.put(x)
            pool.put(z)
    
    def without_pooling():
        for _ in range(1000):
            x = torch.empty(100, 100)
            y = torch.randn(100, 100)
            z = x @ y
    
    # 2. Memory Pinning
    def with_pinned_memory():
        x = torch.randn(1000, 1000).pin_memory()
        return x.to('cuda') if torch.cuda.is_available() else x
    
    def without_pinned_memory():
        x = torch.randn(1000, 1000)
        return x.to('cuda') if torch.cuda.is_available() else x
    
    # Measure execution times
    timer = Timer(stmt='func()', globals={'func': with_pooling})
    pooling_time = timer.timeit(100).mean
    
    timer = Timer(stmt='func()', globals={'func': without_pooling})
    no_pooling_time = timer.timeit(100).mean
    
    print("\nAdvanced Optimization Results:")
    print(f"With pooling: {pooling_time:.4f}s")
    print(f"Without pooling: {no_pooling_time:.4f}s")

demonstrate_advanced_optimization()

# --------------------------------------------------------------------------------
# 6. Best Practices Summary
# --------------------------------------------------------------------------------

print("\n"+"="*60)
print("6. Best Practices Summary")
print("="*60)

print("""
Memory Management Best Practices:

1. Data Types:
   - Use appropriate dtypes (float32 vs float64)
   - Consider mixed precision training
   - Monitor memory-speed tradeoffs

2. Allocation:
   - Prefer contiguous memory allocation
   - Reuse tensors when possible
   - Use in-place operations when appropriate
   - Implement tensor pooling for frequent allocations

3. Memory Leaks:
   - Clear unused tensors with del
   - Use context managers (with torch.no_grad())
   - Monitor memory usage
   - Profile code regularly

4. Advanced Techniques:
   - Use gradient checkpointing for large models
   - Implement custom memory pools
   - Consider quantization
   - Use memory pinning for CPU-GPU transfers

5. Tools:
   - memory_profiler
   - torch.cuda.memory_summary()
   - psutil
   - PyTorch profiler
""")

# --------------------------------------------------------------------------------
# 7. Performance Benchmarking
# --------------------------------------------------------------------------------

print("\n"+"="*60)
print("7. Performance Benchmarking")
print("="*60)

def run_benchmarks():
    sizes = [128, 256, 512, 1024, 2048]
    results = []
    
    for size in sizes:
        # Create tensors
        a = torch.randn(size, size)
        b = torch.randn(size, size)
        
        # Test different operations
        ops = {
            'matmul': lambda: torch.matmul(a, b),
            'element_wise': lambda: a * b,
            'transpose': lambda: a.t(),
            'sum': lambda: torch.sum(a)
        }
        
        for op_name, op in ops.items():
            timer = Timer(stmt='op()', globals={'op': op})
            time = timer.timeit(100).mean
            memory = a.element_size() * a.nelement() / (1024 * 1024)  # MB
            
            results.append({
                'size': size,
                'operation': op_name,
                'time': time,
                'memory': memory
            })
    
    df = pd.DataFrame(results)
    print("\nPerformance Benchmarks:")
    print(df)
    
    # Visualize results
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 2, 1)
    for op in df['operation'].unique():
        data = df[df['operation'] == op]
        plt.plot(data['size'], data['time'], label=op, marker='o')
    plt.title('Operation Time vs Matrix Size')
    plt.xlabel('Matrix Size')
    plt.ylabel('Time (seconds)')
    plt.legend()
    plt.yscale('log')
    
    plt.subplot(1, 2, 2)
    plt.plot(df['size'].unique(), df[df['operation'] == 'matmul']['memory'].unique(), marker='o')
    plt.title('Memory Usage vs Matrix Size')
    plt.xlabel('Matrix Size')
    plt.ylabel('Memory (MB)')
    
    plt.tight_layout()
    plt.savefig('performance_benchmarks.png')
    plt.close()

run_benchmarks() 