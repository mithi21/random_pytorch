import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import networkx as nx
from graphviz import Digraph
from torch.autograd import Function, Variable, grad
import time
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from torch.utils.checkpoint import checkpoint


# --------------------------------------------------------------------------------
# 1. Introduction to Computation Graphs
# --------------------------------------------------------------------------------

print("="*60)
print("1. Introduction to Computation Graphs")
print("="*60)

print("""
Computation Graphs in Deep Learning:
1. Definition: Directed acyclic graphs (DAGs) representing computations
2. Components:
   - Nodes: Operations (add, multiply, activation functions)
   - Edges: Tensor data flow
   - Leaves: Input tensors
   - Roots: Output tensors
3. Properties:
   - Automatic differentiation
   - Lazy evaluation
   - Dynamic nature in PyTorch
""")

# --------------------------------------------------------------------------------
# 2. Graph Construction Visualization
# --------------------------------------------------------------------------------

class ComputationGraphVisualizer:
    def __init__(self):
        self.nodes = []
        self.edges = []
        self.node_attrs = {}
    
    def trace_tensor(self, tensor: torch.Tensor, name: str = "output"):
        """Traces a tensor's computation graph."""
        def add_node(node, name: str):
            if node not in self.nodes:
                self.nodes.append(node)
                if hasattr(node, 'variable'):
                    self.node_attrs[node] = f"{name}\n{tuple(node.variable.shape)}"
                else:
                    self.node_attrs[node] = f"{name}\n{type(node).__name__}"
        
        def add_parents(node):
            if hasattr(node, 'next_functions'):
                for parent in node.next_functions:
                    if parent[0] is not None:
                        self.edges.append((parent[0], node))
                        add_node(parent[0], "intermediate")
                        add_parents(parent[0])
        
        if tensor.grad_fn is not None:
            add_node(tensor.grad_fn, name)
            add_parents(tensor.grad_fn)
    
    def visualize(self, filename: str = "computation_graph"):
        """Creates a visual representation of the graph."""
        dot = Digraph()
        dot.attr(rankdir='LR')  # Left to right layout
        
        # Add nodes
        for node in self.nodes:
            dot.node(str(id(node)), self.node_attrs[node])
        
        # Add edges
        for src, dst in self.edges:
            dot.edge(str(id(src)), str(id(dst)))
        
        dot.render(filename, format='png', cleanup=True)

def demonstrate_graph_construction():
    # Simple computation
    x = torch.tensor([2.0], requires_grad=True)
    y = torch.tensor([3.0], requires_grad=True)
    
    # Build computation graph
    z = x * y
    w = torch.sin(z)
    v = torch.exp(w)
    f = v * x
    
    # Visualize graph
    visualizer = ComputationGraphVisualizer()
    visualizer.trace_tensor(f, "final_output")
    visualizer.visualize("basic_computation")
    
    print("""
    Graph Construction Steps:
    1. x * y → Creates multiplication node
    2. sin(z) → Adds sine operation
    3. exp(w) → Adds exponential
    4. v * x → Connects back to input
    
    Note how each operation creates a new node and edges.
    """)

demonstrate_graph_construction()

# --------------------------------------------------------------------------------
# 3. Dynamic Graph Construction
# --------------------------------------------------------------------------------

class DynamicGraphExample(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 10)
    
    def forward(self, x: torch.Tensor, depth: int) -> torch.Tensor:
        """Demonstrates dynamic graph construction based on input."""
        operations = []
        
        for i in range(depth):
            x = self.linear(x)
            if i % 2 == 0:
                x = torch.relu(x)
                operations.append('ReLU')
            else:
                x = torch.tanh(x)
                operations.append('Tanh')
        
        return x, operations

def demonstrate_dynamic_graphs():
    model = DynamicGraphExample()
    x = torch.randn(1, 10)
    
    # Create graphs of different depths
    outputs = []
    graphs = []
    
    for depth in [2, 4]:
        y, ops = model(x, depth)
        visualizer = ComputationGraphVisualizer()
        visualizer.trace_tensor(y, f"depth_{depth}")
        visualizer.visualize(f"dynamic_graph_depth_{depth}")
        
        outputs.append(y)
        graphs.append(ops)
    
    print("\nDynamic Graph Construction:")
    for depth, ops in enumerate(graphs, 2):
        print(f"\nDepth {depth} operations: {' → '.join(ops)}")

demonstrate_dynamic_graphs()

# --------------------------------------------------------------------------------
# 4. Graph Execution and Memory Management
# --------------------------------------------------------------------------------

@dataclass
class MemoryStats:
    allocated: float
    cached: float
    reserved: float

class GraphExecutionProfiler:
    def __init__(self):
        self.memory_stats = []
        self.execution_times = []
    
    def measure_memory(self) -> MemoryStats:
        if torch.cuda.is_available():
            return MemoryStats(
                allocated=torch.cuda.memory_allocated() / 1024**2,
                cached=torch.cuda.memory_cached() / 1024**2,
                reserved=torch.cuda.memory_reserved() / 1024**2
            )
        return MemoryStats(0, 0, 0)
    
    def profile_execution(self, func, *args, **kwargs):
        start_mem = self.measure_memory()
        start_time = time.time()
        
        result = func(*args, **kwargs)
        
        end_time = time.time()
        end_mem = self.measure_memory()
        
        self.memory_stats.append((start_mem, end_mem))
        self.execution_times.append(end_time - start_time)
        
        return result
    
    def report(self):
        print("\nExecution Profile:")
        for i, ((start_mem, end_mem), exec_time) in enumerate(zip(self.memory_stats, self.execution_times)):
            print(f"\nIteration {i+1}:")
            print(f"Execution time: {exec_time*1000:.2f}ms")
            print(f"Memory allocated: {end_mem.allocated - start_mem.allocated:.2f}MB")
            print(f"Memory cached: {end_mem.cached - start_mem.cached:.2f}MB")

def demonstrate_graph_execution():
    profiler = GraphExecutionProfiler()
    
    def complex_computation(x: torch.Tensor, depth: int) -> torch.Tensor:
        for _ in range(depth):
            x = torch.relu(torch.matmul(x, x))
        return x
    
    # Profile different graph sizes
    sizes = [100, 200, 400]
    for size in sizes:
        x = torch.randn(size, size, requires_grad=True)
        profiler.profile_execution(complex_computation, x, depth=3)
    
    profiler.report()

demonstrate_graph_execution()

# --------------------------------------------------------------------------------
# 5. Advanced Topics (PhD Level)
# --------------------------------------------------------------------------------

print("\n"+"="*60)
print("5. Advanced Graph Optimization")
print("="*60)

class CustomFunction(Function):
    """Example of custom autograd function with optimized backward pass."""
    
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

def demonstrate_advanced_features():
    # 1. Custom autograd function
    custom_relu = CustomFunction.apply
    x = torch.randn(1000, 1000, requires_grad=True)
    y = custom_relu(x)
    
    # 2. Graph optimization techniques
    def with_checkpointing():
        return checkpoint(
            lambda x: torch.relu(torch.matmul(x, x)),
            x
        )

    
    def without_checkpointing():
        return torch.relu(torch.matmul(x, x))
    
    # Profile both approaches
    profiler = GraphExecutionProfiler()
    y1 = profiler.profile_execution(with_checkpointing)
    y2 = profiler.profile_execution(without_checkpointing)
    
    print("""
    Advanced Graph Optimization Techniques:
    
    1. Checkpointing:
       - Trades computation for memory
       - Useful for very deep networks
       - Saves intermediate activations
    
    2. Custom Autograd Functions:
       - Optimized forward/backward passes
       - Memory efficient implementations
       - Custom gradient computation
    
    3. Graph Transformations:
       - Operation fusion
       - Dead code elimination
       - Constant folding
    """)

demonstrate_advanced_features()

# --------------------------------------------------------------------------------
# 6. Best Practices and Pitfalls
# --------------------------------------------------------------------------------

print("\n"+"="*60)
print("6. Best Practices and Pitfalls")
print("="*60)

print("""
Best Practices:
1. Graph Construction:
   - Keep graphs shallow when possible
   - Avoid unnecessary branches
   - Use inplace operations when safe
   
2. Memory Management:
   - Clear unused variables
   - Use checkpointing for deep graphs
   - Monitor memory usage
   
3. Performance:
   - Batch similar operations
   - Minimize graph reconstructions
   - Use profiling tools

Common Pitfalls:
1. Memory Leaks:
   - Retaining references to intermediate tensors
   - Not clearing gradients
   - Accumulating graphs in loops

2. Performance Issues:
   - Excessive graph rebuilding
   - Unoptimized custom functions
   - Poor memory access patterns
""") 