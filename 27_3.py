import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.autograd import Variable
import networkx as nx
import time 
# --------------------------------------------------------------------------------
# 1. Introduction to Dynamic Computation Graphs
# --------------------------------------------------------------------------------

print("="*60)
print("1. Introduction to Dynamic Computation Graphs")
print("="*60)

print("""
Dynamic Computation Graphs (DCGs) are a key feature of PyTorch that allows the neural network
architecture to change during runtime. Unlike static graphs (used in TensorFlow 1.x), 
DCGs are built on-the-fly during the forward pass.

Key Concepts:
- Graph Construction: Built dynamically during forward pass
- Memory Efficiency: Only stores active computations
- Flexibility: Can change architecture based on input
- Debugging: Easier to debug as Python's native tools work
""")

# --------------------------------------------------------------------------------
# 2. Static vs Dynamic Graphs
# --------------------------------------------------------------------------------

print("\n"+"="*60)
print("2. Static vs Dynamic Graphs")
print("="*60)

def demonstrate_dynamic_nature():
    # Example 1: Control flow affecting computation
    def dynamic_network(x, condition):
        if condition:
            return x ** 2
        else:
            return x ** 3
    
    x = torch.tensor([2.0], requires_grad=True)
    
    # Same input, different computations
    y1 = dynamic_network(x, True)
    y2 = dynamic_network(x, False)
    
    print("Example of Dynamic Behavior:")
    print(f"x = {x.item()}")
    print(f"x² = {y1.item()}")
    print(f"x³ = {y2.item()}")
    
    # Visualize computation graphs
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    visualize_graph(y1, "x²")
    plt.subplot(1, 2, 2)
    visualize_graph(y2, "x³")
    plt.savefig('dynamic_graphs.png')
    plt.close()

def visualize_graph(output, title):
    G = nx.DiGraph()
    seen = set()
    
    def add_nodes(var):
        if var not in seen:
            if torch.is_tensor(var):
                G.add_node(str(id(var)), label=str(var.size()))
            else:
                G.add_node(str(id(var)), label=str(type(var).__name__))
            seen.add(var)
            
            if hasattr(var, 'next_functions'):
                for u in var.next_functions:
                    if u[0] is not None:
                        add_nodes(u[0])
                        G.add_edge(str(id(u[0])), str(id(var)))
    
    add_nodes(output.grad_fn)
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_color='lightblue', 
            node_size=1500, arrowsize=20)
    plt.title(title)

demonstrate_dynamic_nature()

# --------------------------------------------------------------------------------
# 3. Control Flow in Dynamic Graphs
# --------------------------------------------------------------------------------

print("\n"+"="*60)
print("3. Control Flow in Dynamic Graphs")
print("="*60)

class DynamicNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(10, 10)
    
    def forward(self, x, iterations):
        for _ in range(iterations):
            x = torch.tanh(self.hidden(x))
        return x

def demonstrate_control_flow():
    model = DynamicNet()
    x = torch.randn(1, 10)
    
    # Same model, different number of iterations
    y1 = model(x, iterations=1)
    y2 = model(x, iterations=3)
    
    print("Network output shapes:")
    print(f"1 iteration: {y1.shape}")
    print(f"3 iterations: {y2.shape}")
    print("\nNote: Same model creates different computation graphs!")

demonstrate_control_flow()

# --------------------------------------------------------------------------------
# 4. Memory Management in Dynamic Graphs
# --------------------------------------------------------------------------------

print("\n"+"="*60)
print("4. Memory Management in Dynamic Graphs")
print("="*60)

def demonstrate_memory_management():
    def measure_memory(func):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        result = func()
        memory_used = torch.cuda.max_memory_allocated() / 1024**2
        return result, memory_used
    
    # Function with retained computation graph
    def with_retention():
        x = torch.randn(1000, 1000, requires_grad=True)
        for _ in range(10):
            x = x @ x
        return x
    
    # Function with manual graph clearing
    def with_clearing():
        x = torch.randn(1000, 1000, requires_grad=True)
        for _ in range(10):
            x = x @ x
            # Clear intermediate computations
            if not torch.is_grad_enabled():
                del x.grad_fn
        return x
    
    if torch.cuda.is_available():
        _, mem1 = measure_memory(with_retention)
        _, mem2 = measure_memory(with_clearing)
        print(f"Memory used with retention: {mem1:.2f} MB")
        print(f"Memory used with clearing: {mem2:.2f} MB")

demonstrate_memory_management()

# --------------------------------------------------------------------------------
# 5. Advanced Dynamic Architectures (PhD Level)
# --------------------------------------------------------------------------------

print("\n"+"="*60)
print("5. Advanced Dynamic Architectures")
print("="*60)

class DynamicTreeLSTM(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.ioux = nn.Linear(input_size, 3 * hidden_size)
        self.iouh = nn.Linear(hidden_size, 3 * hidden_size)
        self.fx = nn.Linear(input_size, hidden_size)
        self.fh = nn.Linear(hidden_size, hidden_size)
    
    def node_forward(self, input, child_c, child_h):
        child_h_sum = torch.sum(child_h, dim=0)
        
        iou = self.ioux(input) + self.iouh(child_h_sum)
        i, o, u = torch.split(iou, self.hidden_size, dim=1)
        i, o, u = torch.sigmoid(i), torch.sigmoid(o), torch.tanh(u)
        
        f = torch.sigmoid(
            self.fx(input).repeat(len(child_h), 1) +
            self.fh(child_h)
        )
        
        fc = torch.sum(f * child_c, dim=0)
        c = i * u + fc
        h = o * torch.tanh(c)
        
        return c, h

print("""
Advanced Concepts in Dynamic Graphs:

1. Tree-Structured Networks:
   - Dynamically process hierarchical data
   - Variable number of children per node
   - Adaptive computation paths

2. Memory Optimization:
   - Gradient checkpointing
   - Just-in-time compilation
   - Dynamic buffer management

3. Research Applications:
   - Graph Neural Networks
   - Neural Program Synthesis
   - Dynamic Architecture Search
""")

# --------------------------------------------------------------------------------
# 6. Best Practices and Pitfalls
# --------------------------------------------------------------------------------

print("\n"+"="*60)
print("6. Best Practices and Pitfalls")
print("="*60)

print("""
Best Practices:
1. Clear unused variables to free memory
2. Use torch.no_grad() for inference
3. Batch similar computations together
4. Profile memory usage regularly

Common Pitfalls:
1. Memory leaks from retained graphs
2. Unnecessary gradient computation
3. Inefficient dynamic batching
4. Complex control flow impacting performance
""")

def demonstrate_best_practices():
    # 1. Memory efficiency
    x = torch.randn(100, 100, requires_grad=True)
    
    # Bad practice
    def inefficient():
        y = x
        for _ in range(10):
            y = torch.relu(y)
        return y
    
    # Good practice
    def efficient():
        with torch.no_grad():
            y = x
            for _ in range(10):
                y = torch.relu(y)
        return y
    
    # Compare execution
    start = time.time()
    inefficient()
    time1 = time.time() - start
    
    start = time.time()
    efficient()
    time2 = time.time() - start
    
    print(f"Inefficient execution time: {time1:.4f}s")
    print(f"Efficient execution time: {time2:.4f}s")

demonstrate_best_practices() 