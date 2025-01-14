import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
import math
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import tiktoken
from collections import defaultdict
import time

# --------------------------------------------------------------------------------
# 1. Introduction to Transformer Architecture
# --------------------------------------------------------------------------------

print("="*60)
print("1. Transformer Architecture Components")
print("="*60)

print("""
Key Components of a Decoder-Only Transformer:
1. Tokenization (BPE)
2. Token & Positional Embeddings (RoPE)
3. Self-Attention Mechanism
4. Feed-Forward Networks
5. Layer Normalization
6. Weight Sharing (Input/Output Embeddings)
7. KQV Caching
""")

# --------------------------------------------------------------------------------
# 2. Byte-Pair Encoding (BPE) Implementation
# --------------------------------------------------------------------------------

from collections import defaultdict
from tqdm import tqdm

class SimpleBPE:
    """Simple BPE tokenizer implementation."""
    
    def __init__(self, vocab_size: int = 256):
        self.vocab_size = vocab_size
        self.vocab = {}  # token -> id
        self.merges = {}  # pair -> new_token
        
    def train(self, text: str, num_merges: int = 1000):
        """Train BPE on text."""
        print(f"\nTraining BPE with vocab_size={self.vocab_size}, num_merges={num_merges}")
        
        words = text.split()
        char_freqs = defaultdict(int)
        
        for word in tqdm(words, desc="Initializing Character Frequencies"):
            for char in word:
                char_freqs[char] += 1
        
        self.vocab = {char: i for i, char in enumerate(char_freqs.keys())}
        print(f"Initial Vocabulary Size: {len(self.vocab)}")
        
        for i in tqdm(range(num_merges), desc="Performing Merges"):
            if len(self.vocab) >= self.vocab_size:
                print(f"Reached max vocab size ({self.vocab_size}) at iteration {i}.")
                break
            
            pair_freqs = defaultdict(int)
            for word in words:
                chars = list(word)
                for j in range(len(chars)-1):
                    pair = (chars[j], chars[j+1])
                    pair_freqs[pair] += 1
            
            if not pair_freqs:
                print(f"No pairs left to merge at iteration {i}.")
                break
            
            best_pair = max(pair_freqs.items(), key=lambda x: x[1])[0]
            new_token = ''.join(best_pair)
            self.vocab[new_token] = len(self.vocab)
            self.merges[best_pair] = new_token
            
            new_words = []
            for word in words:
                new_word = word
                for pair, merged in self.merges.items():
                    new_word = new_word.replace(''.join(pair), merged)
                new_words.append(new_word)
            words = new_words
        
        print("\nTraining Complete!")
        print(f"Final Vocabulary Size: {len(self.vocab)}")
        print(f"Number of Merges: {len(self.merges)}")

def demonstrate_bpe():
    print("\nDemonstrating BPE Tokenization:")
    
    # Simple example text
    text = "The quick brown fox jumps over the lazy dog"
    tokenizer = SimpleBPE(vocab_size=50)
    tokenizer.train(text, num_merges=10)
    
    print("Vocabulary:")
    for token, idx in sorted(tokenizer.vocab.items(), key=lambda x: x[1]):
        print(f"Token: {token:>4} -> ID: {idx}")
    
    print("\nMerges:")
    for (a, b), merged in tokenizer.merges.items():
        print(f"({a}, {b}) -> {merged}")

demonstrate_bpe()

# --------------------------------------------------------------------------------
# 3. Rotary Position Embeddings (RoPE)
# --------------------------------------------------------------------------------

class RotaryEmbedding(nn.Module):
    """
    Implements Rotary Position Embeddings (RoPE).
    RoPE applies rotation to key and query vectors based on their positions.
    """
    
    def __init__(self, dim: int, max_seq_len: int = 2048):
        super().__init__()
        self.dim = dim
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        position = torch.arange(max_seq_len)
        sincos = torch.einsum('i,j->ij', position, inv_freq)
        self.register_buffer('sin', sincos.sin())
        self.register_buffer('cos', sincos.cos())
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply rotary embeddings to input tensor."""
        # Handle input with shape [B, L, num_heads, head_dim]
        if x.ndim == 4:
            B, L, num_heads, head_dim = x.shape
            assert head_dim % 2 == 0, "Head dimension must be even for RoPE"
            
            # Reshape input for rotation
            x_reshape = x.view(B, L, num_heads, -1, 2)  # [B, L, num_heads, head_dim//2, 2]
            
            # Prepare sin and cos for broadcasting
            sin = self.sin[:L, None, None, :].expand(L, B, num_heads, head_dim // 2).permute(1, 0, 2, 3)
            cos = self.cos[:L, None, None, :].expand(L, B, num_heads, head_dim // 2).permute(1, 0, 2, 3)
            
            x1 = x_reshape[..., 0]  # Even indices
            x2 = x_reshape[..., 1]  # Odd indices
            
            # Apply RoPE rotation
            rotated = torch.stack([
                x1 * cos - x2 * sin,
                x1 * sin + x2 * cos
            ], dim=-1)
            
            return rotated.flatten(-2)  # Restore shape to [B, L, num_heads, head_dim]
        
        # Handle input with shape [B, L, D]
        elif x.ndim == 3:
            B, L, D = x.shape
            assert D % 2 == 0, "Embedding dimension must be even for RoPE"
            
            # Reshape input for rotation
            x_reshape = x.view(B, L, -1, 2)  # [B, L, D//2, 2]
            
            # Prepare sin and cos for broadcasting
            sin = self.sin[:L, None, :].expand(L, B, D // 2).permute(1, 0, 2)  # [B, L, D/2]
            cos = self.cos[:L, None, :].expand(L, B, D // 2).permute(1, 0, 2)  # [B, L, D/2]
            
            x1 = x_reshape[..., 0]  # Even indices
            x2 = x_reshape[..., 1]  # Odd indices
            
            # Apply RoPE rotation
            rotated = torch.stack([
                x1 * cos - x2 * sin,
                x1 * sin + x2 * cos
            ], dim=-1)
            
            return rotated.flatten(-2)  # Restore shape to [B, L, D]
        
        else:
            raise ValueError(f"Unsupported input shape: {x.shape}")

def visualize_rope_embeddings():
    """Visualize how RoPE affects vector representations."""
    dim = 64
    seq_len = 10
    rope = RotaryEmbedding(dim)
    
    # Create sample vectors
    x = torch.randn(1, seq_len, dim)  # [1, 10, 64]
    x_rotated = rope(x)  # [1, 10, 64]
    
    # Reshape tensors for visualization
    x_vis = x[0].detach().numpy()  # [10, 64]
    x_rotated_vis = x_rotated[0].detach().numpy()  # [10, 64]
    
    # Option 1: Aggregate across embedding dimensions
    x_vis_agg = x_vis.mean(axis=-1)  # [10] (mean along embedding dimensions)
    x_rotated_vis_agg = x_rotated_vis.mean(axis=-1)  # [10]
    
    # Option 2: Select specific embedding dimensions for visualization
    x_vis_slice = x_vis[:, :32]  # [10, 32]
    x_rotated_vis_slice = x_rotated_vis[:, :32]  # [10, 32]
    
    # Option 3: Combine dimensions into a 2D array
    x_vis_combined = x_vis.reshape(seq_len, -1)  # [10, 64]
    x_rotated_vis_combined = x_rotated_vis.reshape(seq_len, -1)  # [10, 64]
    
    # Create visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot original embeddings
    im1 = ax1.imshow(x_vis_combined, aspect='auto', cmap='viridis')
    ax1.set_title('Original Vectors')
    ax1.set_xlabel('Flattened Dimensions')
    ax1.set_ylabel('Sequence Position')
    plt.colorbar(im1, ax=ax1)
    
    # Plot rotated embeddings
    im2 = ax2.imshow(x_rotated_vis_combined, aspect='auto', cmap='viridis')
    ax2.set_title('Rotated Vectors (RoPE)')
    ax2.set_xlabel('Flattened Dimensions')
    ax2.set_ylabel('Sequence Position')
    plt.colorbar(im2, ax=ax2)
    
    plt.tight_layout()
    plt.savefig('rope_visualization.png')
    plt.show()

visualize_rope_embeddings()

# --------------------------------------------------------------------------------
# 4. Attention Mechanisms with KV Cache
# --------------------------------------------------------------------------------

class CachedAttention(nn.Module):
    """
    Implements attention mechanism with KV caching for efficient inference.
    """
    
    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        
        # Single matrix for Q, K, V projections
        self.qkv = nn.Linear(dim, 3 * dim)
        self.rope = RotaryEmbedding(self.head_dim)
        
        # Initialize KV cache
        self.cache_k = None
        self.cache_v = None
    
    def forward(
        self, 
        x: torch.Tensor,
        use_cache: bool = False,
        clear_cache: bool = False
    ) -> torch.Tensor:
        """Forward pass with optional KV caching."""
        B, L, D = x.shape
        
        # Clear cache if requested
        if clear_cache:
            self.cache_k = None
            self.cache_v = None
        
        # QKV projections
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        
        # Reshape for multi-head attention
        q = q.view(B, L, self.num_heads, self.head_dim)
        k = k.view(B, L, self.num_heads, self.head_dim)
        v = v.view(B, L, self.num_heads, self.head_dim)
        
        # Apply RoPE to queries and keys
        q = self.rope(q)
        k = self.rope(k)
        
        # Update KV cache
        if use_cache:
            if self.cache_k is None:
                self.cache_k = k
                self.cache_v = v
            else:
                self.cache_k = torch.cat([self.cache_k, k], dim=1)
                self.cache_v = torch.cat([self.cache_v, v], dim=1)
            k, v = self.cache_k, self.cache_v
        
        # Scaled dot-product attention
        scale = 1.0 / math.sqrt(self.head_dim)
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale
        attn = F.softmax(attn, dim=-1)
        
        # Combine heads
        out = torch.matmul(attn, v)
        out = out.reshape(B, L, D)
        
        return out

def demonstrate_kv_cache():
    print("\nDemonstrating KV Cache:")
    
    model = CachedAttention(dim=64, num_heads=4)
    
    # Generate sample sequence
    x = torch.randn(1, 10, 64)
    
    # Without cache
    start_time = time.time()
    out1 = model(x, use_cache=False)
    time1 = time.time() - start_time
    
    # With cache
    start_time = time.time()
    out2 = model(x, use_cache=True)
    time2 = time.time() - start_time
    
    print(f"Time without cache: {time1:.4f}s")
    print(f"Time with cache: {time2:.4f}s")
    print(f"Speedup: {time1/time2:.2f}x")

demonstrate_kv_cache()

# --------------------------------------------------------------------------------
# 5. Complete Decoder-Only Transformer
# --------------------------------------------------------------------------------

@dataclass
class TransformerConfig:
    vocab_size: int = 50257
    max_seq_len: int = 2048
    dim: int = 768
    num_layers: int = 12
    num_heads: int = 12
    dropout: float = 0.1

class DecoderOnlyTransformer(nn.Module):
    """
    Complete implementation of a decoder-only transformer.
    Features:
    - Weight sharing between input/output embeddings
    - RoPE positional embeddings
    - KV caching
    - Layer normalization
    """
    
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        
        # Token embeddings (shared with output layer)
        self.token_embedding = nn.Embedding(config.vocab_size, config.dim)
        
        # Attention layers
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                'attention': CachedAttention(config.dim, config.num_heads),
                'ln1': nn.LayerNorm(config.dim),
                'ffn': nn.Sequential(
                    nn.Linear(config.dim, 4 * config.dim),
                    nn.GELU(),
                    nn.Linear(4 * config.dim, config.dim)
                ),
                'ln2': nn.LayerNorm(config.dim)
            }) for _ in range(config.num_layers)
        ])
        
        # Linear output projection reuses token_embedding weights
        self.output_projection = nn.Linear(config.dim, config.vocab_size, bias=False)
        self.output_projection.weight = self.token_embedding.weight  # Shared weights
    
    def forward(
        self,
        input_ids: torch.Tensor,
        use_cache: bool = False
    ) -> torch.Tensor:
        """Forward pass with optional KV caching."""
        x = self.token_embedding(input_ids)
        
        for layer in self.layers:
            # Self-attention with residual
            residual = x
            x = layer['ln1'](x)
            x = layer['attention'](x, use_cache=use_cache)
            x = x + residual
            
            # FFN with residual
            residual = x
            x = layer['ln2'](x)
            x = layer['ffn'](x)
            x = x + residual
        
        # Project to vocabulary
        logits = self.output_projection(x)
        return logits

def demonstrate_transformer():
    print("\nDemonstrating Complete Transformer:")
    
    config = TransformerConfig(
        vocab_size=1000,
        max_seq_len=128,
        dim=256,
        num_layers=4,
        num_heads=8
    )
    
    model = DecoderOnlyTransformer(config)
    
    # Generate sample input
    input_ids = torch.randint(0, config.vocab_size, (1, 64))
    
    # Forward pass
    logits = model(input_ids)
    print(f"Input shape: {input_ids.shape}")
    print(f"Output shape: {logits.shape}")
    
    # Memory usage
    param_count = sum(p.numel() for p in model.parameters())
    print(f"\nModel Parameters: {param_count:,}")
    
    # Verify weight sharing
    shared_weights = torch.equal(
        model.token_embedding.weight,
        model.output_projection.weight
    )
    print(f"Weight sharing verified: {shared_weights}")

demonstrate_transformer()


# --------------------------------------------------------------------------------
# 6. Best Practices and Guidelines
# --------------------------------------------------------------------------------

print("\n"+"="*60)
print("6. Best Practices and Guidelines")
print("="*60)

print("""
Best Practices for Transformer Implementation:

1. Tokenization:
   - Use established BPE implementations (tiktoken, sentencepiece)
   - Consider vocabulary size trade-offs
   - Handle special tokens consistently

2. Position Embeddings:
   - RoPE provides better extrapolation
   - Consider sequence length limitations
   - Implement efficient rotations

3. Attention Mechanisms:
   - Use flash attention when available
   - Implement efficient KV caching
   - Consider attention patterns (sliding window, sparse)

4. Memory Optimization:
   - Share embedding weights
   - Use gradient checkpointing for training
   - Implement efficient inference with caching

5. Training Considerations:
   - Proper initialization is crucial
   - Monitor attention patterns
   - Use learning rate warmup
   - Implement proper masking

Common Pitfalls:
1. Incorrect attention masking
2. Memory issues with long sequences
3. Numerical instability in attention
4. Inefficient KV cache implementation
5. Poor tokenization choices
""")

# Add Shakespeare dataset handling
def load_shakespeare():
    """Load and preprocess Shakespeare text."""
    import requests
    from pathlib import Path
    
    # Download Shakespeare text if not exists
    file_path = Path('shakespeare.txt')
    if not file_path.exists():
        url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
        text = requests.get(url).text
        file_path.write_text(text)
    else:
        text = file_path.read_text()
    
    return text

from tqdm import tqdm

class ShakespeareDataset(torch.utils.data.Dataset):
    def __init__(self, text: str, seq_length: int, tokenizer: SimpleBPE):
        print("Initializing Shakespeare Dataset...")
        self.text = text
        self.seq_length = seq_length
        self.tokenizer = tokenizer
        
        # Tokenize entire text
        print("Tokenizing text...")
        self.tokens = self.tokenize_text(text)
        print(f"Total Tokens: {len(self.tokens)}")
    
    def tokenize_text(self, text: str) -> torch.Tensor:
        """Tokenize text using BPE tokenizer."""
        # Train tokenizer if not already trained
        if not self.tokenizer.vocab:
            print("Training tokenizer...")
            self.tokenizer.train(text, num_merges=1000)
            print("Tokenizer training complete.")
        
        # Tokenize text
        words = text.split()
        token_ids = []
        
        print("Converting words to token IDs...")
        for word in tqdm(words, desc="Tokenizing Words"):
            # Apply merges iteratively
            current = list(word)
            while len(current) > 1:
                for pair, merged in self.tokenizer.merges.items():
                    s = ''.join(current)
                    if ''.join(pair) in s:
                        s = s.replace(''.join(pair), merged)
                        current = list(s)
            
            # Convert to token ids
            for token in current:
                if token in self.tokenizer.vocab:
                    token_ids.append(self.tokenizer.vocab[token])
        
        print("Tokenization complete.")
        return torch.tensor(token_ids)
    
    def __len__(self):
        dataset_length = len(self.tokens) - self.seq_length
        print(f"Dataset Length: {dataset_length} sequences")
        return dataset_length
    
    def __getitem__(self, idx):
        x = self.tokens[idx:idx + self.seq_length]
        y = self.tokens[idx + 1:idx + self.seq_length + 1]
        return x, y

from tqdm import tqdm

from transformers import GPT2Tokenizer

def train_shakespeare_model(use_pretrained_tokenizer: bool = False):
    print("\nTraining on Shakespeare Dataset:")
    
    # Load data
    text = load_shakespeare()
    print(f"Dataset length: {len(text)} characters")
    
    if use_pretrained_tokenizer:
        # Use pretrained tokenizer from HuggingFace
        print("Using pretrained GPT2 tokenizer...")
        pretrained_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        
        # Add a pad_token if not present
        if pretrained_tokenizer.pad_token is None:
            print("Adding a padding token to the pretrained tokenizer...")
            pretrained_tokenizer.pad_token = pretrained_tokenizer.eos_token
        
        # Tokenize and create dataset
        def tokenize_with_pretrained(text, seq_length):
            encodings = pretrained_tokenizer(
                text,
                return_tensors="pt",
                truncation=False,  # Full tokenization without truncation
                padding=False      # Padding not needed here
            )
            tokens = encodings["input_ids"].squeeze()
            dataset_length = max(0, len(tokens) - seq_length)
            print(f"Pretrained Tokenizer Dataset Length: {dataset_length}")
            return tokens, dataset_length
        
        seq_length = 64
        tokens, dataset_length = tokenize_with_pretrained(text, seq_length)
        
        if dataset_length <= 0:
            raise ValueError("Dataset length is too small after tokenization. Check the input text or sequence length.")
        
        class ShakespearePretrainedDataset(torch.utils.data.Dataset):
            def __init__(self, tokens, seq_length):
                self.tokens = tokens
                self.seq_length = seq_length

            def __len__(self):
                return dataset_length

            def __getitem__(self, idx):
                x = self.tokens[idx:idx + self.seq_length]
                y = self.tokens[idx + 1:idx + self.seq_length + 1]
                return x, y

        dataset = ShakespearePretrainedDataset(tokens, seq_length)
    
    else:
        # Use custom BPE tokenizer
        print("Using custom BPE tokenizer...")
        tokenizer = SimpleBPE(vocab_size=2000)  # Larger vocab for Shakespeare
        dataset = ShakespeareDataset(text, seq_length=64, tokenizer=tokenizer)
    
    # Create data loader
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=128,
        shuffle=True,
        num_workers=0  # For Mac compatibility
    )
    
    # Initialize model
    vocab_size = pretrained_tokenizer.vocab_size if use_pretrained_tokenizer else len(tokenizer.vocab)
    config = TransformerConfig(
        vocab_size=vocab_size,
        max_seq_len=64,
        dim=256,
        num_layers=6,
        num_heads=8,
        dropout=0.1
    )
    
    model = DecoderOnlyTransformer(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    
    # Training loop
    num_epochs = 1
    total_steps = 0
    losses = []
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        epoch_loss = 0.0
        progress_bar = tqdm(enumerate(dataloader), total=len(dataloader), desc="Training", leave=True)
        
        for batch_idx, (x, y) in progress_bar:
            optimizer.zero_grad()
            
            # Forward pass
            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, config.vocab_size), y.view(-1))
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Logging
            losses.append(loss.item())
            epoch_loss += loss.item()
            progress_bar.set_postfix({
                "Batch": f"{batch_idx}/{len(dataloader)}",
                "Loss": f"{loss.item():.4f}"
            })
            
            total_steps += 1
        
        avg_epoch_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch + 1} Average Loss: {avg_epoch_loss:.4f}")
    
    # Plot training loss
    plt.figure(figsize=(10, 5))
    plt.plot(losses, label="Loss")
    plt.title('Training Loss')
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('training_loss.png')
    plt.close()
    
    print("\nTraining Complete. Generating Sample Text...")
    
    # Generate sample text
    def generate_text(model, tokenizer, start_text="The ", max_length=100):
        model.eval()
        if use_pretrained_tokenizer:
            tokens = pretrained_tokenizer(start_text, return_tensors="pt")["input_ids"]
        else:
            tokens = torch.tensor([tokenizer.vocab.get(c, 0) for c in start_text]).unsqueeze(0)
        
        with torch.no_grad():
            for _ in range(max_length):
                logits = model(tokens)
                next_token = torch.argmax(logits[0, -1]).item()
                tokens = torch.cat([tokens, torch.tensor([[next_token]])], dim=1)
        
        # Convert tokens back to text
        if use_pretrained_tokenizer:
            generated = pretrained_tokenizer.decode(tokens.squeeze(), skip_special_tokens=True)
        else:
            id_to_token = {v: k for k, v in tokenizer.vocab.items()}
            generated = ''.join(id_to_token.get(t.item(), '') for t in tokens[0])
        return generated
    
    # Generate and print sample text
    print("\nGenerated Sample:")
    print(generate_text(model, tokenizer if not use_pretrained_tokenizer else pretrained_tokenizer))
    
    return model, tokenizer

# Train the model
if __name__ == "__main__":
    use_pretrained = input("Do you want to use a pretrained tokenizer (yes/no)? ").strip().lower() == "yes"
    model, tokenizer = train_shakespeare_model(use_pretrained_tokenizer=use_pretrained)
