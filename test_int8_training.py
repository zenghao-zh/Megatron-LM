#!/usr/bin/env python
"""Quick test script to compare INT8 vs BF16 training."""

import os
os.environ['CUDA_DEVICE_MAX_CONNECTIONS'] = '1'

import torch
import torch.nn as nn
from megatron.core.quantization.int8_training import apply_int8_training, Int8MixedPrecisionTrainingConfig
from megatron.core.quantization.int8_training.int8_tensor import Int8MixedPrecisionTrainingLinearWeight
import megatron.core.quantization.int8_training.int8_tensor as int8_module

def main():
    torch.manual_seed(42)
    
    # 配置
    hidden_size = 960
    ffn_size = 2560
    num_heads = 15
    num_layers = 4
    vocab_size = 49152
    seq_len = 256
    batch_size = 4
    num_steps = 50
    lr = 3e-4
    
    print(f"Config: hidden={hidden_size}, ffn={ffn_size}, layers={num_layers}, vocab={vocab_size}")
    
    class MLP(nn.Module):
        def __init__(self, hidden, ffn):
            super().__init__()
            self.up = nn.Linear(hidden, ffn, bias=False)
            self.down = nn.Linear(ffn, hidden, bias=False)
            self.act = nn.SiLU()
        def forward(self, x):
            return self.down(self.act(self.up(x)))
    
    class TransformerLayer(nn.Module):
        def __init__(self, hidden, ffn, heads):
            super().__init__()
            self.qkv = nn.Linear(hidden, hidden * 3, bias=False)
            self.out = nn.Linear(hidden, hidden, bias=False)
            self.mlp = MLP(hidden, ffn)
            self.norm1 = nn.RMSNorm(hidden)
            self.norm2 = nn.RMSNorm(hidden)
            self.heads = heads
            self.head_dim = hidden // heads
        def forward(self, x):
            B, S, D = x.shape
            h = self.norm1(x)
            qkv = self.qkv(h).reshape(B, S, 3, self.heads, self.head_dim)
            q, k, v = qkv.unbind(2)
            q, k, v = q.transpose(1,2), k.transpose(1,2), v.transpose(1,2)
            attn = torch.nn.functional.scaled_dot_product_attention(q, k, v)
            attn = attn.transpose(1, 2).reshape(B, S, D)
            x = x + self.out(attn)
            return x + self.mlp(self.norm2(x))
    
    class GPT(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Embedding(vocab_size, hidden_size)
            self.layers = nn.ModuleList([TransformerLayer(hidden_size, ffn_size, num_heads) for _ in range(num_layers)])
            self.norm = nn.RMSNorm(hidden_size)
            self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        def forward(self, x):
            x = self.embed(x)
            for layer in self.layers:
                x = layer(x)
            return self.lm_head(self.norm(x))
    
    # Create models
    model_bf16 = GPT().cuda().bfloat16()
    model_int8 = GPT().cuda().bfloat16()
    model_int8.load_state_dict(model_bf16.state_dict())
    
    # Apply INT8 (exclude lm_head)
    config = Int8MixedPrecisionTrainingConfig(output=True, grad_input=True, grad_weight=False)
    apply_int8_training(model_int8, config, filter_fn=lambda n, m: 'lm_head' not in n)
    
    # Verify INT8 is applied
    int8_module._INT8_MM_DEBUG = True
    int8_module._INT8_MM_CALL_COUNT = 0
    
    test_input = torch.randint(0, vocab_size, (1, 32), device='cuda')
    with torch.no_grad():
        _ = model_int8(test_input)
    print(f"INT8 calls per forward: {int8_module._INT8_MM_CALL_COUNT}")
    int8_module._INT8_MM_DEBUG = False
    
    # Train
    opt_bf16 = torch.optim.AdamW(model_bf16.parameters(), lr=lr, weight_decay=0.1)
    opt_int8 = torch.optim.AdamW(model_int8.parameters(), lr=lr, weight_decay=0.1)
    
    print(f"\nTraining {num_steps} steps...")
    print(f"{'Step':>5} | {'BF16':>10} | {'INT8':>10} | {'Diff':>10}")
    print("-" * 45)
    
    for step in range(num_steps):
        torch.manual_seed(step * 1000)
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device='cuda')
        target = torch.randint(0, vocab_size, (batch_size, seq_len), device='cuda')
        
        opt_bf16.zero_grad()
        loss_bf16 = nn.functional.cross_entropy(model_bf16(input_ids).view(-1, vocab_size), target.view(-1))
        loss_bf16.backward()
        torch.nn.utils.clip_grad_norm_(model_bf16.parameters(), 1.0)
        opt_bf16.step()
        
        opt_int8.zero_grad()
        loss_int8 = nn.functional.cross_entropy(model_int8(input_ids).view(-1, vocab_size), target.view(-1))
        loss_int8.backward()
        torch.nn.utils.clip_grad_norm_(model_int8.parameters(), 1.0)
        opt_int8.step()
        
        if step % 10 == 0 or step == num_steps - 1:
            diff = abs(loss_bf16.item() - loss_int8.item())
            print(f"{step:>5} | {loss_bf16.item():>10.4f} | {loss_int8.item():>10.4f} | {diff:>10.4f}")
    
    print("\n✓ If diffs are small (<0.1), INT8 is working correctly!")

if __name__ == "__main__":
    main()
