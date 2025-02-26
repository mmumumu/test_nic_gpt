"""
Full definition of a GPT Language Model with SpiralAttention.
This version replaces the standard causal self-attention with a fixed spiral mixing mechanism,
which only mixes on the channel dimension and preserves causality.
References:
1) the official GPT-2 TensorFlow implementation released by OpenAI
2) huggingface/transformers PyTorch implementation
"""

import math
from dataclasses import dataclass
import inspect
import torch
import torch.nn as nn
from torch.nn import functional as F

###############################################
# Spiral mixing functions and modules
###############################################

def select_spiral(X, kernel_size):
    """
    X: Tensor of shape (B, C, T) where C is treated as rows (channels) and T is time.
    Returns H: Tensor of shape (B, C, T) after spiral (triangle-wave) re-indexing.
    """
    B, C, T = X.shape

    def make_triangle_wave(w: int, length: int) -> torch.Tensor:
        if w <= 1:
            return torch.zeros(length, dtype=torch.long, device=X.device)
        forward  = list(range(0, w))
        backward = list(range(w-2, 0, -1))
        base_pattern = forward + backward  # one full cycle
        cycle_len = len(base_pattern)       # = 2*(w-1)
        repeats = (length // cycle_len) + 1
        big_wave = base_pattern * repeats   # repeat list
        big_wave = big_wave[:length]          # truncate to exactly 'length'
        return torch.tensor(big_wave, dtype=torch.long, device=X.device)

    # Build wave_table: for each window size in [1, kernel_size]
    wave_table = []
    wave_table.append(torch.zeros(T, dtype=torch.long, device=X.device))  # index 0 unused
    for w in range(1, kernel_size + 1):
        wave_table.append(make_triangle_wave(w, T))
    wave_table = torch.stack(wave_table, dim=0)  # shape: (kernel_size+1, T)

    # For each channel index c, compute start_rows and window_sizes.
    c_range = torch.arange(C, device=X.device)
    start_rows = (c_range - (kernel_size - 1)).clamp(min=0)
    window_sizes = (c_range - start_rows) + 1

    # Compute row indices: row_idx[c, t] = start_rows[c] + wave_table[window_sizes[c], t]
    row_idx = start_rows.unsqueeze(1) + wave_table[window_sizes, :]
    row_idx = torch.minimum(row_idx, c_range.unsqueeze(1))

    # Time indices: 0..T-1 for each row.
    time_idx = torch.arange(T, device=X.device).unsqueeze(0).expand(C, T)
    H = X[:, row_idx, time_idx]
    return H

class SpiralFC1D(nn.Module):
    """
    Applies spiral mixing on the channel dimension.
    Expects input of shape (B, T, C), permutes to (B, C, T),
    applies spiral gathering and a pointwise convolution, then permutes back.
    """
    def __init__(self, in_dim, out_dim, kernel_size=3):
        super().__init__()
        self.kernel_size = kernel_size
        self.conv1d = nn.Conv1d(in_dim, out_dim, kernel_size=1)

    def forward(self, X):
        # X: (B, T, C)
        X = X.permute(0, 2, 1)             # (B, C, T)
        H = select_spiral(X, kernel_size=self.kernel_size)  # (B, C, T)
        H = self.conv1d(H)                # (B, out_dim, T)
        H = H.permute(0, 2, 1)             # (B, T, out_dim)
        return H

class SpiralAttention(nn.Module):
    """
    Replacement for the causal self-attention module.
    Instead of computing Q, K, V and learned attention weights,
    this module performs a fixed spiral mixing on the channel dimension.
    It does not mix across time steps, thereby preserving causality.
    Note: the attention_mask parameter is not used.
    """
    def __init__(self, config):
        super().__init__()
        self.kernel_size = 3  # can be made configurable
        self.spiral_fc = SpiralFC1D(in_dim=config.n_embd,
                                    out_dim=config.n_embd,
                                    kernel_size=self.kernel_size)

    def forward(self, x, layer_past=None, attention_mask=None):
        if attention_mask is not None:
            print("Warning: attention_mask is ignored in SpiralAttention.")
        # x: (B, T, C) where C == hidden_dim.
        return self.spiral_fc(x)

###############################################
# GPT Model Definition with SpiralAttention
###############################################

@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50257
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.1
    bias: bool = True

class Block(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = SpiralAttention(config)  # Use SpiralAttention instead of vanilla self-attention
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd)
        )

    def forward(self, x, layer_past=None, attention_mask=None):
        x = x + self.attn(self.ln_1(x), layer_past=layer_past, attention_mask=attention_mask)
        x = x + self.mlp(self.ln_2(x))
        return x

class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict({
            "wte": nn.Embedding(config.vocab_size, config.n_embd),
            "wpe": nn.Embedding(config.block_size, config.n_embd),
            "drop": nn.Dropout(config.dropout),
            "h": nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            "ln_f": nn.LayerNorm(config.n_embd)
        })
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # Weight tying
        self.transformer["wte"].weight = self.lm_head.weight

        print("Number of parameters: %.2fM" % (self.get_num_params() / 1e6))

    def get_num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        tok_emb = self.transformer["wte"](idx)  # (B, T, n_embd)
        pos_ids = torch.arange(0, t, device=device).unsqueeze(0)
        pos_emb = self.transformer["wpe"](pos_ids)  # (B, T, n_embd)
        x = self.transformer["drop"](tok_emb + pos_emb)
        for block in self.transformer["h"]:
            x = block(x)
        x = self.transformer["ln_f"](x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        return logits, loss

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        # Collect all parameters that require gradients.
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
        # Split parameters: weight tensors (2D or higher) get weight decay; biases and LayerNorm parameters (1D) do not.
        decay_params = [p for p in param_dict.values() if p.dim() >= 2]
        nodecay_params = [p for p in param_dict.values() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        # Optionally use fused AdamW if available and on CUDA.
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and (device_type == 'cuda')
        extra_args = {'fused': True} if use_fused else {}
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")
        return optimizer


    def estimate_mfu(self, fwdbwd_per_iter, dt):
        """ estimate model flops utilization (MFU) in units of A100 bfloat16 peak FLOPS """
        # first estimate the number of flops we do per iteration.
        # see PaLM paper Appendix B as ref: https://arxiv.org/abs/2204.02311
        N = self.get_num_params()
        cfg = self.config
        L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd//cfg.n_head, cfg.block_size
        flops_per_token = 6*N + 12*L*H*Q*T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        # express our flops throughput as ratio of A100 bfloat16 peak flops
        flops_achieved = flops_per_iter * (1.0/dt) # per second
        flops_promised = 312e12 # A100 GPU bfloat16 peak flops is 312 TFLOPS
        mfu = flops_achieved / flops_promised
        return mfu

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            # forward the model to get the logits for the index in the sequence
            logits, _ = self(idx_cond)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

        return idx