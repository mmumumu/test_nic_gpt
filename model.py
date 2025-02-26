import math
import torch
import torch.nn as nn
import torch.nn.functional as F

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
        self.in_dim = in_dim
        self.out_dim = out_dim
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
    该模块仅在通道维度上进行固定的螺旋混合，不跨时间维度聚合信息，因此天然满足因果约束。
    注意：attention_mask 参数目前不被使用，如果传入非空掩码，会给出警告。
    """
    def __init__(self, config):
        super().__init__()
        self.hidden_dim = config.n_embd
        self.kernel_size = 3  # 可配置
        self.spiral_fc = SpiralFC1D(in_dim=self.hidden_dim,
                                    out_dim=self.hidden_dim,
                                    kernel_size=self.kernel_size)

    def forward(self, x, layer_past=None, attention_mask=None):
        if attention_mask is not None:
            # 注意：本模块不使用 attention_mask，因此此处给出警告并忽略该参数。
            print("Warning: attention_mask is ignored in SpiralAttention.")
        # x: (B, T, C) where C == hidden_dim.
        return self.spiral_fc(x)

###############################################
# NanoGPT-style Model Definition
###############################################

class GPTConfig:
    def __init__(self, n_layer=12, n_embd=768, n_head=12, block_size=1024):
        self.n_layer = n_layer
        self.n_embd = n_embd
        self.n_head = n_head
        self.block_size = block_size

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        # 用 SpiralAttention 替换标准的自注意力模块
        self.attn = SpiralAttention(config)
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
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict({
            "wte": nn.Embedding(50257, config.n_embd),              # 词嵌入
            "wpe": nn.Embedding(config.block_size, config.n_embd),    # 位置嵌入
            "drop": nn.Dropout(0.1),
            "h": nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            "ln_f": nn.LayerNorm(config.n_embd)
        })
        self.lm_head = nn.Linear(config.n_embd, 50257, bias=False)

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, "Sequence length exceeds block size."
        token_embeddings = self.transformer["wte"](idx)  # (B, T, C)
        position_ids = torch.arange(0, t, device=device).unsqueeze(0)
        position_embeddings = self.transformer["wpe"](position_ids)  # (B, T, C)
        x = self.transformer["drop"](token_embeddings + position_embeddings)
        for block in self.transformer["h"]:
            x = block(x)
        x = self.transformer["ln_f"](x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss


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