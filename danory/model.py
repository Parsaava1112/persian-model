import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        scale = torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        return (x * scale.to(x.dtype)) * self.weight


def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim, theta=10000.0):
        super().__init__()
        if head_dim % 2:
            raise ValueError("RoPE requires an even attention head dimension.")
        inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, x, seq_len):
        positions = torch.arange(seq_len, device=x.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(positions, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos().to(dtype=x.dtype)[None, None, :, :]
        sin = emb.sin().to(dtype=x.dtype)[None, None, :, :]
        return cos, sin


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        n_embd = int(cfg["n_embd"])
        self.n_head = int(cfg["n_head"])
        self.n_kv_head = int(cfg["n_kv_head"])
        if n_embd % self.n_head:
            raise ValueError("n_embd must be divisible by n_head.")
        if self.n_head % self.n_kv_head:
            raise ValueError("n_head must be divisible by n_kv_head.")
        self.head_dim = n_embd // self.n_head
        if self.head_dim % 2:
            raise ValueError("n_embd / n_head must be even for RoPE.")

        bias = bool(cfg.get("bias", False))
        self.q_proj = nn.Linear(n_embd, self.n_head * self.head_dim, bias=bias)
        self.k_proj = nn.Linear(n_embd, self.n_kv_head * self.head_dim, bias=bias)
        self.v_proj = nn.Linear(n_embd, self.n_kv_head * self.head_dim, bias=bias)
        self.out_proj = nn.Linear(n_embd, n_embd, bias=bias)
        self.rope = RotaryEmbedding(self.head_dim, float(cfg.get("rope_theta", 10000.0)))
        self.dropout = float(cfg.get("dropout", 0.0))

    def forward(self, x):
        batch, seq_len, channels = x.shape
        q = self.q_proj(x).view(batch, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch, seq_len, self.n_kv_head, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch, seq_len, self.n_kv_head, self.head_dim).transpose(1, 2)

        cos, sin = self.rope(q, seq_len)
        q = q * cos + rotate_half(q) * sin
        k = k * cos + rotate_half(k) * sin

        repeat_factor = self.n_head // self.n_kv_head
        if repeat_factor > 1:
            k = k.repeat_interleave(repeat_factor, dim=1)
            v = v.repeat_interleave(repeat_factor, dim=1)

        y = F.scaled_dot_product_attention(
            q, k, v,
            is_causal=True,
            dropout_p=self.dropout if self.training else 0.0
        )
        y = y.transpose(1, 2).contiguous().view(batch, seq_len, channels)
        return self.out_proj(y)


class SwiGLU(nn.Module):
    def __init__(self, dim, bias=False, hidden_dim=None):
        super().__init__()
        hidden = int(hidden_dim) if hidden_dim is not None else int(math.ceil((8.0 * dim / 3.0) / 64.0) * 64)
        self.gate = nn.Linear(dim, hidden, bias=bias)
        self.up = nn.Linear(dim, hidden, bias=bias)
        self.down = nn.Linear(hidden, dim, bias=bias)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        dim = int(cfg["n_embd"])
        self.norm1 = RMSNorm(dim)
        self.attn = CausalSelfAttention(cfg)
        self.norm2 = RMSNorm(dim)
        self.mlp = SwiGLU(dim, bool(cfg.get("bias", False)), cfg.get("ffn_dim"))
        self.dropout = nn.Dropout(float(cfg.get("dropout", 0.0)))

    def forward(self, x):
        x = x + self.dropout(self.attn(self.norm1(x)))
        x = x + self.dropout(self.mlp(self.norm2(x)))
        return x


class DanoryGPT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = dict(cfg)
        self._validate_config(cfg)
        dim = int(cfg["n_embd"])
        self.token_emb = nn.Embedding(int(cfg["vocab_size"]), dim)
        self.pos_drop = nn.Dropout(float(cfg.get("dropout", 0.0)))
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(int(cfg["n_layer"]))])
        self.norm_f = RMSNorm(dim)
        self.lm_head = nn.Linear(dim, int(cfg["vocab_size"]), bias=False)
        if cfg.get("tie_weights", True):
            self.lm_head.weight = self.token_emb.weight
        self.apply(self._init_linear)
        nn.init.normal_(self.token_emb.weight, mean=0.0, std=0.02)
        # Scale residual output projections for a deeper, more stable stack.
        residual_std = 0.02 / math.sqrt(2.0 * int(cfg["n_layer"]))
        for block in self.blocks:
            nn.init.normal_(block.attn.out_proj.weight, mean=0.0, std=residual_std)
            nn.init.normal_(block.mlp.down.weight, mean=0.0, std=residual_std)

    @staticmethod
    def _validate_config(cfg):
        required = ("vocab_size", "block_size", "n_layer", "n_head", "n_kv_head", "n_embd")
        missing = [k for k in required if k not in cfg]
        if missing:
            raise KeyError(f"Missing model config keys: {missing}")
        if min(int(cfg[k]) for k in required) <= 0:
            raise ValueError("Model dimensions must be positive.")
        if int(cfg["n_embd"]) % int(cfg["n_head"]):
            raise ValueError("n_embd must be divisible by n_head.")
        if int(cfg["n_head"]) % int(cfg["n_kv_head"]):
            raise ValueError("n_head must be divisible by n_kv_head.")
        if (int(cfg["n_embd"]) // int(cfg["n_head"])) % 2:
            raise ValueError("Attention head dimension must be even.")

    @staticmethod
    def _init_linear(module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def parameter_count(self, trainable_only=True):
        params = self.parameters()
        return sum(p.numel() for p in params if (p.requires_grad or not trainable_only))

    def forward(self, idx, targets=None, loss_mask=None):
        if idx.ndim != 2:
            raise ValueError("Input must have shape [batch, sequence].")
        if idx.size(1) > int(self.cfg["block_size"]):
            raise ValueError(f"Input length exceeds block_size={self.cfg['block_size']}.")
        if idx.numel() and (idx.min() < 0 or idx.max() >= int(self.cfg["vocab_size"])):
            raise ValueError("Input contains token IDs outside the vocabulary.")

        x = self.pos_drop(self.token_emb(idx))
        for block in self.blocks:
            x = block(x)
        logits = self.lm_head(self.norm_f(x))
        loss = None

        if targets is not None:
            raw = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
                reduction="none",
                ignore_index=-100
            ).view_as(targets)
            valid = targets.ne(-100)
            mask = valid.to(raw.dtype)
            if loss_mask is not None:
                mask = mask * loss_mask.to(device=raw.device, dtype=raw.dtype)
            loss = (raw * mask).sum() / mask.sum().clamp_min(1.0)
        return logits, loss

    @staticmethod
    def _top_k(logits, top_k):
        if top_k is None or int(top_k) <= 0:
            return logits
        k = min(int(top_k), logits.size(-1))
        threshold = torch.topk(logits, k, dim=-1).values[..., -1, None]
        return logits.masked_fill(logits < threshold, -float("inf"))

    @staticmethod
    def _top_p(logits, top_p):
        if top_p is None or float(top_p) >= 1.0:
            return logits
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        probs = F.softmax(sorted_logits, dim=-1)
        cumulative = probs.cumsum(dim=-1)
        remove_sorted = cumulative > float(top_p)
        remove_sorted[..., 1:] = remove_sorted[..., :-1].clone()
        remove_sorted[..., 0] = False
        remove_mask = torch.zeros_like(logits, dtype=torch.bool)
        remove_mask.scatter_(-1, sorted_indices, remove_sorted)
        return logits.masked_fill(remove_mask, -float("inf"))

    @torch.inference_mode()
    def generate(
        self, idx, max_new_tokens=100, temperature=0.8, top_k=40,
        top_p=0.9, repetition_penalty=1.10, eos_token_id=None,
        forbidden_token_ids=(), min_new_tokens=8, do_sample=True, repetition_window=128,
        no_repeat_ngram_size=0
    ):
        self.eval()
        if idx.ndim != 2 or idx.size(1) == 0:
            raise ValueError("Prompt must have shape [batch, sequence] and cannot be empty.")
        idx = idx.long()
        if max_new_tokens < 0:
            raise ValueError("max_new_tokens cannot be negative.")
        if temperature <= 0:
            raise ValueError("temperature must be positive.")
        if not 0 < top_p <= 1:
            raise ValueError("top_p must be in (0, 1].")
        if repetition_penalty < 1:
            raise ValueError("repetition_penalty must be >= 1.")
        if int(no_repeat_ngram_size) < 0:
            raise ValueError("no_repeat_ngram_size cannot be negative.")
        if idx.min() < 0 or idx.max() >= int(self.cfg["vocab_size"]):
            raise ValueError("Prompt contains token IDs outside vocabulary.")
        if max_new_tokens == 0:
            return idx

        min_new_tokens = max(0, min(int(min_new_tokens), int(max_new_tokens)))
        finished = torch.zeros(idx.size(0), dtype=torch.bool, device=idx.device)
        generated = 0

        for _ in range(int(max_new_tokens)):
            context = idx[:, -int(self.cfg["block_size"]):]
            logits, _ = self(context)
            logits = logits[:, -1, :].float() / float(temperature)

            for token_id in forbidden_token_ids:
                if token_id is not None and 0 <= int(token_id) < logits.size(-1):
                    logits[:, int(token_id)] = -float("inf")

            if eos_token_id is not None and generated < min_new_tokens:
                logits[:, int(eos_token_id)] = -float("inf")

            if repetition_penalty > 1:
                for b in range(idx.size(0)):
                    history = idx[b, -int(repetition_window):] if repetition_window and int(repetition_window) > 0 else idx[b]
                    seen = history.unique()
                    seen = seen[(seen >= 0) & (seen < logits.size(-1))]
                    vals = logits[b, seen]
                    logits[b, seen] = torch.where(vals < 0, vals * repetition_penalty, vals / repetition_penalty)

            # Prevent exact repeated n-grams without globally banning useful words.
            ngram_n = int(no_repeat_ngram_size)
            if ngram_n > 1 and idx.size(1) >= ngram_n - 1:
                for b in range(idx.size(0)):
                    seq = idx[b].tolist()
                    prefix = tuple(seq[-(ngram_n - 1):])
                    banned = set()
                    for j in range(len(seq) - ngram_n + 1):
                        if tuple(seq[j:j + ngram_n - 1]) == prefix:
                            banned.add(seq[j + ngram_n - 1])
                    if banned:
                        valid_banned = [t for t in banned if 0 <= t < logits.size(-1)]
                        if valid_banned:
                            logits[b, valid_banned] = -float("inf")

            if do_sample:
                logits = self._top_k(logits, top_k)
                logits = self._top_p(logits, top_p)
                probs = F.softmax(logits, dim=-1)
                if not torch.isfinite(probs).all() or (probs.sum(dim=-1) <= 0).any():
                    raise RuntimeError("Invalid generation probabilities; inspect token IDs and filters.")
                next_id = torch.multinomial(probs, num_samples=1)
            else:
                next_id = torch.argmax(logits, dim=-1, keepdim=True)

            # Keep completed batch rows on EOS while other rows continue.
            if eos_token_id is not None:
                eos_fill = torch.full_like(next_id, int(eos_token_id))
                next_id = torch.where(finished[:, None], eos_fill, next_id)

            idx = torch.cat((idx, next_id), dim=1)
            generated += 1
            if eos_token_id is not None:
                finished |= next_id[:, 0].eq(int(eos_token_id))
                if finished.all() and generated >= min_new_tokens:
                    break
        return idx
