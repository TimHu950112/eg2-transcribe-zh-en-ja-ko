"""The EmbeddingGemma 2 audio encoder in a form that traces: same modules and weights, but the
chunked-attention mask is a precomputed constant for a fixed window instead of transformers'
vmap-built one (which neither torch.jit nor coremltools can follow).

A window is 30 s of audio = 3000 mel frames -> 750 encoder frames. Longer recordings are cut
into windows on the device (the encoder only looks ~0.5 s back per layer, so a little overlap
at each cut is enough).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel

MEL_FRAMES = 3000
ENC_FRAMES = 750


def blocked_mask(seq_len, chunk, left, right):
    """[1, 1, num_blocks, chunk, context] bool: query q sees keys q-left+2 .. q+right (as transformers builds it) inside the sequence."""
    past = left - 1
    ctx = chunk + past + right
    nb = (seq_len + chunk - 1) // chunk
    q = torch.arange(nb)[:, None, None] * chunk + torch.arange(chunk)[None, :, None]  # nb, chunk, 1
    k = torch.arange(nb)[:, None, None] * chunk - past + torch.arange(ctx)[None, None, :]  # nb, 1, ctx
    ok = (k >= 0) & (k < seq_len) & (q < seq_len) & (k > q - past) & (k <= q + right)
    return ok[None, None]


class StaticAttention(nn.Module):
    """Gemma4AudioAttention for one fixed length: block/context indices and the relative-position
    keys are constants, the soft cap a Python float — nothing for a converter to evaluate."""

    def __init__(self, a, seq_len, pos):
        super().__init__()
        self.a = a
        self.T = seq_len
        self.H, self.D = a.num_heads, a.head_dim
        self.C, self.P, self.F = a.chunk_size, a.max_past_horizon, a.max_future_horizon
        self.ctx = a.context_size
        self.nb = (seq_len + self.C - 1) // self.C
        self.cap = float(a.attention_logits_soft_cap)
        # Key/value windows: block b reads padded positions b*C .. b*C+ctx-1.
        idx = torch.arange(self.nb)[:, None] * self.C + torch.arange(self.ctx)[None, :]
        self.register_buffer("kv_idx", idx.reshape(-1), persistent=False)
        with torch.no_grad():
            rk = a.relative_k_proj(pos).view(-1, self.H, self.D).permute(1, 2, 0).contiguous()  # H, D, P
        self.register_buffer("rel_k", rk, persistent=False)
        self.n_pos = int(rk.shape[-1])
        self.register_buffer("qmul", (a.q_scale * F.softplus(a.per_dim_scale.detach())).float(), persistent=False)

    def forward(self, hidden_states, position_embeddings=None, attention_mask=None):
        x, mask = hidden_states, attention_mask
        a, B, T, H, D, C, nb, ctx = self.a, x.shape[0], self.T, self.H, self.D, self.C, self.nb, self.ctx
        q = a.q_proj(x).view(B, T, H, D) * self.qmul
        k = a.k_proj(x).view(B, T, H, D) * a.k_scale
        v = a.v_proj(x).view(B, T, H, D)
        q = F.pad(q, (0, 0, 0, 0, 0, nb * C - T)).view(B, nb, C, H, D)
        pad_k = (0, 0, 0, 0, self.P, self.F + C - 1)
        k = F.pad(k, pad_k).index_select(1, self.kv_idx).view(B, nb, ctx, H, D)
        v = F.pad(v, pad_k).index_select(1, self.kv_idx).view(B, nb, ctx, H, D)
        qh = q.permute(0, 3, 1, 2, 4)                       # B H nb C D
        ac = qh @ k.permute(0, 3, 1, 4, 2)                  # B H nb C ctx
        bd = (qh.reshape(B, H, nb * C, D) @ self.rel_k).reshape(B, H, nb, C, -1)
        bd = F.pad(bd, (0, ctx + 1 - self.n_pos)).reshape(B, H, nb, C * (ctx + 1))[..., : C * ctx].reshape(B, H, nb, C, ctx)
        w = torch.tanh((ac + bd) / self.cap) * self.cap
        # -1e4, not -1e9: the graph runs in float16 on the device, where -1e9 is -inf and a row
        # with nothing visible (the padding past the end) softmaxes to NaN on the CPU path.
        w = w.masked_fill(~mask, -1e4)
        w = torch.softmax(w, dim=-1)
        o = w @ v.permute(0, 3, 1, 2, 4)                    # B H nb C D
        o = o.permute(0, 2, 3, 1, 4).reshape(B, nb * C, H * D)[:, :T]
        return a.post(o), None


class TraceableEncoder(nn.Module):
    def __init__(self, tower, layers=None, mel_frames=MEL_FRAMES):
        super().__init__()
        self.t = tower
        c = tower.config
        self.n_layers = layers or c.num_hidden_layers
        T = (mel_frames + 3) // 4
        self.register_buffer("mask", blocked_mask(T, c.attention_chunk_size, c.attention_context_left,
                                                  c.attention_context_right), persistent=False)
        pos = tower.rel_pos_enc(torch.zeros(1, T, c.hidden_size))
        for layer in tower.layers[: self.n_layers]:
            layer.self_attn = StaticAttention(layer.self_attn, T, pos)
        # Clamps at ±1e10 are no-ops in practice and only cost the converter; the clippable
        # linears' learned ranges stay.
        for layer in tower.layers:
            for m in (layer, layer.feed_forward1, layer.feed_forward2, layer.lconv1d):
                m.gradient_clipping = 1e10

    def forward(self, mel):
        h, _ = self.t.subsample_conv_projection(mel, None)
        for layer in self.t.layers[: self.n_layers]:
            h = layer(h, attention_mask=self.mask, position_embeddings=None)
        return self.t.output_proj(h)


def load_tower():
    full = AutoModel.from_pretrained("google/embeddinggemma-2", dtype=torch.float32)
    tower = full.audio_tower.eval()
    del full
    return tower


if __name__ == "__main__":
    mel = torch.randn(1, MEL_FRAMES, 128)
    with torch.no_grad():
        ref = load_tower()(mel, torch.ones(1, MEL_FRAMES, dtype=torch.bool), return_dict=True).last_hidden_state
        out = TraceableEncoder(load_tower()).eval()(mel)
    print(out.shape, "rel diff vs transformers:", ((out - ref).norm() / ref.norm()).item())
