"""The reader: frozen EmbeddingGemma 2 audio frames (25/s, 1536-d) -> characters (CTC).

The encoder only looks back (no right context), so the reader's first job is to look both ways:
a few bidirectional Transformer layers over the frames. Each frame then speaks for two output
slots (50/s): Japanese kana and spelled-out English go past 25 symbols a second, and CTC needs
at least one slot per symbol plus blanks between repeats ("ll", "っっ").
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


LANGS = ["zh", "en", "ja", "ko"]


class RotaryAttention(nn.Module):
    def __init__(self, d, heads):
        super().__init__()
        self.h, self.dk = heads, d // heads
        self.qkv = nn.Linear(d, 3 * d)
        self.out = nn.Linear(d, d)

    def rope(self, x, pos):
        half = self.dk // 2
        freq = 1.0 / (10000 ** (torch.arange(half, device=x.device, dtype=torch.float32) / half))
        ang = pos[:, None].float() * freq[None]
        cos, sin = ang.cos()[None, None].to(x.dtype), ang.sin()[None, None].to(x.dtype)
        x1, x2 = x[..., :half], x[..., half:]
        return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], -1)

    def forward(self, x, mask, T=None):
        B, T_, D = x.shape
        T = T or T_
        q, k, v = self.qkv(x).view(B, T, 3, self.h, self.dk).permute(2, 0, 3, 1, 4)
        pos = torch.arange(T, device=x.device)
        q, k = self.rope(q, pos), self.rope(k, pos)
        attn = None if mask is None else mask[:, None, None, :]
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=attn)
        return self.out(y.transpose(1, 2).reshape(B, T, self.h * self.dk))


class Block(nn.Module):
    def __init__(self, d, heads, ff, drop, kernel=15):
        super().__init__()
        self.n1, self.att = nn.LayerNorm(d), RotaryAttention(d, heads)
        self.n2 = nn.LayerNorm(d)
        self.conv = nn.Conv1d(d, d, kernel, padding=kernel // 2, groups=d)  # local detail attention misses
        self.n3 = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, ff), nn.SiLU(), nn.Dropout(drop), nn.Linear(ff, d))
        self.drop = nn.Dropout(drop)

    def forward(self, x, mask, T=None):
        x = x + self.drop(self.att(self.n1(x), mask, T))
        c = self.n2(x)
        if mask is not None:
            c = c * mask[..., None]
        x = x + self.drop(self.conv(c.transpose(1, 2)).transpose(1, 2))
        return x + self.drop(self.ff(self.n3(x)))


class Reader(nn.Module):
    def __init__(self, vocab, in_dim=1536, d=512, layers=6, heads=8, ff=2048, drop=0.1, up=2, layer_in=0, layer_dim=1024):
        super().__init__()
        self.cfg = dict(vocab=vocab, in_dim=in_dim, d=d, layers=layers, heads=heads, ff=ff, drop=drop, up=up,
                        layer_in=layer_in, layer_dim=layer_dim)
        self.inp = nn.Sequential(nn.LayerNorm(in_dim), nn.Linear(in_dim, d), nn.Dropout(drop))
        # Every encoder layer, not only the last (`layer_in` hidden states, mixed by learned
        # weights): EmbeddingGemma's top layers serve meaning, and spelling detail sits lower.
        # The projection starts at zero, so a reader trained on the last layer alone starts
        # exactly where it was.
        self.layer_in = layer_in
        if layer_in:
            self.mix_w = nn.Parameter(torch.zeros(layer_in))
            self.mix_norm = nn.LayerNorm(layer_dim, elementwise_affine=False)
            self.mix_proj = nn.Linear(layer_dim, d)
            nn.init.zeros_(self.mix_proj.weight); nn.init.zeros_(self.mix_proj.bias)
        # 0 = written style (punctuation, case, digits); 1 = plain style (sources transcribed without
        # them). Every source teaches recognition; only written-style ones teach punctuation, and
        # the app always asks for style 0.
        self.style = nn.Embedding(2, d)
        # Which language is spoken: 0 = unknown (the reader works it out), else LANGS[i-1]. The
        # app passes the note's language when it knows it; training hides it 30% of the time so
        # "unknown" works too. It stops a Japanese sentence drifting into Hangul.
        self.lang = nn.Embedding(len(LANGS) + 1, d)
        self.blocks = nn.ModuleList([Block(d, heads, ff, drop) for _ in range(layers)])
        self.norm = nn.LayerNorm(d)
        self.up = up
        self.expand = nn.Linear(d, d * up)
        self.head = nn.Linear(d, vocab)

    def mix(self, hidden):
        """hidden: (L, B, T, layer_dim) -> (B, T, d)."""
        w = self.mix_w.softmax(0)
        return self.mix_proj((self.mix_norm(hidden) * w[:, None, None, None]).sum(0))

    def forward(self, x, lengths, style=None, lang=None, hidden=None):
        """x: (B, T, in_dim); lengths: (B,) frames; hidden: (L, B, T, layer_dim) every encoder
        layer when `layer_in`. Returns log-probs (B, T*up, V) and out lengths."""
        B, T, _ = x.shape
        mask = torch.arange(T, device=x.device)[None] < lengths[:, None]
        if style is None:
            style = torch.zeros(B, dtype=torch.long, device=x.device)
        if lang is None:
            lang = torch.zeros(B, dtype=torch.long, device=x.device)
        h = self.inp(x) + self.style(style)[:, None] + self.lang(lang)[:, None]
        if self.layer_in:
            h = h + self.mix(hidden)
        for b in self.blocks:
            h = b(h, mask)
        h = self.norm(h)
        h = self.expand(h).view(B, T * self.up, -1)
        return self.head(F.silu(h)).log_softmax(-1), lengths * self.up


    def forward_static(self, x, T, style, lang, hidden=None):
        """One window of exactly T frames, no padding: what the exported graph runs (every shape
        a constant, nothing for a converter to work out)."""
        h = self.inp(x) + self.style(style)[:, None] + self.lang(lang)[:, None]
        if self.layer_in:
            h = h + self.mix(hidden)
        for b in self.blocks:
            h = b(h, None, T)
        h = self.expand(self.norm(h)).view(1, T * self.up, self.cfg["d"])
        return self.head(F.silu(h)).log_softmax(-1)


def spec_augment(x, lengths, time_masks=4, time_width=0.05, feat_drop=0.1):
    """Masks spans of frames and drops random feature channels (applied to encoder frames)."""
    B, T, D = x.shape
    x = x.clone()
    for b in range(B):
        L = int(lengths[b])
        for _ in range(time_masks):
            w = int(torch.randint(0, max(1, int(L * time_width)) + 1, (1,)))
            s = int(torch.randint(0, max(1, L - w), (1,)))
            x[b, s:s + w] = 0
    if feat_drop:
        keep = (torch.rand(B, 1, D, device=x.device) > feat_drop).to(x.dtype)
        x = x * keep / (1 - feat_drop)
    return x
