import torch
import torch.nn as nn


class Xtoy(nn.Module):
    def __init__(self, dx, dy):
        """ Map node features to global features """
        super().__init__()
        self.lin = nn.Linear(4 * dx, dy)

    def forward(self, X, x_mask):
        """ X: bs, n, dx. """
        x_mask = x_mask.expand(-1, -1, X.shape[-1])
        float_imask = 1 - x_mask.float()
        m = X.sum(dim=1) / torch.sum(x_mask, dim=1)
        mi = (X + 1e5 * float_imask).min(dim=1)[0]
        ma = (X - 1e5 * float_imask).max(dim=1)[0]
        std = torch.sum(((X - m[:, None, :]) ** 2) * x_mask, dim=1) / torch.sum(x_mask, dim=1)
        z = torch.hstack((m, mi, ma, std))
        out = self.lin(z)
        return out


class Etoy(nn.Module):
    def __init__(self, d, dy):
        """ Map edge features to global features. """
        super().__init__()
        self.lin = nn.Linear(4 * d, dy)

    def forward(self, E, e_mask1, e_mask2):
        """ E: bs, n, n, de
            Features relative to the diagonal of E could potentially be added.
        """
        mask = (e_mask1 * e_mask2).expand(-1, -1, -1, E.shape[-1])
        float_imask = 1 - mask.float()
        divide = torch.sum(mask, dim=(1, 2))
        m = E.sum(dim=(1, 2)) / divide
        mi = (E + 1e5 * float_imask).min(dim=2)[0].min(dim=1)[0]
        ma = (E - 1e5 * float_imask).max(dim=2)[0].max(dim=1)[0]
        std = torch.sum(((E - m[:, None, None, :]) ** 2) * mask, dim=(1, 2)) / divide
        z = torch.hstack((m, mi, ma, std))
        out = self.lin(z)
        return out


class SpectraCrossAttention(nn.Module):
    """Cross-attention from graph node features to spectral peak tokens.

    Allows each node in the molecular graph to selectively attend to relevant
    peaks from the mass spectrum encoder, providing fine-grained spectral
    conditioning beyond a single global vector.

    Args:
        dx: Dimension of graph node features (queries).
        d_peak: Dimension of peak token features (keys/values).
        n_head: Number of attention heads.
        dropout: Dropout probability on attention weights.
    """

    def __init__(self, dx: int, d_peak: int, n_head: int = 8, dropout: float = 0.1):
        super().__init__()
        assert dx % n_head == 0, f"dx ({dx}) must be divisible by n_head ({n_head})"
        self.dx = dx
        self.d_peak = d_peak
        self.n_head = n_head
        self.df = dx // n_head

        # Queries from graph nodes, keys/values from peak tokens
        self.q = nn.Linear(dx, dx)
        self.k = nn.Linear(d_peak, dx)
        self.v = nn.Linear(d_peak, dx)
        self.out_proj = nn.Linear(dx, dx)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(dx)

    def forward(self, X, peak_tokens, node_mask, peak_mask):
        """
        Args:
            X: (bs, n, dx) — graph node features (queries).
            peak_tokens: (bs, Np, d_peak) — encoder peak token features (keys/values).
            node_mask: (bs, n) — True for valid graph nodes.
            peak_mask: (bs, Np) — True for valid (non-padded) peaks.
        Returns:
            Updated X with same shape (bs, n, dx), with residual connection.
        """
        bs, n, _ = X.shape
        Np = peak_tokens.size(1)

        Q = self.q(X)                        # (bs, n, dx)
        K = self.k(peak_tokens)              # (bs, Np, dx)
        V = self.v(peak_tokens)              # (bs, Np, dx)

        # Reshape for multi-head: (bs, n_head, seq_len, df)
        Q = Q.view(bs, n, self.n_head, self.df).transpose(1, 2)    # (bs, nh, n, df)
        K = K.view(bs, Np, self.n_head, self.df).transpose(1, 2)   # (bs, nh, Np, df)
        V = V.view(bs, Np, self.n_head, self.df).transpose(1, 2)   # (bs, nh, Np, df)

        # Attention scores: (bs, nh, n, Np)
        attn = (Q @ K.transpose(-2, -1)) / (self.df ** 0.5)

        # Mask out padded peaks: peak_mask is (bs, Np), expand to (bs, 1, 1, Np)
        if peak_mask is not None:
            peak_pad_mask = ~peak_mask.bool()  # True = padded
            attn = attn.masked_fill(peak_pad_mask[:, None, None, :], float('-inf'))

        attn = torch.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        # Weighted sum of values: (bs, nh, n, df) -> (bs, n, dx)
        out = (attn @ V).transpose(1, 2).reshape(bs, n, self.dx)
        out = self.out_proj(out)

        # Mask invalid nodes
        x_mask = node_mask.unsqueeze(-1)  # (bs, n, 1)
        out = out * x_mask

        # Residual + LayerNorm
        X = self.norm(X + out)
        return X


def masked_softmax(x, mask, **kwargs):
    if mask.sum() == 0:
        return x
    x_masked = x.clone()
    x_masked[mask == 0] = -float("inf")
    return torch.softmax(x_masked, **kwargs)
