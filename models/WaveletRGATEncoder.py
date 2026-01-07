import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

# ---- 1. Chebyshev Wavelet (single layer K-order Chebyshev polynomial) ----
class ChebWavelet(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, K: int = 3, dropout: float = 0.1):
        super().__init__()
        self.K = K
        self.lin = nn.Linear(in_dim * K, out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, X: torch.Tensor, L_rescaled: torch.sparse.FloatTensor) -> torch.Tensor:
        """
        X: [N, Fin]
        L_rescaled: sparse [N,N], preprocessed (I - D^-1/2 A D^-1/2 - I)
        Returns: [N, Fout]
        """
        # T0(X) = X
        T_k = [X]
        # T1(X) = L_rescaled @ X
        T_k.append(torch.sparse.mm(L_rescaled, X))
        # T_k = 2 L_rescaled T_{k-1} - T_{k-2}
        for _ in range(2, self.K):
            T_next = 2.0 * torch.sparse.mm(L_rescaled, T_k[-1]) - T_k[-2]
            T_k.append(T_next)
        H = torch.cat(T_k[:self.K], dim=-1)
        H = self.dropout(self.lin(H))
        return H  # [N, out_dim]


# ---- 2. Relation-aware GAT with semantic prior sim1_edge ----
class RelGATLayerSemantic(nn.Module):
    def __init__(self,
                 in_dim: int,
                 out_dim: int,
                 n_rel: int,
                 rel_dim: Optional[int] = None,
                 heads: int = 2,
                 dropout: float = 0.1,
                 concat: bool = True,
                 gamma_init: float = 1.0,
                 learnable_beta: bool = True,
                 beta_init: float = 1.0):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.heads = heads
        self.concat = concat

        # For concat=True, ensure out_dim is divisible by heads; determine per-head dim d_k
        if concat:
            assert out_dim % heads == 0, f"out_dim({out_dim}) must be divisible by heads({heads}) when concat=True"
            d_k = out_dim // heads
        else:
            d_k = out_dim

        self.W = nn.Linear(in_dim, heads * d_k, bias=False)

        self.a_src = nn.Parameter(torch.randn(heads, d_k))
        self.a_dst = nn.Parameter(torch.randn(heads, d_k))


        rd = rel_dim if rel_dim is not None else out_dim
        self.rel_emb = nn.Embedding(n_rel, rd)

        self.beta = nn.Parameter(torch.tensor(float(beta_init)), requires_grad=False)

        self.dropout = nn.Dropout(dropout)
        self.leakyrelu = nn.LeakyReLU(0.2)

        self.msg_lin = nn.Linear(d_k, d_k, bias=False)

    def forward(self,
                H: torch.Tensor,                 # Wavelet output; [N, Fin]
                edge_index: torch.Tensor,        # [2, E] (Long)
                sim1_edge: Optional[torch.Tensor] = None,   # [E] (Float, can be None)
                ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        N = H.size(0)
        h = self.heads
        device = H.device

        # Move all input tensors to same device
        if edge_index.device != device:
            edge_index = edge_index.to(device)
        if sim1_edge is not None and sim1_edge.device != device:
            sim1_edge = sim1_edge.to(device)


        # Linear projection & multi-head split
        H_proj = self.W(H)                       # [N, h*d_k]
        d_k = H_proj.size(-1) // h
        H_proj = H_proj.view(N, h, d_k)          # [N, h, d_k]

        src, dst = edge_index[0], edge_index[1]  # [E] (Long, on device)
        h_src = H_proj[src]                      # [E, h, d_k]
        h_dst = H_proj[dst]                      # [E, h, d_k]

        # Attention logits
        e_struct = (h_src * self.a_src).sum(-1) + (h_dst * self.a_dst).sum(-1)  # [E, h]

        if sim1_edge is not None:
            e_sem = self.beta * sim1_edge.unsqueeze(-1)     # [E, h]
            e = e_struct + e_sem

        e = self.leakyrelu(e)

        # softmax (normalize by edges with same dst)
        e_stable = e - e.max(dim=0, keepdim=True).values                  # [E, h]
        exp_e = torch.exp(e_stable)                                       # [E, h] (device)

        # Create tensor with explicit device
        denom = torch.zeros(N, h, dtype=exp_e.dtype, device=device)
        denom = denom.index_add(0, dst, exp_e)

        alpha = exp_e / (denom[dst] + 1e-12)                              # [E, h]
        alpha = self.dropout(alpha)

        # Message passing and aggregation
        m = alpha.unsqueeze(-1) * h_src                                    # [E, h, d_k]
        out = torch.zeros(N, h, d_k, dtype=m.dtype, device=device)
        out = out.index_add(0, dst, m)

        # Merge heads
        if self.concat:
            out_h = out.reshape(N, h * d_k)                                # [N, out_dim]
        else:
            out_h = out.mean(dim=1)                                        # [N, out_dim]

        # Relation aggregation base (without alpha, for relation-level update)
        msg_base = self.msg_lin(h_src).mean(dim=1)                         # [E, d_k]
        alpha_mean = alpha.mean(dim=1)                                     # [E]

        return out_h, alpha_mean, msg_base


class WaveletRGATEncoder(nn.Module):
    """
    Input:
      X:            [N, in_dim]
      edge_index:   [2, E_all]
      edge_type:    [E_all]
      sim1_edge:    [E_all] or None
      edge_weight:  [E_all] or None
      L_rescaled:   sparse [N, N]
    Output:
      Z_ent: [N, out_dim]
      Z_rel: [R_tot, out_dim]
    """
    def __init__(self,
                 in_dim: int,
                 hidden_dim: int,
                 out_dim: int,
                 n_rel_total: int,
                 rel_dim: Optional[int] = None,
                 cheb_K: int = 3,
                 gat_heads: int = 4,
                 dropout: float = 0.1,
                 concat_heads: bool = True,
                 beta_init: float = 1.0,
                 rel_embedding: Optional[nn.Embedding] = None,
                 tie_rel_tables: bool = True,
                ):
        super().__init__()
        self.out_dim = out_dim
        self.concat_heads = concat_heads
        self.gat_heads = gat_heads


        self.wavelet = ChebWavelet(in_dim, hidden_dim, K=cheb_K, dropout=dropout)

        self.rgat = RelGATLayerSemantic(hidden_dim, out_dim,
                                        n_rel=n_rel_total, rel_dim=rel_dim,
                                        heads=gat_heads, dropout=dropout,
                                        concat=concat_heads,
                                        beta_init=beta_init)

        # PReLU channels fixed to out_dim
        self.act = nn.PReLU(out_dim)
        if concat_heads:
            assert out_dim % gat_heads == 0, \
                f"out_dim({out_dim}) must be divisible by gat_heads({gat_heads}) when concat_heads=True"
            self.rel_msg_dim = out_dim // gat_heads
        else:
            self.rel_msg_dim = out_dim

        self.rel_upd = nn.Sequential(
            nn.Linear(self.rel_msg_dim, out_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(out_dim, out_dim)
        )

        if rel_embedding is None:
            self.rel_table = nn.Embedding(n_rel_total, out_dim)
        else:
            assert rel_embedding.num_embeddings >= n_rel_total
            assert rel_embedding.embedding_dim == out_dim
            if tie_rel_tables:
                self.rel_table = rel_embedding
                self.rgat.rel_emb = rel_embedding
            else:
                self.rel_table = nn.Embedding(n_rel_total, out_dim)
                with torch.no_grad():
                    self.rel_table.weight[:n_rel_total].copy_(rel_embedding.weight[:n_rel_total])
                    self.rgat.rel_emb.weight[:n_rel_total].copy_(rel_embedding.weight[:n_rel_total])

    def forward(self,
                X: torch.Tensor,
                edge_index: torch.Tensor,
                edge_type: torch.Tensor,
                sim1_edge: Optional[torch.Tensor] = None,
                L_rescaled: Optional[torch.sparse.FloatTensor] = None
                ) -> Tuple[torch.Tensor, torch.Tensor]:

        device = X.device

        if L_rescaled is not None:
            if L_rescaled.device != device:
                L_rescaled = L_rescaled.coalesce().to(device)
            else:
                L_rescaled = L_rescaled.coalesce()

        if edge_index.device != device:
            edge_index = edge_index.to(device)
        if edge_type.device != device:
            edge_type = edge_type.to(device)
        if sim1_edge is not None and sim1_edge.device != device:
            sim1_edge = sim1_edge.to(device)


        H = self.wavelet(X, L_rescaled)                           # [N, hidden_dim]

        Z_ent_raw, alpha_mean, msg_base = self.rgat(H, edge_index, sim1_edge)
        Z_ent = self.act(Z_ent_raw)                                # [N, out_dim]

        E = edge_type.size(0)
        R_tot = self.rel_table.num_embeddings
        assert R_tot > 0, "n_rel_total==0， 0， n_rel/n_rel_total/add_inverse_rel 。"

        if E > 0:
            et_min = int(edge_type.min().item())
            et_max = int(edge_type.max().item())
            assert 0 <= et_min and et_max < R_tot, \
                f"edge_type : [{et_min}, {et_max}]  R_tot={R_tot}"

        d_msg = self.rel_msg_dim
        assert msg_base.size(-1) == d_msg, f"msg_base  {d_msg}， {msg_base.size(-1)}"

        wmsg = alpha_mean.unsqueeze(-1) * msg_base

        rel_sum = torch.zeros(R_tot, d_msg, dtype=wmsg.dtype, device=device)
        rel_cnt = torch.zeros(R_tot, 1,     dtype=wmsg.dtype, device=device)

        if E > 0:
            rel_sum = rel_sum.index_add(0, edge_type, wmsg)
            rel_cnt = rel_cnt.index_add(0, edge_type, torch.ones(E, 1, dtype=wmsg.dtype, device=device))

        rel_avg = rel_sum / rel_cnt.clamp_min(1.0)                # [R_tot, d_msg]

        rel_init = self.rel_table.weight
        if rel_init.device != device:
            rel_init = rel_init.to(device)

        rel_delta = self.rel_upd(rel_avg)                         # [R_tot, out_dim]
        Z_rel = rel_init + rel_delta                              # [R_tot, out_dim]
        return Z_ent, Z_rel



