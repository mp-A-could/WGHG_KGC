import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class HyperbolicKGPred(nn.Module):
    """
    Prediction module using semantic gating with Lorentz + Givens + slerp.
    
    Args:
        ent_embed: [B, hind] head entity structural embedding
        rel_embed: [B, hind] relation structural embedding
        ent_label: [B, hind] true tail entity structural embedding
        sim: [B] or [B,1] semantic similarity (Sim_2)

    Hyperparameters:
        hind: structural embedding dimension
        d: direction space dimension (<= hind)
        P: number of Givens rotations
    """

    def __init__(self, hind: int, d: int = 128, P: int = None,
                 rho_max: float = 9.0, delta: float = 1e-3, eps: float = 1e-8,
                 givens_pairs=None):
        super().__init__()

        assert d <= hind, "d  hind"
        self.hind = hind
        self.d = d
        self.P = P if P is not None else 2 * d
        self.rho_max = float(rho_max)
        self.delta = float(delta)
        self.eps = float(eps)

        self.W_v = nn.Linear(hind, d)

        self.w_rho_ent = nn.Linear(hind, 1)

        self.w_rho_rel = nn.Linear(hind, 1)

        self.W_theta = nn.Linear(hind, self.P)


        self.gate_rho_a = nn.Parameter(torch.tensor([2.0]))
        self.gate_rho_b = nn.Parameter(torch.tensor([-0.5]))
        self.gate_phi_a = nn.Parameter(torch.tensor([2.0]))
        self.gate_phi_b = nn.Parameter(torch.tensor([-0.5]))

        self.eta_rho = nn.Parameter(torch.tensor([2.0]))

        self.pre_mlp = nn.Sequential(
            nn.Linear(hind, hind),
            nn.ReLU(),
            nn.Linear(hind, hind),
        )

        if givens_pairs is None:
            self.givens_pairs = self._default_pairs(self.d, self.P)  # List[(i,j)]
        else:
            assert len(givens_pairs) == self.P
            self.givens_pairs = givens_pairs

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W_v.weight);
        nn.init.zeros_(self.W_v.bias)
        nn.init.xavier_uniform_(self.w_rho_ent.weight);
        nn.init.constant_(self.w_rho_ent.bias, 0.5)
        nn.init.xavier_uniform_(self.w_rho_rel.weight);
        nn.init.zeros_(self.w_rho_rel.bias)
        nn.init.xavier_uniform_(self.W_theta.weight);
        nn.init.zeros_(self.W_theta.bias)
        nn.init.xavier_uniform_(self.pre_mlp[0].weight)
        nn.init.zeros_(self.pre_mlp[0].bias)
        nn.init.zeros_(self.pre_mlp[2].weight)
        nn.init.zeros_(self.pre_mlp[2].bias)

    @staticmethod
    def _default_pairs(d: int, P: int):
      
        pairs = []
        for m in range(P):
            i = m % d
            j = (i + 1) % d
            pairs.append((i, j))
        return pairs

    def _entity_to_polar(self, E: torch.Tensor):
     
        V = self.W_v(E)  # [B, d]
        u = V / (V.norm(dim=1, keepdim=True) + self.eps)
        rho = F.softplus(self.w_rho_ent(E))  # [B, 1], >=0
        return rho, u

    def _clip_rho_(self, rho: torch.Tensor):
        return torch.clamp(rho, 0.0, self.rho_max)

    def _apply_givens(self, u: torch.Tensor, thetas: torch.Tensor):
     
        out = u.clone()
        c = torch.cos(thetas)  # [B, P]
        s = torch.sin(thetas)  # [B, P]
        for m, (i, j) in enumerate(self.givens_pairs):
            ui = out[:, i].clone()
            uj = out[:, j].clone()
            out[:, i] = c[:, m] * ui - s[:, m] * uj
            out[:, j] = s[:, m] * ui + c[:, m] * uj
        out = out / (out.norm(dim=1, keepdim=True) + self.eps)
        return out

    def _slerp(self, u1: torch.Tensor, u2: torch.Tensor, alpha: torch.Tensor):
     
        dot = (u1 * u2).sum(dim=1, keepdim=True).clamp(-1.0 + self.eps, 1.0 - self.eps)  # [B,1]
        omega = torch.acos(dot)  # [B,1]
        sin_omega = torch.sin(omega)
        mask = (omega > self.delta).float()  # [B,1]
        s1 = torch.sin((1.0 - alpha) * omega) / (sin_omega + self.eps)
        s2 = torch.sin(alpha * omega) / (sin_omega + self.eps)
        u_slerp = s1 * u1 + s2 * u2
        u_lerp = (1.0 - alpha) * u1 + alpha * u2
        u_out = mask * u_slerp + (1.0 - mask) * u_lerp
        u_out = u_out / (u_out.norm(dim=1, keepdim=True) + self.eps)
        return u_out

    @staticmethod
    def _build_lorentz(rho: torch.Tensor, u: torch.Tensor):
      
        X0 = torch.cosh(rho)  # [B,1]
        Xs = torch.sinh(rho) * u  # [B,d]
        return torch.cat([X0, Xs], dim=1)

    @staticmethod
    def _lorentz_inner(X: torch.Tensor, Y: torch.Tensor):
      
        x0 = X[:, :1]
        xs = X[:, 1:]
        y0 = Y[:, :1]
        ys = Y[:, 1:]
        return -x0 * y0 + (xs * ys).sum(dim=1, keepdim=True)

    def forward(self, ent_embed: torch.Tensor, rel_embed: torch.Tensor,
                ent_label: torch.Tensor, sim: torch.Tensor):
       
        B = ent_embed.size(0)

        is_eval_mode = (sim.dim() == 2 and sim.size(1) > 1)  # [B, N_ent]

        if not is_eval_mode:
            if sim.dim() == 1:
                sim = sim.view(B, 1)  # [B,1]

        ent_embed = ent_embed + self.pre_mlp(ent_embed)
        rel_embed = rel_embed + self.pre_mlp(rel_embed)
        ent_label = ent_label + self.pre_mlp(ent_label)

        use_acosh_eval = False
        rho_h, u_h = self._entity_to_polar(ent_embed)  # [B,1], [B,d]
        rho_h = self._clip_rho_(rho_h)

        delta_rho = self.w_rho_rel(rel_embed)  # [B,1]
        thetas = self.W_theta(rel_embed)  # [B,P]
        thetas = math.pi * torch.tanh(thetas)  # (-pi, pi]

        if not is_eval_mode:

            g_rho = torch.sigmoid(self.gate_rho_a * sim + self.gate_rho_b)  # [B,1]
            g_phi = torch.sigmoid(self.gate_phi_a * sim + self.gate_phi_b)  # [B,1]

            rho_prime = rho_h + delta_rho  * g_rho
            rho_prime = torch.clamp(rho_prime, 0.0, self.rho_max)

            u_rot = self._apply_givens(u_h, thetas)  # Q_r u_h
            u_prime = self._slerp(u_h, u_rot, g_phi)
        else:

            u_rot = self._apply_givens(u_h, thetas)  # [B,d]
            pass

        if ent_label.size(0) == B and not is_eval_mode:
            rho_t, u_t = self._entity_to_polar(ent_label)  # [B,1], [B,d]
            rho_t = self._clip_rho_(rho_t)
            Xh = self._build_lorentz(rho_prime, u_prime)
            Xt = self._build_lorentz(rho_t, u_t)
            score = self._lorentz_inner(Xh, Xt)  # [B,1]
            #score =rho_prime
            return {
                "score": score.squeeze(-1),  # [B]
                "rho_h": rho_h, "rho_t": rho_t, "rho_prime": rho_prime,
                "u_h": u_h, "u_t": u_t, "u_rot": u_rot, "u_prime": u_prime,
                "g_rho": g_rho, "g_phi": g_phi,
                "delta_rho": delta_rho, "thetas": thetas,
                "Xh": Xh, "Xt": Xt
            }
        else:
            N = ent_label.size(0)  # N_ent

            #
            rho_all, u_all = self._entity_to_polar(ent_label)  # [N,1], [N,d]
            rho_all = self._clip_rho_(rho_all)


            g_rho_mat = torch.sigmoid(self.gate_rho_a * sim + self.gate_rho_b)  # [B, N]
            g_phi_mat = torch.sigmoid(self.gate_phi_a * sim + self.gate_phi_b)  # [B, N]


            rho_prime_mat = rho_h + delta_rho * g_rho_mat  # [B, 1] + [B, 1] + [B, N] -> [B, N]
            rho_prime_mat = torch.clamp(rho_prime_mat, 0.0, self.rho_max)

            u_h_exp = u_h.unsqueeze(1).expand(B, N, -1)  # [B, N, d]
            u_rot_exp = u_rot.unsqueeze(1).expand(B, N, -1)  # [B, N, d]
            g_phi_exp = g_phi_mat.unsqueeze(-1)  # [B, N, 1]

            u_h_flat = u_h_exp.reshape(B * N, -1)  # [B*N, d]
            u_rot_flat = u_rot_exp.reshape(B * N, -1)  # [B*N, d]
            g_phi_flat = g_phi_exp.reshape(B * N, 1)  # [B*N, 1]

            u_prime_flat = self._slerp(u_h_flat, u_rot_flat, g_phi_flat)  # [B*N, d]
            u_prime_mat = u_prime_flat.reshape(B, N, -1)  # [B, N, d]

            C_mat = torch.cosh(rho_prime_mat)  # [B, N]
            D_all = torch.cosh(rho_all).squeeze(-1)  # [N]
            time_contrib = -C_mat * D_all.unsqueeze(0)  # [B, N]

            A_mat = torch.sinh(rho_prime_mat).unsqueeze(-1) * u_prime_mat  # [B, N, d]
            B_all = torch.sinh(rho_all) * u_all  # [N, d]

            space_contrib = (A_mat * B_all.unsqueeze(0)).sum(dim=-1)  # [B, N]

            S_inner = time_contrib + space_contrib  # [B, N]

            if use_acosh_eval:
                Z = torch.clamp(-S_inner, min=1.0 + self.eps)
                score_mat = -torch.acosh(Z)  # [B, N]
            else:
                score_mat = S_inner  # [B, N]
            g_rho_mean = g_rho_mat.mean(dim=1, keepdim=True)  # [B, N] -> [B, 1]
            g_phi_mean = g_phi_mat.mean(dim=1, keepdim=True)  # [B, N] -> [B, 1]
            rho_prime_mean = rho_prime_mat.mean(dim=1, keepdim=True)  # [B, N] -> [B, 1]
            return {
                "score": score_mat,  # [B, N_ent]
                "rho_h": rho_h, "rho_prime": rho_prime_mean,  # [B, 1]
                "u_h": u_h, "u_rot": u_rot, "u_prime": u_prime_mat.mean(dim=1),  # [B, d]
                "g_rho": g_rho_mean, "g_phi": g_phi_mean,  # [B, 1]
                "delta_rho": delta_rho, "thetas": thetas,
            }
